#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright   2026  (author: Ibrahim Almajai)
# Apache 2.0

"""
This file computes avhubert features of the grid dataset.
It looks for manifests in the directory data/manifests and writes the
features and cut manifests to --feats-dir (prepare.sh: data/avhubert_mf_l8
for the mean-face / layer-8 system).
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from fairseq import checkpoint_utils
from lhotse import CutSet, LilcomChunkyWriter, NumpyHdf5Writer
from lhotse.features.io import KaldiWriter
from lhotse.recipes.utils import read_manifests_if_cached


# CLI arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract AV-HuBERT visual features from video recordings."
    )
    parser.add_argument(
        "--avhubert-code-dir",
        type=Path,
        required=True,
        help="Path to the AV-HuBERT source directory (added temporarily to sys.path).",
    )
    parser.add_argument(
        "--avhubert-ckpt",
        type=Path,
        required=True,
        help="Path to the AV-HuBERT pretrained checkpoint (.pt file).",
    )
    parser.add_argument(
        "--dlib-predictor",
        type=Path,
        default=Path("download/dlib/shape_predictor_68_face_landmarks.dat"),
        help="Path to the dlib 68-point landmark model. " "Default: %(default)s",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Number of encoder layers to keep (0-indexed upper bound). "
        "Default: %(default)s",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=6,
        help="Number of parallel worker processes. Default: %(default)s",
    )
    parser.add_argument(
        "--mouth-w",
        type=int,
        default=64,
        help="Mouth crop width in pixels. Default: %(default)s",
    )
    parser.add_argument(
        "--mouth-h",
        type=int,
        default=64,
        help="Mouth crop height in pixels. Default: %(default)s",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="Run face detection every N frames; reuse previous landmarks otherwise. "
        "Default: %(default)s",
    )
    parser.add_argument(
        "--roi-mode",
        choices=["centroid", "meanface"],
        default="meanface",
        help="Mouth-ROI geometry. 'centroid': crop --mouth-w/h around the "
        "mouth centre from dlib landmarks (the original method). "
        "'meanface': canonical AV-HuBERT preprocessing: landmarks "
        "warped onto a mean face (download/20words_mean_face.npy) "
        "before a fixed mouth crop (local/meanface_roi.py), cached as "
        "<clip>.mfroi.npz next to the clip and reused when present. "
        "Default: %(default)s",
    )
    parser.add_argument(
        "--feature-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Storage precision of the written features (--storage hdf5 "
        "only). float16 halves the disk footprint; for the layer-8 phone "
        "system it decodes the test set to identical hypotheses. "
        "Default: %(default)s",
    )
    parser.add_argument(
        "--storage",
        choices=["hdf5", "lilcom", "kaldi8"],
        default="kaldi8",
        help="Feature storage. 'kaldi8': Kaldi ark with 8-bit per-column "
        "compression (KaldiWriter, compression_method=2), a quarter of "
        "float32 (2.0 against 7.1 GB for GRID layer 8); the recipe trained "
        "and decoded on it matches uncompressed features (three seeds). "
        "'hdf5': uncompressed (NumpyHdf5Writer). 'lilcom': lossy lilcom "
        "compression (LilcomChunkyWriter, as icefall uses for fbank), "
        "about 10%% larger than kaldi8. Default: %(default)s",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Clips per AV-HuBERT forward pass in each worker; shorter clips "
        "are padded and masked. 16 is 1.9x faster than 1 on GRID, 2.9x "
        "with --autocast. Default: %(default)s",
    )
    parser.add_argument(
        "--autocast",
        action="store_true",
        help="Run AV-HuBERT under fp16 autocast (CUDA only); it speeds up "
        "batched extraction only. With --batch-size 16 it changes 2 of the "
        "4,000 GRID test hypotheses and no WER. Features are still written "
        "at --feature-dtype.",
    )
    parser.add_argument(
        "--feats-dir",
        type=Path,
        required=True,
        help="Directory where the feature files will be written.",
    )
    return parser.parse_args()


# AV-HuBERT imports (scoped to avoid polluting the global namespace)
@contextlib.contextmanager
def _avhubert_on_path(path: Path):
    """Temporarily add AV-HuBERT source directory to sys.path."""
    str_path = str(path)
    sys.path.insert(0, str_path)
    try:
        yield
    finally:
        sys.path.remove(str_path)


def load_globals_avbubert(args: argparse.Namespace):
    """
    Initialise and return all shared resources derived from ``args``.

    Returns
    -------
    dict with keys: avhubert_utils, detector, predictor, device, model, transform
    """
    # AV-HuBERT imports
    if not args.avhubert_code_dir.exists():
        raise FileNotFoundError(
            f"AV-HuBERT code directory not found: {args.avhubert_code_dir}"
        )

    with _avhubert_on_path(args.avhubert_code_dir):
        from avhubert.utils import Compose, Normalize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Using device: {device}")

    if not args.avhubert_ckpt.exists():
        raise FileNotFoundError(f"AV-HuBERT checkpoint not found: {args.avhubert_ckpt}")

    # A fine-tuned (seq2seq) checkpoint stores the path of the pretrained
    # model it started from; if that path no longer exists, look for the same
    # file name next to the checkpoint.
    state = checkpoint_utils.load_checkpoint_to_cpu(str(args.avhubert_ckpt))
    model_cfg = state["cfg"]["model"]
    w2v = model_cfg["w2v_path"] if "w2v_path" in model_cfg else None
    if w2v and not os.path.exists(w2v):
        state["cfg"]["model"]["w2v_path"] = os.path.join(
            os.path.dirname(os.path.abspath(str(args.avhubert_ckpt))),
            os.path.basename(w2v),
        )
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [str(args.avhubert_ckpt)], state=state
    )
    del state
    model = models[0]
    if hasattr(model, "encoder") and hasattr(model.encoder, "w2v_model"):
        # fine-tuned checkpoint: read features from its AV-HuBERT encoder
        logging.info("Fine-tuned checkpoint: extracting from encoder.w2v_model")
        model = model.encoder.w2v_model
    model.encoder.layers = model.encoder.layers[: args.layer]
    model.to(device).eval()
    logging.info(
        f"Loaded AV-HuBERT checkpoint: {args.avhubert_ckpt} "
        f"(layers 0–{args.layer - 1}, device: {device})"
    )

    # Image transform
    transform = Compose(
        [
            Normalize(0.0, 255.0),
            Normalize(task.cfg.image_mean, task.cfg.image_std),
        ]
    )

    return dict(
        device=device,
        model=model,
        transform=transform,
    )


def load_globals_dlib(args: argparse.Namespace):
    """
    Initialise and return all shared resources derived from ``args``.

    Returns
    -------
    dict with keys: avhubert_utils, detector, predictor, device, model, transform
    """

    # Dlib
    if not args.dlib_predictor.exists():
        raise FileNotFoundError(
            f"dlib landmark model not found: {args.dlib_predictor}\n"
            "Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(args.dlib_predictor))

    return dict(
        detector=detector,
        predictor=predictor,
    )


# Mean-face ROI settings: the reference face (from
# https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks,
# fetched by prepare.sh stage 0) and the height frames are downscaled to for
# face detection only, as in lipreader.py. Together they reproduce the .mfroi.npz
# caches the paper's features were extracted from.
MEAN_FACE_PATH = Path("download/20words_mean_face.npy")
DETECT_HEIGHT = 480
_MEAN_FACE: dict = {}


def extract_features_batch(
    frames_list: list, device, model, layer, autocast: bool = False
) -> list:
    """AV-HuBERT features for several clips' normalised ROI frames (each
    (T, 88, 88)) in one forward pass, padded to the longest clip and masked.
    Returns one (T, D) float32 array per clip."""
    lens = [len(f) for f in frames_list]
    T = max(lens)
    x = torch.zeros(len(frames_list), 1, T, *np.shape(frames_list[0])[1:])
    for i, f in enumerate(frames_list):
        x[i, 0, : lens[i]] = torch.as_tensor(np.asarray(f), dtype=torch.float32)
    padding_mask = torch.arange(T)[None, :] >= torch.tensor(lens)[:, None]
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.float16, enabled=autocast and device.type == "cuda"
    ):
        features, _ = model.extract_finetune(
            source={"video": x.to(device), "audio": None},
            padding_mask=padding_mask.to(device),
            output_layer=layer,
        )
    features = features.float().cpu().numpy()
    return [features[i, :n] for i, n in enumerate(lens)]


def extract_features_from_visual(
    video: str,
    dlib_detector,
    dlib_predictor,
    device,
    model,
    transform,
    layer,
    mouth_w: int = 64,
    mouth_h: int = 64,
    detect_every: int = 1,
    roi_mode: str = "centroid",
    autocast: bool = False,
) -> np.ndarray | None:
    """
    Extract AV-HuBERT features from the mouth ROI of a video.

    Landmarks are cached as ``<basename>.landmarks.npz`` and mouth frames as
    ``<basename>.mouth_frames.npz``, both reused on subsequent calls to skip
    re-detection and re-cropping.

    Parameters
    ----------
    video : str
        Path to the input video file.
    mouth_w, mouth_h : int
        Crop dimensions (pixels) around the mouth centre. Defaults: 64 × 64.
    detect_every : int
        Run face detection every N frames; intermediate frames reuse the
        previous landmark. Default: 1 (every frame).

    Returns
    -------
    np.ndarray of shape ``(T, D)``
        AV-HuBERT feature sequence, or ``None`` if detection failed or the
        video has fewer than 74 frames.
    """
    frames = load_roi_frames(
        video,
        dlib_detector,
        dlib_predictor,
        transform,
        mouth_w,
        mouth_h,
        detect_every,
        roi_mode,
    )
    if frames is None:
        return None
    return extract_features_batch([frames], device, model, layer, autocast)[0]


def load_roi_frames(
    video: str,
    dlib_detector,
    dlib_predictor,
    transform,
    mouth_w: int = 64,
    mouth_h: int = 64,
    detect_every: int = 1,
    roi_mode: str = "centroid",
):
    """Normalised mouth-ROI frames of a video, (T, 88, 88), using and
    filling the per-clip caches; ``None`` if detection failed or the video
    has fewer than 74 frames."""
    MIN_FRAMES = 74
    ROI_SIZE = (88, 88)
    MOUTH_LEFT, MOUTH_RIGHT = 48, 54  # dlib 68-point landmark indices

    video_path = Path(video)
    landmarks_file = video_path.with_suffix(".landmarks.npz")
    mouth_frames_file = video_path.with_suffix(".mouth_frames.npz")

    if roi_mode == "meanface":
        # Mean-face-aligned ROIs (temporal smoothing -> similarity warp onto
        # the mean face -> 96x96 mouth crop -> 88x88 centre crop; see
        # local/meanface_roi.py). Stored raw as uint8, unlike
        # .mouth_frames.npz which already holds transform()ed floats, so
        # normalise here.
        mfroi_file = video_path.with_suffix(".mfroi.npz")
        if not mfroi_file.exists():
            from meanface_roi import extract_meanface_frames, load_mean_face

            if "mean_face" not in _MEAN_FACE:
                _MEAN_FACE["mean_face"] = load_mean_face(MEAN_FACE_PATH)
            landmarks = _detect_landmarks(
                video_path,
                dlib_detector,
                dlib_predictor,
                detect_every,
                detect_height=DETECT_HEIGHT,
            )
            if landmarks is None:
                logging.warning(f"Skipping {video}: no face detected.")
                return None
            roi = extract_meanface_frames(
                video_path, landmarks, _MEAN_FACE["mean_face"]
            )
            np.savez_compressed(mfroi_file, roi=roi, fps=25.0)
        frames = transform(np.load(mfroi_file)["roi"])
    # Mouth frames (or cache load)
    elif mouth_frames_file.exists():
        frames = np.load(mouth_frames_file)["frames"]
        if frames.dtype == np.uint8:
            # Caches written before 2026-10 hold raw pixels.
            frames = transform(frames.astype(np.float32))
        frames = list(frames)
        logging.info(f"Loaded cached mouth frames from {mouth_frames_file}.")
    else:
        # Landmarks (or cache load)
        if landmarks_file.exists():
            landmarks = np.load(landmarks_file)["landmarks"]
            logging.info(f"Loaded cached landmarks from {landmarks_file}.")
        else:
            landmarks = _detect_landmarks(
                video_path, dlib_detector, dlib_predictor, detect_every
            )
            if landmarks is None:
                return None
            np.savez_compressed(landmarks_file, landmarks=landmarks.astype(np.int16))
            logging.info(f"Saved landmarks to {landmarks_file}.")

        # ROI extraction
        frames = _extract_mouth_frames(
            video_path, landmarks, mouth_w, mouth_h, ROI_SIZE, MOUTH_LEFT, MOUTH_RIGHT
        )
        if len(frames) < MIN_FRAMES:
            logging.warning(
                f"Skipping {video}: only {len(frames)} frames (min {MIN_FRAMES})."
            )
            return None

        frames = transform(np.stack(frames))

        np.savez_compressed(mouth_frames_file, frames=frames)
        logging.info(f"Saved mouth frames to {mouth_frames_file}.")

    if len(frames) < MIN_FRAMES:
        logging.warning(
            f"Skipping {video}: only {len(frames)} frames (min {MIN_FRAMES})."
        )
        return None
    return frames


def _open_video(path: Path) -> cv2.VideoCapture:
    """Open a video file, raising ``IOError`` if it cannot be read."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    return cap


def _detect_landmarks(
    video_path: Path,
    dlib_detector,
    dlib_predictor,
    detect_every: int,
    detect_height: int | None = None,
    fallback_landmarker=None,
) -> np.ndarray | None:
    """
    Run dlib face + landmark detection on every ``detect_every``-th frame.

    If ``detect_height`` is set and frames are taller, the (slow, ~linear in
    pixels) face *detection* runs on a copy downscaled to that height, while
    the landmark predictor still runs at full resolution for precision.
    Landmarks are always in original-frame coordinates.

    If ``fallback_landmarker`` is given (a callable taking the BGR frame and
    returning a (68, 2) landmark array or None), it is tried per frame only
    when the fast HOG detector finds nothing; meant for detectors more robust to
    glasses, tight framing and mild pose (e.g. MediaPipe; see lipreader.py).

    Returns an ``(T, 68, 2)`` int array, or ``None`` if no face was detected
    on the first keyed frame.
    """
    cap = _open_video(video_path)
    landmarks, last_lm = [], None

    try:
        for frame_idx in itertools.count():
            success, image = cap.read()
            if not success:
                break

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if frame_idx % detect_every == 0:
                scale = 1.0
                small = gray
                if detect_height and gray.shape[0] > detect_height:
                    scale = detect_height / gray.shape[0]
                    small = cv2.resize(
                        gray, (round(gray.shape[1] * scale), detect_height)
                    )
                faces = dlib_detector(small)
                if faces:
                    face = faces[0]
                    if scale != 1.0:
                        face = dlib.rectangle(
                            round(face.left() / scale),
                            round(face.top() / scale),
                            round(face.right() / scale),
                            round(face.bottom() / scale),
                        )
                    shape = dlib_predictor(gray, face)
                    last_lm = np.array([[p.x, p.y] for p in shape.parts()])
                elif fallback_landmarker is not None:
                    # HOG found nothing: the fallback supplies landmarks
                    # directly from the full-resolution BGR frame.
                    last_lm = fallback_landmarker(image)
                    if last_lm is None:
                        logging.warning(
                            f"No face detected in {video_path} at frame {frame_idx}."
                        )
                        return None
                else:
                    logging.warning(
                        f"No face detected in {video_path} at frame {frame_idx}."
                    )
                    return None

            landmarks.append(last_lm)
    finally:
        cap.release()

    return np.array(landmarks) if landmarks else None


def _extract_mouth_frames(
    video_path: Path,
    landmarks: np.ndarray,
    mouth_w: int,
    mouth_h: int,
    roi_size: tuple[int, int],
    idx_left: int,
    idx_right: int,
) -> list[np.ndarray]:
    """
    Crop and resize the mouth region for every frame using pre-computed landmarks.

    Returns a list of grayscale ``roi_size`` uint8 arrays.
    """
    h_w, h_h = mouth_w // 2, mouth_h // 2
    cap = _open_video(video_path)
    frames = []

    try:
        for frame_idx in itertools.count():
            success, image = cap.read()
            if not success:
                break

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lm = landmarks[frame_idx]

            cx = int((lm[idx_left, 0] + lm[idx_right, 0]) / 2)
            cy = int((lm[idx_left, 1] + lm[idx_right, 1]) / 2)

            x1 = max(0, cx - h_w)
            y1 = max(0, cy - h_h)
            roi = gray[y1 : cy + h_h, x1 : cx + h_w]

            frames.append(cv2.resize(roi, roi_size))
    finally:
        cap.release()

    return frames


_worker_globals: dict = {}


def _worker_init(args: argparse.Namespace) -> None:
    """Called once per worker process to load model and detector."""
    global _worker_globals
    _worker_globals = load_globals_avbubert(args)
    _worker_globals.update(load_globals_dlib(args))
    _worker_globals["feature_dtype"] = np.dtype(
        getattr(args, "feature_dtype", "float32")
    )
    _worker_globals["batch_size"] = getattr(args, "batch_size", 1)
    _worker_globals["storage"] = getattr(args, "storage", "kaldi8")
    _worker_globals["autocast"] = getattr(args, "autocast", False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info(f"Worker {os.getpid()} initialised.")


def process_worker(args: tuple) -> list:
    """
    Extract visual features for a subset of recordings and write them to a
    dedicated HDF5 file for this worker.

    Parameters
    ----------
    args : tuple
        ``(worker_id, recordings_subset, supervisions_subset, feats_dir, partition)``

    Returns
    -------
    list[Cut]
        Successfully processed cuts with stored feature references.

    Notes
    -----
    - Skips recordings where feature extraction returns ``None``.
    - Duplicates the first frame if only 74 frames are returned (expects 75).
    - All cuts are assigned a fixed duration of 3.0 s at 25 fps.
    """
    (
        worker_id,
        recordings_subset,
        supervisions_subset,
        feats_dir,
        partition,
        layer,
        roi_mode,
    ) = args
    # Access resources initialised by _worker_init
    model = _worker_globals["model"]
    transform = _worker_globals["transform"]
    detector = _worker_globals["detector"]
    predictor = _worker_globals["predictor"]
    device = _worker_globals["device"]

    FIXED_DURATION: float = 3.0
    FRAME_SHIFT: float = 0.04
    EXPECTED_FRAMES: int = 75
    TEMPORAL_DIM: int = 0

    storage = _worker_globals["storage"]
    stem = feats_dir / f"{partition}_avhubert_{worker_id:02d}"
    if storage == "hdf5":
        writer_cm = NumpyHdf5Writer(stem.with_suffix(".h5"))
        dtype = _worker_globals["feature_dtype"]
    elif storage == "lilcom":
        writer_cm = LilcomChunkyWriter(stem.with_suffix(".lca"))
        dtype = np.float32
    else:
        writer_cm = KaldiWriter(stem, compression_method=2)
        dtype = np.float32
    video_cuts = []
    sup_by_rec = {s.recording_id: s for s in supervisions_subset}
    pending = []  # (recording, supervision, frames)

    def flush(writer):
        if not pending:
            return
        try:
            feats_list = extract_features_batch(
                [f for _, _, f in pending],
                device,
                model,
                layer,
                _worker_globals["autocast"],
            )
        except Exception as e:
            logging.error(
                f"Worker {worker_id} - error extracting "
                f"{[r.id for r, _, _ in pending]}: {e}",
                exc_info=True,
            )
            pending.clear()
            return
        for (recording, supervision, _), feats in zip(pending, feats_list):
            if feats.shape[0] == EXPECTED_FRAMES - 1:
                feats = np.vstack([feats[[0]], feats])
                logging.warning(
                    f"Worker {worker_id} - {recording.id}: only "
                    f"{EXPECTED_FRAMES - 1} frames found; duplicated the "
                    f"first frame to reach {EXPECTED_FRAMES}."
                )

            video_cut = recording.to_cut()
            video_cut.duration = FIXED_DURATION
            video_cut.channel = [0]

            supervision.duration = FIXED_DURATION
            video_cut.supervisions = [supervision]

            stored_key = writer.store_array(
                key=video_cut.id,
                value=feats.astype(dtype),
                frame_shift=FRAME_SHIFT,
                temporal_dim=TEMPORAL_DIM,
            )
            video_cut.video_features = stored_key
            video_cuts.append(video_cut)
        pending.clear()

    with writer_cm as writer:
        for recording in recordings_subset:
            supervision = sup_by_rec.get(recording.id)
            if supervision is None:
                logging.warning(
                    f"Worker {worker_id} - no supervision for {recording.id}, skipping."
                )
                continue
            try:
                frames = load_roi_frames(
                    recording.sources[0].source,
                    detector,
                    predictor,
                    transform,
                    roi_mode=roi_mode,
                )
                if frames is None:
                    logging.warning(
                        f"Worker {worker_id} - skipping {recording.id}: "
                        "feature extraction returned None."
                    )
                    continue
            except Exception as e:
                logging.error(
                    f"Worker {worker_id} - error processing {recording.id}: {e}",
                    exc_info=True,
                )
                continue
            pending.append((recording, supervision, frames))
            if len(pending) >= _worker_globals["batch_size"]:
                flush(writer)
        flush(writer)

    logging.info(
        f"Worker {worker_id} finished, wrote {len(video_cuts)} cuts to {stem} ({storage})"
    )
    return video_cuts


def compute_avhubert_grid():
    # ... manifest loading ...
    src_dir = Path("data/manifests")
    feats_dir = Path(f"{args.feats_dir}")
    try:
        feats_dir.mkdir(parents=True)
    except FileExistsError:
        logging.info(f"Warning: '{feats_dir}' is an existing path.")

    dataset_parts = (
        "test",
        "train",
    )
    prefix = "grid"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    for partition, m in manifests.items():
        logging.info(f"\nProcessing {partition} with {args.n_workers} workers")

        recordings = list(m["recordings"])
        supervisions = list(m["supervisions"])

        # Split into roughly equal chunks
        chunk_size = len(recordings) // args.n_workers + 1
        tasks = []
        for i in range(args.n_workers):
            start = i * chunk_size
            end = min(start + chunk_size, len(recordings))
            if start >= end:
                break
            tasks.append(
                (
                    i,
                    recordings[start:end],
                    supervisions[start:end],
                    feats_dir,
                    partition,
                    args.layer,
                    args.roi_mode,
                )
            )

        all_cuts = []

        with ProcessPoolExecutor(
            max_workers=args.n_workers, initializer=_worker_init, initargs=(args,)
        ) as ex:
            results = ex.map(process_worker, tasks)
            for worker_cuts in results:
                all_cuts.extend(worker_cuts)

        # Final CutSet
        cut_set = CutSet.from_cuts(all_cuts)
        cut_set.to_file(feats_dir / f"grid_cuts_{partition}.jsonl.gz")

        logging.info(
            f"Done {partition} → {len(all_cuts)} cuts, stored in {args.n_workers} {args.storage} files"
        )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    compute_avhubert_grid()
