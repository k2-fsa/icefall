#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright   2026  (author: Ibrahim Almajai)
# Apache 2.0

"""
This file computes avhubert features of the grid dataset.
It looks for manifests in the directory data/manifests.

The generated avhubert features are saved in data/avhubert.
"""
from __future__ import annotations
import dlib
import torch.nn.functional as F
import cv2
from fairseq import checkpoint_utils

import logging
import os
import sys
from pathlib import Path
import torch
from lhotse import CutSet, NumpyHdf5Writer
from lhotse.recipes.utils import read_manifests_if_cached
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import logging
import torch.multiprocessing as mp
import itertools
import contextlib
import argparse


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
        help="Path to the dlib 68-point landmark model. "
             "Default: %(default)s",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=9,
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
        "--feats-dir",
        type=Path,
        required=True,
        help="Directory where output .h5 feature files will be written.",
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


def load_globals(args: argparse.Namespace):
    """
    Initialise and return all shared resources derived from ``args``.

    Returns
    -------
    dict with keys: avhubert_utils, detector, predictor, device, model, transform
    """
    # AV-HuBERT imports 
    if not args.avhubert_code_dir.exists():
        raise FileNotFoundError(f"AV-HuBERT code directory not found: {args.avhubert_code_dir}")

    with _avhubert_on_path(args.avhubert_code_dir):
        from avhubert.utils import Compose, Normalize

    # Dlib 
    if not args.dlib_predictor.exists():
        raise FileNotFoundError(
            f"dlib landmark model not found: {args.dlib_predictor}\n"
            "Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )

    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(args.dlib_predictor))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if not args.avhubert_ckpt.exists():
        raise FileNotFoundError(f"AV-HuBERT checkpoint not found: {args.avhubert_ckpt}")

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([str(args.avhubert_ckpt)])
    model = models[0]
    model.encoder.layers = model.encoder.layers[:args.layer]
    model.to(device).eval()
    logging.info(
        f"Loaded AV-HuBERT checkpoint: {args.avhubert_ckpt} "
        f"(layers 0–{args.layer - 1}, device: {device})"
    )

    # Image transform 
    transform = Compose([
        Normalize(0.0, 255.0),
        Normalize(task.cfg.image_mean, task.cfg.image_std),
    ])

    return dict(
        detector=detector,
        predictor=predictor,
        device=device,
        model=model,
        transform=transform,
    )


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
    MIN_FRAMES = 74
    ROI_SIZE = (88, 88)
    MOUTH_LEFT, MOUTH_RIGHT = 48, 54  # dlib 68-point landmark indices

    video_path = Path(video)
    landmarks_file = video_path.with_suffix(".landmarks.npz")
    mouth_frames_file = video_path.with_suffix(".mouth_frames.npz")

    # Mouth frames (or cache load)
    if mouth_frames_file.exists():
        frames = list(np.load(mouth_frames_file)["frames"])
        logging.info(f"Loaded cached mouth frames from {mouth_frames_file}.")
    else:
        # Landmarks (or cache load) 
        if landmarks_file.exists():
            landmarks = np.load(landmarks_file)["landmarks"]
            logging.info(f"Loaded cached landmarks from {landmarks_file}.")
        else:
            landmarks = _detect_landmarks(video_path, dlib_detector,  dlib_predictor ,detect_every)
            if landmarks is None:
                return None
            np.savez_compressed(landmarks_file, landmarks=landmarks.astype(np.int16))
            logging.info(f"Saved landmarks to {landmarks_file}.")

        # ROI extraction
        frames = _extract_mouth_frames(
            video_path, landmarks, mouth_w, mouth_h, ROI_SIZE, MOUTH_LEFT, MOUTH_RIGHT
        )
        if len(frames) < MIN_FRAMES:
            logging.warning(f"Skipping {video}: only {len(frames)} frames (min {MIN_FRAMES}).")
            return None

        np.savez_compressed(mouth_frames_file, frames=np.array(frames, dtype=np.uint8))
        logging.info(f"Saved mouth frames to {mouth_frames_file}.")

    if len(frames) < MIN_FRAMES:
        logging.warning(f"Skipping {video}: only {len(frames)} frames (min {MIN_FRAMES}).")
        return None
   
    frames_np = transform(np.float32(np.stack(frames)))
    tensor = torch.from_numpy(frames_np).unsqueeze(0).unsqueeze(0).to(device)
     # AV-HuBERT feature extraction
    with torch.no_grad():
        features, _ = model.extract_finetune(
            source={"video": tensor, "audio": None},
            padding_mask=None,
            output_layer=layer,
        )

    return features.squeeze(0).cpu().numpy()


def _open_video(path: Path) -> cv2.VideoCapture:
    """Open a video file, raising ``IOError`` if it cannot be read."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    return cap


def _detect_landmarks(video_path: Path, dlib_detector,  dlib_predictor, detect_every: int) -> np.ndarray | None:
    """
    Run dlib face + landmark detection on every ``detect_every``-th frame.

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
                faces = dlib_detector(gray)
                if not faces:
                    logging.warning(f"No face detected in {video_path} at frame {frame_idx}.")
                    return None
                shape = dlib_predictor(gray, faces[0])
                last_lm = np.array([[p.x, p.y] for p in shape.parts()])

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
    _worker_globals = load_globals(args)
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
    worker_id, recordings_subset, supervisions_subset, feats_dir, partition, layer = args
    # Access resources initialised by _worker_init
    model     = _worker_globals["model"]
    transform = _worker_globals["transform"]
    detector  = _worker_globals["detector"]
    predictor = _worker_globals["predictor"]
    device    = _worker_globals["device"]
    
    FIXED_DURATION: float = 3.0
    FRAME_SHIFT: float = 0.04
    EXPECTED_FRAMES: int = 75
    TEMPORAL_DIM: int = 0

    h5_path = feats_dir / f"{partition}_avhubert_{worker_id:02d}.h5"
    video_cuts = []
    sup_by_rec = {s.recording_id: s for s in supervisions_subset}
    with NumpyHdf5Writer(h5_path) as writer:
        for recording in recordings_subset:
            supervision = sup_by_rec.get(recording.id)
            if supervision is None:
                logging.warning(f"Worker {worker_id} - no supervision for {recording.id}, skipping.")
                continue
            try:
                feats = extract_features_from_visual(recording.sources[0].source, detector, \
                    predictor, device, model, transform, layer)

                if feats is None:
                    logging.warning(
                        f"Worker {worker_id} - skipping {recording.id}: "
                        "feature extraction returned None."
                    )
                    continue

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
                    value=feats,
                    frame_shift=FRAME_SHIFT,
                    temporal_dim=TEMPORAL_DIM,
                )
                video_cut.video_features = stored_key
                video_cuts.append(video_cut)

            except Exception as e:
                logging.error(
                    f"Worker {worker_id} - error processing {recording.id}: {e}",
                    exc_info=True,
                )

    logging.info(
        f"Worker {worker_id} finished — wrote {len(video_cuts)} cuts to {h5_path}"
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
            tasks.append((
                i,
                recordings[start:end],
                supervisions[start:end],
                feats_dir,
                partition,
                args.layer, 
                
            ))

        all_cuts = []

        with ProcessPoolExecutor(max_workers=args.n_workers, initializer=_worker_init, initargs=(args,)) as ex:
            results = ex.map(process_worker, tasks)
            for worker_cuts in results:
                all_cuts.extend(worker_cuts)

        # Final CutSet
        cut_set = CutSet.from_cuts(all_cuts)
        cut_set.to_file(feats_dir / f"grid_cuts_{partition}.jsonl.gz")

        logging.info(f"Done {partition} → {len(all_cuts)} cuts, stored in {args.n_workers} .h5 files")
        

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = parse_args()
    mp.set_start_method('spawn', force=True)    
    compute_avhubert_grid()
