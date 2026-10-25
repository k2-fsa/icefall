#!/usr/bin/env python3
# Copyright   2026  (author: Ibrahim Almajai)
# Apache 2.0

"""
Lipreading inference for the GRID conformer_ctc2 recipe: one video in, the
recognised GRID sentence out.

Pipeline (all stages reused from the recipe):

    video ->  dlib face + 68 landmarks        (local/compute_avhubert_grid.py)
          ->  mouth ROI, 88x88 grey (mean-face aligned or centroid crop)
          ->  AV-HuBERT feature extract        (extract_finetune, output_layer=L)
          ->  Conformer-CTC forward
          ->  CTC greedy decode, or HLG one-best

Also usable as a command-line tool:

    ./conformer_ctc2/lipreader.py \
        --checkpoint conformer_ctc2/exp-phone-l8/pretrained.pt \
        --roi-mode meanface --layer 8 \
        --tokens data/lang_phone/tokens_model.txt \
        --method 1best \
        --HLG data/lang_phone/HLG.pt \
        --words-file data/lang_phone/words.txt \
        --avhubert-ckpt download/avhubert-ckpts/base_vox_iter5.pt \
        download/grid-corpus/s33/pgwh8n.mpg

The defaults (encoder-dim 128, 6 enc / 3 dec layers, layer 8, mean-face ROIs)
match the released phone/layer-8 model (pretrained.pt on Hugging Face, see
RESULTS.md). tokens_model.txt is tokens.txt without the #0/#1 disambiguation
symbols plus <sos/eos>, matching the model's output layer. Clips at other
frame rates (e.g. 30/60 fps phone video) are resampled to 25 fps with ffmpeg
before feature extraction.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Bundled py3.8-compatible MediaPipe (the env's own build is broken on 3.8),
# used as a fallback landmarker. It must be imported BEFORE anything
# that pulls in the env's protobuf 5.x (fairseq, tensorboard): MediaPipe
# needs the bundled protobuf<4 and whichever protobuf loads first wins.
# Nothing else in this process needs protobuf 5.
MEDIAPIPE_DIR = Path(__file__).resolve().parent.parent / "download/mediapipe-py38"
try:
    sys.path.insert(0, str(MEDIAPIPE_DIR))
    import mediapipe as _mediapipe  # noqa: E402
except Exception:  # missing dir/deps: run without the fallback
    _mediapipe = None

# This file's own directory, so that a symlink to it from another project
# still finds the recipe modules next to it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmn import apply_cmn
from conformer import Conformer  # noqa: E402

from icefall.checkpoint import load_checkpoint
from icefall.utils import str2bool

# `compute_avhubert_grid` lives in the recipe's local/ dir, which is not on
# sys.path when running conformer_ctc2/lipreader.py; add it before importing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local"))
from compute_avhubert_grid import (  # noqa: E402
    _detect_landmarks,
    _extract_mouth_frames,
    load_globals_avbubert,
    load_globals_dlib,
)

# Mouth-ROI extraction constants (same values used in feature precomputation).
ROI_SIZE = (88, 88)
MOUTH_LEFT, MOUTH_RIGHT = 48, 54  # dlib 68-point landmark indices
MOUTH_W = MOUTH_H = 64
BLANK_ID = 0  # icefall CTC blank
TARGET_FPS = 25.0  # GRID / AV-HuBERT frame rate; other rates are resampled
# Frames taller than this are downscaled for face *detection* only (dlib's
# HOG scan is ~linear in pixels; phone videos are 1080p+). Landmarks and the
# mouth crop still use full resolution.
DETECT_HEIGHT = 480
# Accepted clip length. Below the minimum no GRID-style sentence fits; above
# the maximum, per-frame landmarking would take too long for one request.
# The tolerance keeps nominal-length clips from being rejected over container
# metadata rounding.
MIN_DURATION_S = 2.0
MAX_DURATION_S = 15.0  # uploaded files
MAX_DURATION_WEBCAM_S = 20.0  # webcam recordings get a little more headroom
DURATION_TOL_S = 0.1

# Head-size normalization: scale the mouth crop to the face's interocular
# distance (crop = CROP_PER_IOD * IOD) before resizing to ROI_SIZE. GRID's
# fixed 64px crop corresponds to a median IOD of 48.4px, so a median GRID
# face gets the training crop.
GRID_IOD_REF = 48.4
CROP_PER_IOD = MOUTH_W / GRID_IOD_REF  # ~1.322


def _degrade_roi(roi: np.ndarray, noise_sigma: float, blur_sigma: float) -> np.ndarray:
    """Optionally blur then add Gaussian noise to ROI frames (T,H,W uint8),
    for robustness tests. Blur first: camera blur precedes
    sensor noise in a real pipeline."""
    import cv2

    out = roi.astype(np.float32)
    if blur_sigma > 0:
        out = np.stack([cv2.GaussianBlur(f, (0, 0), blur_sigma) for f in out])
    if noise_sigma > 0:
        rng = np.random.default_rng()
        out = out + rng.normal(0.0, noise_sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_mediapipe_fallback():
    """Build a fallback landmarker from MediaPipe Face Detection (BlazeFace,
    short-range): fast (~17 ms/frame CPU) and robust to glasses, tight
    framing and mild pose, the cases where dlib's HOG detector fails.

    Returns a callable (BGR frame) -> (68, 2) int array or None. Only the
    points the pipeline consumes are meaningful: the mouth-corner midpoint
    (indices 48/54, set so their mean is MediaPipe's mouth center; measured
    offset vs dlib is ~0.01-0.07 IOD with no systematic bias) and the eye
    blocks 36:42 / 42:48 (set to the eye keypoints, so the interocular
    distance for head-size normalization stays valid, within ~10% of dlib's).
    """
    import cv2

    if _mediapipe is None:
        raise RuntimeError(f"bundled mediapipe not importable ({MEDIAPIPE_DIR})")
    mp = _mediapipe

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    KP = mp.solutions.face_detection.FaceKeyPoint

    def fallback(image_bgr):
        h, w = image_bgr.shape[:2]
        result = detector.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not result.detections:
            return None
        kps = result.detections[0].location_data.relative_keypoints

        def px(k):
            return (kps[k].x * w, kps[k].y * h)

        lm = np.full((68, 2), px(KP.MOUTH_CENTER), dtype=float)
        lm[36:42] = px(KP.RIGHT_EYE)  # subject's right eye = dlib 36-41
        lm[42:48] = px(KP.LEFT_EYE)  # subject's left eye = dlib 42-47
        return np.round(lm).astype(int)

    return fallback


def _median_iod(landmarks: np.ndarray) -> float:
    """Median interocular distance (px) across frames from dlib 68 landmarks."""
    lm = landmarks.astype(float)
    reye = lm[:, 36:42].mean(axis=1)  # right-eye centre per frame
    leye = lm[:, 42:48].mean(axis=1)  # left-eye centre per frame
    return float(np.median(np.linalg.norm(leye - reye, axis=1)))


# HLG lattice-decoding hyper-parameters (from decode.py get_params()).
SEARCH_BEAM = 20
OUTPUT_BEAM = 4
MIN_ACTIVE_STATES = 30
MAX_ACTIVE_STATES = 10000


def load_tokens(path: Path):
    """Parse tokens.txt into (id -> symbol dict, num_classes).

    Lexicon disambiguation symbols ('#0', '#1', ...) are skipped: they belong to
    the L/HLG graph, not the model's CTC output, so num_classes must not count
    them (e.g. GRID bpe-58 has 58 output classes but 60 lines in tokens.txt).
    """
    id2sym = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            sym, idx = parts[0], int(parts[1])
            if sym.startswith("#"):  # lexicon disambig symbol, not a model class
                continue
            id2sym[idx] = sym
    if not id2sym:
        raise ValueError(f"No tokens parsed from {path}")
    return id2sym, max(id2sym) + 1


def ctc_greedy_decode(nnet_output: torch.Tensor, id2sym: dict) -> str:
    """(1, T, C) log-probs -> text: argmax, collapse repeats, drop blanks.

    Mirrors pretrained.py's token->word mapping (BPE '▁' -> space).
    """
    tokens = []
    prev = None
    for i in nnet_output[0].argmax(dim=-1).tolist():
        if i != BLANK_ID and i != prev:
            tokens.append(id2sym.get(i, ""))
        prev = i
    # Group BPE tokens ('▁' marks a word start) into words.
    words = []
    for sym in tokens:
        if sym.startswith("▁") or not words:
            words.append(sym.replace("▁", ""))
        else:
            words[-1] += sym
    return " ".join(w for w in words if w)


class LipReader:
    """Loads the AV-HuBERT front-end + Conformer-CTC model once, reusable per clip."""

    def __init__(self, args: argparse.Namespace):
        g = load_globals_avbubert(args)  # {device, model (AV-HuBERT, eval), transform}
        g.update(load_globals_dlib(args))  # {detector, predictor}
        self.detector = g["detector"]
        self.predictor = g["predictor"]
        # Per-frame fallback, tried only when the fast HOG detector fails
        # (glasses, tight framing, mild pose). Missing/broken MediaPipe
        # disables the fallback instead of crashing.
        self.fallback_landmarker = None
        if getattr(args, "mediapipe_fallback", True):
            try:
                self.fallback_landmarker = _load_mediapipe_fallback()
                logging.info("MediaPipe fallback landmarker enabled")
            except Exception:
                logging.exception("MediaPipe fallback unavailable; continuing without")
        self.avhubert = g["model"]
        self.transform = g["transform"]
        self.device = g["device"]
        self.layer = args.layer
        self.normalize_head = args.normalize_head
        # Mouth-ROI geometry. "meanface" aligns every frame to a reference face
        # before cropping (see local/meanface_roi.py) and must match whatever
        # the loaded checkpoint was trained on; a model trained on one
        # geometry decodes the other poorly.
        # Feature mean normalisation; must match the checkpoint's training.
        self.cmn = getattr(args, "cmn", "none")
        self.roi_mode = getattr(args, "roi_mode", "centroid")
        self.mean_face = None
        if self.roi_mode == "meanface":
            from meanface_roi import load_mean_face  # local/ is on sys.path

            self.mean_face = load_mean_face(args.mean_face)
            logging.info(f"Mouth ROI: mean-face aligned (from {args.mean_face})")

        self.id2sym, num_classes = load_tokens(args.tokens)

        model = Conformer(
            num_features=getattr(args, "feature_dim", 768),
            num_classes=num_classes,
            subsampling_factor=1,
            d_model=args.encoder_dim,
            nhead=8,
            dim_feedforward=1024,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
        )
        load_checkpoint(str(args.checkpoint), model, strict=True)
        self.model = model.to(self.device).eval()
        logging.info(
            f"Loaded model from {args.checkpoint} "
            f"(num_classes={num_classes}, layer={self.layer})"
        )

        # Optional lexicon-constrained decoding graph (HLG).
        self.method = args.method
        self.HLG = None
        self.word_table = None
        if self.method == "1best":
            if args.HLG is None or args.words_file is None:
                raise ValueError("--method 1best requires --HLG and --words-file")
            import k2

            self.HLG = k2.Fsa.from_dict(
                torch.load(args.HLG, map_location="cpu", weights_only=False)
            ).to(self.device)
            if not hasattr(self.HLG, "lm_scores"):
                self.HLG.lm_scores = self.HLG.scores.clone()
            self.word_table = k2.SymbolTable.from_file(str(args.words_file))
            logging.info(f"Loaded HLG from {args.HLG} and words from {args.words_file}")

    def _roi_and_features(
        self, video: str, noise_sigma: float = 0.0, blur_sigma: float = 0.0
    ):
        """Return (raw ROI frames (T,88,88) uint8, AV-HuBERT features (T,768))."""
        # Temporal normalization: the model was trained on 25 fps GRID video,
        # so resample other frame rates (e.g. 30/60 fps uploads) before
        # landmarking. No-op for GRID clips.
        resampled = None
        fps = _video_fps(video)
        if fps and abs(fps - TARGET_FPS) > 0.1:
            resampled = _resample_to_target_fps(video)
            if resampled is None:
                logging.warning(
                    f"{video} is {fps:.2f} fps and could not be resampled to "
                    f"{TARGET_FPS:g} fps (ffmpeg missing or failed); "
                    "recognition accuracy may degrade."
                )
            else:
                logging.info(f"Resampled {video}: {fps:.2f} -> {TARGET_FPS:g} fps")
        try:
            return self._roi_and_features_25fps(
                video,
                Path(resampled) if resampled else Path(video),
                noise_sigma,
                blur_sigma,
            )
        finally:
            if resampled is not None:
                Path(resampled).unlink(missing_ok=True)

    def _roi_and_features_25fps(
        self,
        video: str,
        video_path: Path,
        noise_sigma: float = 0.0,
        blur_sigma: float = 0.0,
    ):
        """`_roi_and_features` body; video_path is the (possibly resampled)
        clip to process, video the original name for messages."""
        landmarks = _detect_landmarks(
            video_path,
            self.detector,
            self.predictor,
            1,
            detect_height=DETECT_HEIGHT,
            fallback_landmarker=self.fallback_landmarker,
        )
        if landmarks is None:
            logging.warning(f"No face detected in {video}")
            raise RuntimeError(
                "no face was detected in the clip. Please use a frontal "
                "talking-face video with the whole face visible."
            )
        if self.roi_mode == "meanface":
            # Pose and scale normalization: every frame is warped onto a
            # reference face (similarity fit on eye corners + nose tip) before
            # the mouth crop, as in AV-HuBERT's own pretraining data.
            from meanface_roi import extract_meanface_frames

            roi = extract_meanface_frames(video_path, landmarks, self.mean_face)
            if roi.shape[0] == 0:
                raise RuntimeError(
                    "the mouth region could not be aligned in this clip. "
                    "Please use a frontal talking-face video."
                )
            return self._features_from_roi(roi, noise_sigma, blur_sigma)
        # Head-size normalization: scale the mouth crop to the detected face
        # size (via IOD) so the mouth-to-head ratio matches GRID regardless of
        # how big the face is in the frame. On GRID this yields ~64px (no-op).
        if self.normalize_head:
            crop = max(16, round(CROP_PER_IOD * _median_iod(landmarks)))
            logging.info(
                f"Head-normalized mouth crop: {crop}px "
                f"(IOD {_median_iod(landmarks):.1f}px)"
            )
        else:
            crop = MOUTH_W
        raw = _extract_mouth_frames(
            video_path, landmarks, crop, crop, ROI_SIZE, MOUTH_LEFT, MOUTH_RIGHT
        )
        return self._features_from_roi(np.stack(raw), noise_sigma, blur_sigma)

    def _features_from_roi(
        self, roi: np.ndarray, noise_sigma: float = 0.0, blur_sigma: float = 0.0
    ):
        """Shared tail of both ROI geometries: optional degradation, then
        AV-HuBERT. Returns (T,88,88) uint8 ROI and (T,768) features."""
        if noise_sigma or blur_sigma:  # robustness tests
            roi = _degrade_roi(roi, noise_sigma, blur_sigma)
        frames = self.transform(roi)  # normalized float (T, 88, 88)
        tensor = torch.FloatTensor(frames).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats, _ = self.avhubert.extract_finetune(
                source={"video": tensor, "audio": None},
                padding_mask=None,
                output_layer=self.layer,
            )
        return roi, feats.squeeze(0)  # (T,88,88), (T,768)

    def _decode_1best(self, nnet_output: torch.Tensor) -> str:
        """Lexicon-constrained decoding via HLG + one-best path (like decode.py)."""
        from icefall.decode import one_best_decoding
        from icefall.utils import get_texts

        lattice = self._build_lattice(nnet_output, OUTPUT_BEAM)
        best_path = one_best_decoding(lattice=lattice, use_double_scores=True)
        return " ".join(self.word_table[i] for i in get_texts(best_path)[0])

    def _build_lattice(self, nnet_output: torch.Tensor, output_beam: int):
        from icefall.decode import get_lattice

        T = nnet_output.shape[1]
        # Single, unpadded clip: one segment covering all T frames.
        supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
        return get_lattice(
            nnet_output=nnet_output,
            decoding_graph=self.HLG,
            supervision_segments=supervision_segments,
            search_beam=SEARCH_BEAM,
            output_beam=output_beam,
            min_active_states=MIN_ACTIVE_STATES,
            max_active_states=MAX_ACTIVE_STATES,
            subsampling_factor=1,
        )

    @torch.no_grad()
    def recognize(self, video: str, noise_sigma: float = 0.0, blur_sigma: float = 0.0):
        """Return (recognised_text, ROI frames as recognized)."""
        # Backstop: callers may enforce stricter source-specific limits
        # up front, so only the more lenient bound is checked here.
        err = _duration_error(video, MAX_DURATION_WEBCAM_S)
        if err:
            raise ValueError(err)
        return self._infer(video, noise_sigma, blur_sigma)

    def _infer(self, video: str, noise_sigma: float, blur_sigma: float):
        """One inference on `self.device`: (text, ROI frames)."""
        roi, feats = self._roi_and_features(video, noise_sigma, blur_sigma)
        feature = feats.unsqueeze(0).to(self.device)  # (1, T, 768)
        # Must match the --cmn the checkpoint was trained with.
        feature = apply_cmn(feature, self.cmn)
        nnet_output, _, _ = self.model(feature, None)  # (1, T, C)
        if self.method == "1best":
            return self._decode_1best(nnet_output), roi
        return ctc_greedy_decode(nnet_output.cpu(), self.id2sym), roi


def _video_fps(video: str) -> float:
    """Container frame rate as reported by OpenCV (0.0 if unknown)."""
    import cv2

    cap = cv2.VideoCapture(video)
    try:
        return cap.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        cap.release()


def _video_duration(video: str) -> float:
    """Duration in seconds. Uses container metadata (fast, works for normal
    files); falls back to reading packet timestamps with ffprobe for
    variable-frame-rate clips like browser-recorded WebM, whose containers
    store no usable duration (OpenCV returns a garbage frame count).
    Returns 0.0 only if both fail."""
    import cv2

    cap = cv2.VideoCapture(video)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps > 0 and 0 < frames < 1e7:
            return frames / fps
    finally:
        cap.release()
    return _ffprobe_duration(video)


def _ffprobe_duration(video: str) -> float:
    """Duration from the last video packet's timestamp (demux only, no pixel
    decode). Works for VFR/WebM. 0.0 if ffprobe is missing or fails."""
    import shutil
    import subprocess

    if shutil.which("ffprobe") is None:
        return 0.0
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "packet=pts_time",
                "-of",
                "csv=p=0",
                video,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    times = [float(t) for t in out.split() if t and t != "N/A"]
    return max(times) if times else 0.0


def _duration_error(video: str, max_duration: float = MAX_DURATION_S):
    """Rejection message if the clip's length is outside the accepted range,
    None if acceptable (or the duration is unknown)."""
    duration = _video_duration(video)
    if duration and not (
        MIN_DURATION_S - DURATION_TOL_S <= duration <= max_duration + DURATION_TOL_S
    ):
        return (
            f"Clip is {duration:.1f}s long; please use a clip between "
            f"{MIN_DURATION_S:g}s and {max_duration:g}s."
        )
    return None


def _resample_to_target_fps(src: str):
    """Re-encode src at TARGET_FPS (video only). Returns the path to a temp
    file the caller must delete, or None if ffmpeg is unavailable/fails."""
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        return None
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    try:
        # mpeg4 (not libx264): handles odd frame dimensions, and the output is
        # only read back by OpenCV, never a browser.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                src,
                "-vf",
                f"fps={TARGET_FPS:g}",
                "-an",
                "-c:v",
                "mpeg4",
                "-q:v",
                "2",
                out,
            ],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError):
        Path(out).unlink(missing_ok=True)
        return None
    return out


def add_model_args(p: argparse.ArgumentParser) -> None:
    """Arguments that build a LipReader (model, front-end, decoding)."""
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trained Conformer-CTC checkpoint (.pt).",
    )
    p.add_argument(
        "--tokens",
        type=Path,
        required=True,
        help="tokens.txt (sizes the model; used by ctc-greedy decoding).",
    )
    p.add_argument(
        "--method",
        choices=["ctc-greedy", "1best"],
        default="1best",
        help="Decoding method. 1best is lexicon-constrained (uses --HLG and "
        "--words-file); ctc-greedy uses --tokens only and groups BPE tokens "
        "into words, so it suits BPE models only.",
    )
    p.add_argument(
        "--HLG",
        type=Path,
        default=Path("data/lang_phone/HLG.pt"),
        help="HLG.pt decoding graph (used by --method 1best).",
    )
    p.add_argument(
        "--words-file",
        type=Path,
        default=Path("data/lang_phone/words.txt"),
        help="words.txt symbol table (used by --method 1best).",
    )
    p.add_argument(
        "--avhubert-code-dir",
        type=Path,
        default=Path("av_hubert"),
        help="AV-HuBERT source dir (added temporarily to sys.path).",
    )
    p.add_argument(
        "--avhubert-ckpt",
        type=Path,
        required=True,
        help="AV-HuBERT pretrained checkpoint (.pt).",
    )
    p.add_argument(
        "--dlib-predictor",
        type=Path,
        default=Path("download/dlib/shape_predictor_68_face_landmarks.dat"),
        help="dlib 68-point landmark model.",
    )
    p.add_argument(
        "--feature-dim",
        type=int,
        default=768,
        help="SSL feature dim: 768 AV-HuBERT Base, 1024 Large.",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=8,
        help="AV-HuBERT encoder output layer (must match training).",
    )
    p.add_argument(
        "--roi-mode",
        choices=["centroid", "meanface"],
        default="meanface",
        help="Mouth-ROI geometry. 'centroid': crop around the mouth "
        "centre, scaled by measured head size (--normalize-head). "
        "'meanface': warp each frame onto a reference face "
        "first (AV-HuBERT's own preprocessing), which also "
        "corrects pose and in-plane rotation. Must match the "
        "geometry --checkpoint was trained on. "
        "Default: %(default)s",
    )
    p.add_argument(
        "--mean-face",
        type=Path,
        default=Path("download/20words_mean_face.npy"),
        help="Reference landmarks for --roi-mode meanface. " "Default: %(default)s",
    )
    p.add_argument(
        "--normalize-head",
        type=str2bool,
        default=True,
        help="Scale the mouth crop to the detected face size (IOD) so "
        "the mouth-to-head ratio matches GRID for arbitrary "
        "videos. No-op for GRID-sized faces. Default: %(default)s",
    )
    p.add_argument(
        "--mediapipe-fallback",
        type=str2bool,
        default=True,
        help="When dlib's HOG detector finds no face in a frame, "
        "fall back to MediaPipe face detection (fast, robust "
        "to glasses/tight framing). Default: %(default)s",
    )
    # Model geometry; must match the trained checkpoint.
    p.add_argument("--encoder-dim", type=int, default=128)
    p.add_argument("--num-encoder-layers", type=int, default=6)
    p.add_argument("--num-decoder-layers", type=int, default=3)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lipread one video with a GRID conformer_ctc2 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_args(p)
    p.add_argument("video", nargs="?", default=None, help="Video path.")
    p.add_argument(
        "--noise",
        type=float,
        nargs="+",
        default=None,
        help="Gaussian noise sigma(s) applied to the mouth ROI "
        "before recognition (0-50). "
        "Multiple values re-run the same video once per "
        "value, reusing the loaded model (a noise/blur "
        "robustness sweep). Paired index-wise with --blur.",
    )
    p.add_argument(
        "--blur",
        type=float,
        nargs="+",
        default=None,
        help="Gaussian blur sigma(s) applied to the mouth ROI "
        "before recognition (0-5). "
        "Same broadcasting/pairing rule as --noise.",
    )
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = get_args()
    reader = LipReader(args)
    if args.video is None:
        raise SystemExit("Provide a video path.")
    noise = args.noise or [0.0]
    blur = args.blur or [0.0]
    n = max(len(noise), len(blur))
    if len(noise) == 1 and n > 1:
        noise = noise * n
    if len(blur) == 1 and n > 1:
        blur = blur * n
    if len(noise) != len(blur):
        raise SystemExit(
            "--noise/--blur must have matching lengths "
            "(or a single value to broadcast)"
        )
    for n_sigma, b_sigma in zip(noise, blur):
        text, _ = reader.recognize(args.video, noise_sigma=n_sigma, blur_sigma=b_sigma)
        print(f"[noise={n_sigma:g} blur={b_sigma:g}] {text}")


if __name__ == "__main__":
    main()
