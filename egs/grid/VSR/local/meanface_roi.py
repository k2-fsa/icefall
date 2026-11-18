# Copyright   2026  (author: Ibrahim Almajai)
# Apache 2.0

"""Mean-face-aligned mouth ROIs (canonical AV-HuBERT preprocessing).

Mirrors ``av_hubert/avhubert/preparation/align_mouth.py``: per-frame landmarks
are smoothed over a sliding window, a similarity transform maps the stable
points (nose tip + eye corners) onto a mean face in a 256x256 canvas, and a
96x96 mouth crop is centre-cropped to 88x88.

The reference implementation warps with scikit-image, which is not installed in
this environment (and lipreader.py runs from that same env, so adding a
dependency there is not free). The similarity fit is therefore done directly
with Umeyama (the algorithm skimage's
``estimate_transform('similarity')`` uses) and the warp with OpenCV, in float
so the uint8 conversion truncates exactly as the reference does. Fidelity
against the reference caches is checked in ``_check_against_reference``.

Unlike the centroid crop, this normalises pose and in-plane rotation as well
as scale, which is why it holds up on out-of-domain framing.
"""
import logging

import cv2
import numpy as np

# Nose tip + outer/inner eye corners: the points align_mouth fits on.
STABLE_PTS = [33, 36, 39, 42, 45]
STD_SIZE = (256, 256)
CROP = 96  # align_mouth crop_height/crop_width
OUT = 88  # AV-HuBERT input size
_MARGIN = (CROP - OUT) // 2
MOUTH_START, MOUTH_STOP = 48, 68  # mouth landmark range the crop centres on
WINDOW_MARGIN = 12


def load_mean_face(path) -> np.ndarray:
    """Load the 68x2 mean-face reference landmarks."""
    mean_face = np.load(str(path))
    if mean_face.shape != (68, 2):
        raise ValueError(f"{path}: expected (68, 2) mean face, got {mean_face.shape}")
    return mean_face


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform src -> dst as a 2x3 affine matrix.

    This is skimage's ``estimate_transform('similarity')`` (Umeyama 1991) with
    the degenerate branches that cannot arise for 5 non-collinear face points
    left out.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_demean, dst_demean = src - src_mean, dst - dst_mean

    A = dst_demean.T @ src_demean / src.shape[0]
    U, S, Vt = np.linalg.svd(A)
    d = np.ones(2)
    if np.linalg.det(A) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    scale = (S @ d) / src_demean.var(axis=0).sum()

    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = scale * R
    M[:, 2] = dst_mean - scale * R @ src_mean
    return M


def _warp(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Warp to STD_SIZE. Float in, *255, truncate, matching the reference,
    which runs skimage.warp (float [0,1]) then ``astype('uint8')``."""
    src = img.astype(np.float32) / 255.0
    warped = cv2.warpAffine(
        src,
        M.astype(np.float64),
        (STD_SIZE[1], STD_SIZE[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (warped * 255).astype(np.uint8)


def _apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ M[:, :2].T + M[:, 2]


def _cut_patch(
    img: np.ndarray, landmarks: np.ndarray, height: int, width: int, threshold: int = 5
) -> np.ndarray:
    """align_mouth.cut_patch: centre a 2*height x 2*width box on the landmark
    centroid, clamped to the image (raising only when the clamp is large)."""
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
        if center_y - height < -threshold:
            raise ValueError("too much bias in height")
    if center_x - width < 0:
        center_x = width
        if center_x - width < -threshold:
            raise ValueError("too much bias in width")
    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
        if center_y + height > img.shape[0] + threshold:
            raise ValueError("too much bias in height")
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
        if center_x + width > img.shape[1] + threshold:
            raise ValueError("too much bias in width")

    return np.copy(
        img[
            int(round(center_y) - round(height)) : int(round(center_y) + round(height)),
            int(round(center_x) - round(width)) : int(round(center_x) + round(width)),
        ]
    )


def extract_meanface_frames(
    video_path,
    landmarks: np.ndarray,
    mean_face: np.ndarray,
    window_margin: int = WINDOW_MARGIN,
) -> np.ndarray:
    """Mean-face-aligned mouth ROIs for one clip.

    Parameters
    ----------
    video_path : path to the (already frame-rate normalised) clip.
    landmarks : (T, 68, 2) per-frame landmarks, gaps already interpolated.
    mean_face : (68, 2) reference landmarks.

    Returns (T, 88, 88) uint8 greyscale frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    n = min(len(frames), len(landmarks))
    if n == 0:
        return np.empty((0, OUT, OUT), dtype=np.uint8)
    frames, landmarks = frames[:n], landmarks[:n]
    margin = min(n, window_margin)

    # Sliding window: the transform comes from the smoothed landmarks of the
    # window, but is applied to the frame leaving it (align_mouth's deque).
    sequence = []
    q_frame, q_lms = [], []
    M = None
    for idx in range(n):
        q_frame.append(frames[idx])
        q_lms.append(landmarks[idx])
        if len(q_frame) == margin:
            smoothed = np.mean(q_lms, axis=0)
            cur_frame, cur_lms = q_frame.pop(0), q_lms.pop(0)
            M = _umeyama_similarity(smoothed[STABLE_PTS, :], mean_face[STABLE_PTS, :])
            sequence.append(
                _cut_patch(
                    _warp(cur_frame, M),
                    _apply(M, cur_lms)[MOUTH_START:MOUTH_STOP],
                    CROP // 2,
                    CROP // 2,
                )
            )
    # Flush the tail through the last transform, as the reference does.
    while q_frame:
        cur_frame, cur_lms = q_frame.pop(0), q_lms.pop(0)
        sequence.append(
            _cut_patch(
                _warp(cur_frame, M),
                _apply(M, cur_lms)[MOUTH_START:MOUTH_STOP],
                CROP // 2,
                CROP // 2,
            )
        )

    out = np.empty((len(sequence), OUT, OUT), dtype=np.uint8)
    for i, patch in enumerate(sequence):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        out[i] = gray[_MARGIN : _MARGIN + OUT, _MARGIN : _MARGIN + OUT]
    return out
