"""Image pre-processing + ONNX model wrapper, shared by the gateway and the cloud server.

No PyTorch needed at run time - only numpy, Pillow and onnxruntime.
"""
from __future__ import annotations

import io
import time

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop_roi(frame_rgb: np.ndarray, roi, out_size: int = 256) -> Image.Image:
    """Cut the square area where the box appears and resize it to out_size x out_size."""
    x, y, w, h = [int(v) for v in roi]
    H, W = frame_rgb.shape[:2]
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    crop = frame_rgb[y:y + h, x:x + w]
    return Image.fromarray(crop).resize((out_size, out_size), Image.BILINEAR)


def random_hard_photo(img: Image.Image, rnd) -> Image.Image:
    """TRAINING ONLY: the same photo as on a worse day on the line: less light and / or a smear along the
    belt (the box moved while the picture was taken). rnd = Python's `random` module or a random.Random.
    Why: measured on the test photos, a model trained only on clean photos loses up to 20 points when the
    photo gets darker and blurred (results/shift_scores.json). A webcam over a moving belt gives such photos."""
    a = np.asarray(img, np.float32)
    if rnd.random() < 0.6:                                   # less light: 35 % .. 100 %
        a = a * rnd.uniform(0.35, 1.0)
    if rnd.random() < 0.6:                                   # smear along x: 1 % .. 5 % of the width
        k = int(round(a.shape[1] * rnd.uniform(0.01, 0.05))) | 1
        if k >= 3:
            pad = k // 2
            b = np.pad(a, ((0, 0), (pad, pad), (0, 0)), mode="edge")
            c = np.cumsum(np.concatenate([np.zeros((b.shape[0], 1, b.shape[2]), np.float32), b], axis=1), axis=1)
            a = (c[:, k:] - c[:, :-k]) / k
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def to_jpeg(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def from_jpeg(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


class OnnxInspector:
    """Runs a DefectNet ONNX file: image -> P(damaged) + damage heat-map."""

    def __init__(self, path: str, threads: int = 0, providers=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        avail = ort.get_available_providers()
        prov = [p for p in (providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]) if p in avail]
        self.sess = ort.InferenceSession(path, so, providers=prov or ["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shp = inp.shape
        self.size = int(shp[2]) if isinstance(shp[2], int) else 128
        self.outputs = [o.name for o in self.sess.get_outputs()]
        self.provider = self.sess.get_providers()[0]

    def preprocess(self, img: Image.Image) -> np.ndarray:
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BILINEAR)
        a = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
        return a.transpose(2, 0, 1)[None].astype(np.float32)

    def predict(self, img: Image.Image):
        """returns (p_damaged, heatmap 2-D list, compute_seconds)"""
        t = time.perf_counter()
        x = self.preprocess(img)
        outs = self.sess.run(None, {self.input_name: x})
        logits = outs[0][0]
        z = float(logits[1] - logits[0])
        p = 1.0 / (1.0 + np.exp(-z))
        heat = outs[1][0, 0].astype(np.float64).round(3).tolist() if len(outs) > 1 else None
        return p, heat, time.perf_counter() - t
