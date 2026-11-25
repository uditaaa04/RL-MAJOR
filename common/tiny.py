"""
The ESP32's tiny model, PC side: file format, the exact same arithmetic as the firmware, and the
one function that turns a box photo into the few bytes sent to the ESP32.

The model is a handful of 3x3 convolutions (ReLU), an average, and one output.
File layout ("TNY1", little-endian, everything on 4-byte boundaries) - the firmware reads the same:
    "TNY1" | in_size, in_channels, n_conv, 0 | mean f32 | std f32
    per conv: cin, cout, stride, 0 | weights f32[cout*cin*9] | bias f32[cout]
    dense: weights f32[last cout] | bias f32

No PyTorch here: only numpy and Pillow, so the gateway can load it anywhere.
"""
from __future__ import annotations

import struct

import numpy as np
from PIL import Image

MAGIC = b"TNY1"


def pack(size: int, ch: int, mean: float, std: float, convs, dense_w, dense_b) -> bytes:
    """convs: list of (weights[cout, cin, 3, 3], bias[cout], stride)."""
    out = [MAGIC, bytes([size, ch, len(convs), 0]), struct.pack("<ff", float(mean), float(std))]
    for w, b, stride in convs:
        w = np.asarray(w, "<f4")
        out += [bytes([w.shape[1], w.shape[0], int(stride), 0]), w.tobytes(), np.asarray(b, "<f4").tobytes()]
    out += [np.asarray(dense_w, "<f4").tobytes(), struct.pack("<f", float(dense_b))]
    return b"".join(out)


def _parse(blob: bytes) -> dict:
    if len(blob) < 20 or blob[:4] != MAGIC:
        raise ValueError("not a tiny-model file")
    size, ch, nconv = blob[4], blob[5], blob[6]
    mean, std = struct.unpack_from("<ff", blob, 8)
    off, c, convs = 16, ch, []
    for _ in range(nconv):
        cin, cout, stride = blob[off], blob[off + 1], blob[off + 2]
        off += 4
        if cin != c:
            raise ValueError("tiny-model file: layer sizes do not chain")
        w = np.frombuffer(blob, "<f4", cout * cin * 9, off).reshape(cout, cin, 3, 3)
        off += 4 * w.size
        b = np.frombuffer(blob, "<f4", cout, off)
        off += 4 * cout
        convs.append((w, b, stride))
        c = cout
    dw = np.frombuffer(blob, "<f4", c, off)
    off += 4 * c
    db = struct.unpack_from("<f", blob, off)[0]
    if off + 4 != len(blob):
        raise ValueError("tiny-model file: wrong length")
    return {"size": size, "ch": ch, "mean": mean, "std": std, "convs": convs, "dw": dw, "db": db}


def parse(blob: bytes) -> dict:
    """Read a tiny-model file. Raises ValueError for anything that is not a complete, well-formed file."""
    try:
        return _parse(blob)
    except (struct.error, IndexError, TypeError) as e:
        raise ValueError(f"tiny-model file is damaged: {e}") from None


def forward(model: dict, px) -> np.ndarray:
    """px: uint8 array [N, ch, size, size] (or one photo [ch*size*size]) -> P(damaged) per photo.
    Same steps and same single-precision numbers as modelRun() in the firmware."""
    size, ch = model["size"], model["ch"]
    x = np.asarray(px, np.uint8).reshape(-1, ch, size, size).astype(np.float32)
    x = x * np.float32(1.0 / (255.0 * model["std"])) - np.float32(model["mean"] / model["std"])
    for w, b, s in model["convs"]:
        h = x.shape[2]
        ho = (h + 2 - 3) // s + 1
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((x.shape[0], w.shape[0], ho, ho), np.float32)
        for ky in range(3):
            for kx in range(3):
                patch = xp[:, :, ky:ky + s * (ho - 1) + 1:s, kx:kx + s * (ho - 1) + 1:s]
                out += np.einsum("oc,nchw->nohw", w[:, :, ky, kx], patch, optimize=True)
        x = np.maximum(out + b[None, :, None, None], 0)
    z = x.mean((2, 3)) @ model["dw"] + np.float32(model["db"])
    return 1.0 / (1.0 + np.exp(-z.astype(np.float64)))


def to_pixels(img: Image.Image, size: int, ch: int = 1) -> bytes:
    """Box photo -> the bytes the ESP32 receives. Used for training, for the pretend ESP32 and for
    the real board, so all three see exactly the same pixels."""
    if ch == 1:
        return img.convert("L").resize((size, size), Image.BILINEAR).tobytes()
    a = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), np.uint8)
    return a.transpose(2, 0, 1).tobytes()


def checksum(data: bytes) -> int:
    return sum(data) & 0xFFFF


def mults(model: dict) -> int:
    """Number of multiply-adds for one photo (used to guess the ESP32's compute time when no board is connected)."""
    h, n = model["size"], 0
    for w, _, s in model["convs"]:
        h = (h + 2 - 3) // s + 1
        n += h * h * w.shape[0] * w.shape[1] * 9
    return n


def n_params(model: dict) -> int:
    return sum(w.size + b.size for w, b, _ in model["convs"]) + model["dw"].size + 1


class TinyModel:
    def __init__(self, blob: bytes):
        self.blob = bytes(blob)
        self.m = parse(self.blob)
        self.size, self.ch = self.m["size"], self.m["ch"]
        self.mults = mults(self.m)
        self.params = n_params(self.m)

    @staticmethod
    def load(path: str) -> "TinyModel":
        with open(path, "rb") as f:
            return TinyModel(f.read())

    def pixels(self, img: Image.Image) -> bytes:
        return to_pixels(img, self.size, self.ch)

    def pixels_blank(self) -> bytes:
        return bytes(self.size * self.size * self.ch)

    def predict_pixels(self, px: bytes) -> float:
        return float(forward(self.m, np.frombuffer(px, np.uint8))[0])

    def predict(self, img: Image.Image) -> float:
        return self.predict_pixels(self.pixels(img))
