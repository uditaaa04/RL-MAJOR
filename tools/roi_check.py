"""
Check where the camera crop (camera.roi in config) sits - run on the gateway.

    python -m tools.roi_check

Saves roi_check.jpg (full frame + the crop box drawn in cyan) and roi_crop.jpg
(what the AI models will see). Put a box under the camera, adjust camera.roi
(x, y, w, h) in config/system.yaml until the box sits in the middle of the crop.
"""
from __future__ import annotations

import time

from PIL import Image, ImageDraw

from common.config import load_config, project_path
from common.vision_io import crop_roi


def main():
    cfg = load_config()
    from gateway_node.camera import Camera
    cam = Camera(cfg)
    time.sleep(1.0)
    frame = cam.latest_frame()
    cam.close()
    if frame is None:
        raise SystemExit("no camera frame")
    img = Image.fromarray(frame)
    x, y, w, h = cfg["camera"]["roi"]
    d = ImageDraw.Draw(img)
    d.rectangle([x, y, x + w, y + h], outline=(34, 211, 238), width=4)
    img.save(project_path("roi_check.jpg"))
    crop_roi(frame, cfg["camera"]["roi"], cfg["camera"]["upload_size"]).save(project_path("roi_crop.jpg"))
    print("saved roi_check.jpg and roi_crop.jpg")


if __name__ == "__main__":
    main()
