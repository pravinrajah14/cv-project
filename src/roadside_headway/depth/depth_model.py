from __future__ import annotations

import numpy as np

DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    """Wraps Depth Anything V2 (Small) for relative monocular depth.

    In this fixed-camera pipeline, depth is an AUXILIARY signal only — real-world
    position comes from the ground-plane homography (calibration module), not
    from depth. Depth is used to detect which of two overlapping vehicles is
    nearer the camera (the occluder), so the occluded vehicle's unreliable
    visible contact point can be replaced with a track-history extrapolation
    instead (see tracking.occlusion). This is a deliberate departure from the
    vehicle-mounted version of the original idea, where depth would be the
    primary distance source.

    Depth Anything's output convention: larger value = nearer the camera.

    torch/transformers are imported lazily in __init__ so the rest of this
    package (and `median_depth_in_box` below) stays importable and testable
    without the full ML stack installed.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a relative depth map (H, W) matching the input frame's resolution."""
        import torch
        from PIL import Image

        image = Image.fromarray(frame_bgr[:, :, ::-1])
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth  # (1, h', w') at model resolution
            depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )
        return depth.squeeze().float().cpu().numpy()


def median_depth_in_box(depth_map: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(x1, 0), max(y1, 0)
    # x2/y2 are exclusive slice bounds, so the valid clamp is shape[i], not shape[i] - 1
    # (clamping to shape[i] - 1 would silently drop the box's last row/column).
    x2 = min(x2, depth_map.shape[1])
    y2 = min(y2, depth_map.shape[0])
    if x2 <= x1 or y2 <= y1:
        return float("nan")
    return float(np.median(depth_map[y1:y2, x1:x2]))
