# utils.py
import cv2
import numpy as np

def resize_frame_if_needed(frame: np.ndarray, max_width: int) -> np.ndarray:
    """
    Resize frame if it exceeds maximum width while maintaining aspect ratio.

    Args:
        frame: Input frame.
        max_width: Maximum allowed width.

    Returns:
        Resized frame or original if within limits.
    """
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return frame

def create_ghost_trail(composite: np.ndarray, trail_accumulator: np.ndarray) -> np.ndarray:
    """
    Create a ghost trail effect with a soft glow.

    Args:
        composite: Base composite image.
        trail_accumulator: Accumulated trail data.

    Returns:
        Enhanced composite with ghost trail effect.
    """
    # Extract trail intensity as the max value across color channels
    trail_intensity = np.max(trail_accumulator, axis=2)
    max_intensity = np.max(trail_intensity)

    if max_intensity > 0:
        # Create a glow effect by heavily blurring the intensity map
        glow = cv2.GaussianBlur(trail_intensity, (21, 21), 0)
        glow = cv2.GaussianBlur(glow, (21, 21), 0)

        # Normalize glow to the range [0, 255]
        max_glow = np.max(glow)
        if max_glow > 0:
            glow = (glow / max_glow * 255).astype(np.uint8)

            # Create a colored glow (e.g., cyan-ish)
            glow_colored = np.zeros_like(composite)
            glow_colored[:, :, 0] = glow * 0.5  # Blue
            glow_colored[:, :, 1] = glow * 0.8  # Green
            glow_colored[:, :, 2] = glow * 0.3  # Red

            # Blend the glow with the composite image
            alpha = 0.4
            composite = cv2.addWeighted(composite, 1, glow_colored, alpha, 0)

    return composite