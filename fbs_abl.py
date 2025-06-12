# fbs_abl.py
import cv2
import numpy as np
from typing import Optional
from utils import create_ghost_trail


class FBSProcessor:
    """
    Simplified Fast Background Subtraction with Adaptive Block Learning
    Optimized for low-memory environments
    """

    def __init__(self, block_size: int = 8, threshold: int = 25, trail_opacity: float = 0.7):
        """
        Initialize the FBS processor

        Args:
            block_size: Size of blocks for processing (default: 8x8)
            threshold: Motion detection threshold
            trail_opacity: Opacity of movement trails
        """
        self.block_size = block_size
        self.threshold = threshold
        self.trail_opacity = trail_opacity
        self.background_model = None
        self.background_initialized = False
        self.trail_accumulator = None
        self.frame_height = None
        self.frame_width = None
        self.last_frame = None
        self.frame_count = 0

    def _initialize_background(self, frame: np.ndarray) -> None:
        """
        Initialize background model with the first frame

        Args:
            frame: First frame of the video
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.blocks_y = self.frame_height // self.block_size
        self.blocks_x = self.frame_width // self.block_size

        self.background_model = np.zeros((self.blocks_y, self.blocks_x), dtype=np.float32)

        for by in range(self.blocks_y):
            for bx in range(self.blocks_x):
                y_start, x_start = by * self.block_size, bx * self.block_size
                y_end, x_end = y_start + self.block_size, x_start + self.block_size
                block = gray_frame[y_start:y_end, x_start:x_end]
                if block.size > 0:
                    self.background_model[by, bx] = np.mean(block)

        self.trail_accumulator = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.float32)
        self.last_frame = frame.copy()
        self.background_initialized = True

    def _detect_motion_blocks(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect motion using block-based comparison

        Args:
            frame: Current frame (BGR)

        Returns:
            Binary mask of detected motion
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)

        for by in range(self.blocks_y):
            for bx in range(self.blocks_x):
                y_start, x_start = by * self.block_size, bx * self.block_size
                y_end, x_end = y_start + self.block_size, x_start + self.block_size

                block = gray_frame[y_start:y_end, x_start:x_end]
                if block.size == 0:
                    continue

                current_mean = np.mean(block)
                diff = abs(current_mean - self.background_model[by, bx])

                if diff > self.threshold:
                    motion_mask[y_start:y_end, x_start:x_end] = 255
                    # Simple adaptive learning for gradual changes
                    self.background_model[by, bx] = (0.95 * self.background_model[by, bx] + 0.05 * current_mean)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        return motion_mask

    def _accumulate_trail(self, motion_mask: np.ndarray) -> None:
        """
        Accumulate motion trail over time

        Args:
            motion_mask: Binary mask of detected motion
        """
        trail_color = np.array([0, 0, 255], dtype=np.float32)  # BGR format for red

        if np.any(motion_mask):
            blurred_mask = cv2.GaussianBlur(motion_mask, (15, 15), 0).astype(np.float32) / 255.0

            for c in range(3):
                self.trail_accumulator[:, :, c] = np.maximum(
                    self.trail_accumulator[:, :, c],
                    blurred_mask * trail_color[c] * self.trail_opacity
                )

        # Apply decay to create a fading effect
        self.trail_accumulator *= 0.98

    def process_frame(self, frame: np.ndarray) -> None:
        """
        Process a single frame

        Args:
            frame: Input frame (BGR)
        """
        if not self.background_initialized:
            self._initialize_background(frame)
            return

        motion_mask = self._detect_motion_blocks(frame)
        self._accumulate_trail(motion_mask)
        self.last_frame = frame.copy()
        self.frame_count += 1

    def get_trail_visualization(self) -> Optional[np.ndarray]:
        """
        Get final trail visualization overlaid on the last frame

        Returns:
            Composite image with trails or None if not initialized
        """
        if self.last_frame is None or self.trail_accumulator is None:
            return None

        composite = self.last_frame.astype(np.float32)
        trail_mask = np.sum(self.trail_accumulator, axis=2) > 1.0  # Threshold to apply blend

        # Blend trails onto the composite image
        composite[trail_mask] = composite[trail_mask] * 0.3 + self.trail_accumulator[trail_mask] * 0.7
        composite = np.clip(composite, 0, 255)

        # Apply additional ghost trail/glow effect
        composite = create_ghost_trail(composite.astype(np.uint8), self.trail_accumulator)

        cv2.putText(
            composite,
            f"Frames Processed: {self.frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return composite.astype(np.uint8)