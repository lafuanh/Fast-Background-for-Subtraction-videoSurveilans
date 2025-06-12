# fbs_abl.py

import cv2
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
from utils import create_ghost_trail


class FBSProcessor:
    """
    Enhanced Fast Background Subtraction with Adaptive Block Learning
    Now includes timestamp tracking for forensic analysis
    """

    def __init__(self, block_size: int = 8, threshold: int = 25, trail_opacity: float = 0.7,
                 start_datetime: Optional[datetime] = None, fps: int = 30):
        """
        Initialize the FBS processor with timestamp capabilities

        Args:
            block_size: Size of blocks for processing (default: 8x8)
            threshold: Motion detection threshold
            trail_opacity: Opacity of movement trails
            start_datetime: Video start time for timestamp calculation
            fps: Frames per second of the video
        """
        self.block_size = block_size
        self.threshold = threshold
        self.trail_opacity = trail_opacity
        self.start_datetime = start_datetime or datetime.now()
        self.fps = fps

        # Background model variables
        self.background_model = None
        self.background_initialized = False
        self.trail_accumulator = None
        self.frame_height = None
        self.frame_width = None
        self.last_frame = None
        self.frame_count = 0

        # Timestamp tracking
        self.motion_timeline: List[Tuple[datetime, float]] = []
        self.motion_intensity_threshold = 5.0  # Minimum intensity to record as motion event

        # Enhanced visualization data
        self.timestamp_overlay_positions: List[Tuple[datetime, int, int]] = []

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

    def _calculate_frame_timestamp(self, frame_number: int) -> datetime:
        """
        Calculate the timestamp for a specific frame

        Args:
            frame_number: Frame number in the video

        Returns:
            Timestamp for the frame
        """
        seconds_elapsed = frame_number / self.fps
        return self.start_datetime + timedelta(seconds=seconds_elapsed)

    def _detect_motion_blocks(self, frame: np.ndarray) -> Tuple[np.ndarray, float, List[Tuple[int, int]]]:
        """
        Detect motion using block-based comparison and return motion centers

        Args:
            frame: Current frame (BGR)

        Returns:
            Tuple of (binary mask, motion intensity percentage, list of motion centers)
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        motion_centers = []
        total_blocks = self.blocks_y * self.blocks_x
        motion_blocks = 0

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
                    motion_blocks += 1

                    # Record motion center for timestamp overlay
                    center_x = x_start + self.block_size // 2
                    center_y = y_start + self.block_size // 2
                    motion_centers.append((center_x, center_y))

                    # Simple adaptive learning for gradual changes
                    self.background_model[by, bx] = (0.95 * self.background_model[by, bx] + 0.05 * current_mean)

        # Post-processing to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        # Calculate motion intensity as percentage
        motion_intensity = (motion_blocks / total_blocks) * 100 if total_blocks > 0 else 0

        return motion_mask, motion_intensity, motion_centers

    def _accumulate_trail(self, motion_mask: np.ndarray, timestamp: datetime,
                          motion_centers: List[Tuple[int, int]]) -> None:
        """
        Accumulate motion trail over time with timestamp information

        Args:
            motion_mask: Binary mask of detected motion
            timestamp: Current frame timestamp
            motion_centers: List of motion center coordinates
        """
        trail_color = np.array([0, 0, 255], dtype=np.float32)  # BGR format for red

        if np.any(motion_mask):
            blurred_mask = cv2.GaussianBlur(motion_mask, (15, 15), 0).astype(np.float32) / 255.0

            for c in range(3):
                self.trail_accumulator[:, :, c] = np.maximum(
                    self.trail_accumulator[:, :, c],
                    blurred_mask * trail_color[c] * self.trail_opacity
                )

            # Store timestamp overlay positions for significant motion centers
            if motion_centers:
                # Select a representative motion center (e.g., the one closest to center of frame)
                frame_center_x, frame_center_y = self.frame_width // 2, self.frame_height // 2
                closest_center = min(motion_centers,
                                     key=lambda p: (p[0] - frame_center_x) ** 2 + (p[1] - frame_center_y) ** 2)
                self.timestamp_overlay_positions.append((timestamp, closest_center[0], closest_center[1]))

        # Apply decay to create a fading effect
        self.trail_accumulator *= 0.98

    def process_frame(self, frame: np.ndarray, frame_number: int) -> None:
        """
        Process a single frame with timestamp tracking

        Args:
            frame: Input frame (BGR)
            frame_number: Frame number for timestamp calculation
        """
        if not self.background_initialized:
            self._initialize_background(frame)
            return

        # Calculate timestamp for this frame
        timestamp = self._calculate_frame_timestamp(frame_number)

        # Detect motion and get intensity
        motion_mask, motion_intensity, motion_centers = self._detect_motion_blocks(frame)

        # Record motion event if intensity is above threshold
        if motion_intensity >= self.motion_intensity_threshold:
            self.motion_timeline.append((timestamp, motion_intensity))

        # Accumulate trail with timestamp info
        self._accumulate_trail(motion_mask, timestamp, motion_centers)

        self.last_frame = frame.copy()
        self.frame_count += 1

    def get_motion_timeline(self) -> List[Tuple[datetime, float]]:
        """
        Get the timeline of detected motion events

        Returns:
            List of tuples containing (timestamp, motion_intensity)
        """
        return self.motion_timeline.copy()

    def get_trail_visualization(self) -> Optional[np.ndarray]:
        """
        Get final trail visualization with timestamp overlays

        Returns:
            Composite image with trails and timestamps or None if not initialized
        """
        if self.last_frame is None or self.trail_accumulator is None:
            return None

        composite = self.last_frame.astype(np.float32)
        trail_mask = np.sum(self.trail_accumulator, axis=2) > 1.0

        # Blend trails onto the composite image
        composite[trail_mask] = composite[trail_mask] * 0.3 + self.trail_accumulator[trail_mask] * 0.7
        composite = np.clip(composite, 0, 255)

        # Apply ghost trail effect
        composite = create_ghost_trail(composite.astype(np.uint8), self.trail_accumulator)

        # Add timestamp overlays at motion locations
        self._add_timestamp_overlays(composite)

        # Add processing information
        info_text = [
            f"Frames Processed: {self.frame_count}",
            f"Motion Events: {len(self.motion_timeline)}",
            f"Video Start: {self.start_datetime.strftime('%H:%M:%S')}"
        ]

        if self.motion_timeline:
            first_motion = min(self.motion_timeline, key=lambda x: x[0])
            last_motion = max(self.motion_timeline, key=lambda x: x[0])
            info_text.extend([
                f"First Motion: {first_motion[0].strftime('%H:%M:%S')}",
                f"Last Motion: {last_motion[0].strftime('%H:%M:%S')}"
            ])

        # Draw information text
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(
                composite,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            # Add black outline for better readability
            cv2.putText(
                composite,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                4,
                cv2.LINE_AA
            )

        return composite.astype(np.uint8)

    def _add_timestamp_overlays(self, composite: np.ndarray) -> None:
        """
        Add timestamp overlays at motion detection locations

        Args:
            composite: Composite image to add overlays to
        """
        # Limit the number of timestamp overlays to avoid cluttering
        max_overlays = 10
        if len(self.timestamp_overlay_positions) > max_overlays:
            # Sample evenly distributed timestamps
            step = len(self.timestamp_overlay_positions) // max_overlays
            selected_positions = self.timestamp_overlay_positions[::step]
        else:
            selected_positions = self.timestamp_overlay_positions

        for timestamp, x, y in selected_positions:
            # Format timestamp for display
            time_str = timestamp.strftime('%H:%M:%S')

            # Calculate text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )

            # Ensure timestamp doesn't go off-screen
            x = max(text_width // 2, min(x, composite.shape[1] - text_width // 2))
            y = max(text_height + 5, min(y, composite.shape[0] - 5))

            # Draw semi-transparent background for timestamp
            overlay = composite.copy()
            cv2.rectangle(
                overlay,
                (x - text_width // 2 - 3, y - text_height - 3),
                (x + text_width // 2 + 3, y + baseline + 3),
                (0, 0, 0),
                -1
            )
            cv2.addWeighted(composite, 0.7, overlay, 0.3, 0, composite)

            # Draw timestamp text
            cv2.putText(
                composite,
                time_str,
                (x - text_width // 2, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),  # Cyan color for visibility
                1,
                cv2.LINE_AA
            )