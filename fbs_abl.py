# fbs_abl.py

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Any
from datetime import datetime, timedelta
from collections import deque
import json
from scipy import ndimage
from scipy.spatial import distance
from utils import create_ghost_trail, calculate_optical_flow


class FBSProcessor:
    """
    Advanced Fast Background Subtraction with Adaptive Block Learning
    """

    def __init__(self, block_size: int = 8, threshold: int = 25, learning_rate: float = 0.05,
                 min_contour_area: int = 200, trail_opacity: float = 0.7,
                 start_datetime: Optional[datetime] = None, fps: int = 30,
                 analysis_mode: str = "Quick Analysis"):
        """
        Initialize FBS processor

        isinya:
            block_size: Size of blocks for processing
            threshold: Motion detection threshold
            learning_rate: Background adaptation rate
            min_contour_area: Minimum area to consider as motion
            trail_opacity: Opacity of movement trails
            start_datetime: Video start time
            fps: Frames per second
            analysis_mode: Analysis depth mode
        """
        self.block_size = block_size
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.min_contour_area = min_contour_area
        self.trail_opacity = trail_opacity
        self.start_datetime = start_datetime or datetime.now()
        self.fps = fps
        self.analysis_mode = analysis_mode

        # Background model
        self.background_model = None
        self.background_variance = None
        self.background_initialized = False

        # Motion tracking
        self.motion_history = deque(maxlen=300)  # Store last 10 seconds at 30fps
        self.motion_accumulator = None
        self.motion_heatmap = None

        # Frame data
        self.frame_height = None
        self.frame_width = None
        self.frame_count = 0
        self.prev_frame = None

        # Advanced tracking
        self.motion_timeline: List[Tuple[datetime, float]] = []
        self.frame_motion_data: Dict[int, Dict] = {}
        self.motion_tracks: Dict[int, List[Tuple[int, int, int]]] = {}  # Track ID -> [(frame, x, y)]
        self.next_track_id = 0

        # Forensic analysis data
        self.motion_regions_history: List[Dict] = []
        self.optical_flow_history = deque(maxlen=10)
        self.background_history = deque(maxlen=5)

        # Statistics
        self.total_motion_pixels = 0
        self.total_processed_pixels = 0
        self.motion_intensity_threshold = 5.0

    def _initialize_background(self, frame: np.ndarray) -> None:
        """
        Initialize background model with variance tracking
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        self.blocks_y = self.frame_height // self.block_size
        self.blocks_x = self.frame_width // self.block_size

        # Initialize background mean and variance
        self.background_model = np.zeros((self.blocks_y, self.blocks_x), dtype=np.float32)
        self.background_variance = np.ones((self.blocks_y, self.blocks_x), dtype=np.float32) * 10

        # Calculate initial background statistics
        for by in range(self.blocks_y):
            for bx in range(self.blocks_x):
                y_start, x_start = by * self.block_size, bx * self.block_size
                y_end, x_end = y_start + self.block_size, x_start + self.block_size
                block = gray_frame[y_start:y_end, x_start:x_end]
                if block.size > 0:
                    self.background_model[by, bx] = np.mean(block)
                    self.background_variance[by, bx] = np.var(block) + 10

        # Initialize tracking arrays
        self.motion_accumulator = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        self.motion_heatmap = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        self.prev_frame = gray_frame
        self.background_initialized = True

        # Store initial background
        self.background_history.append(self.background_model.copy())

    def process_frame_advanced(self, frame: np.ndarray, frame_number: int) -> Dict[str, Any]:
        """
        Process frame with motion detection and analysis

        Returns:
            Dictionary containing motion analysis results
        """
        if not self.background_initialized:
            self._initialize_background(frame)
            return {
                'has_motion': False,
                'timestamp': self._calculate_timestamp(frame_number),
                'intensity': 0,
                'regions': [],
                'contours': []
            }

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Detect motion using adaptive background subtraction
        motion_mask, block_differences = self._adaptive_background_subtraction(gray_frame)

        # Apply morphological operations for noise reduction
        motion_mask = self._apply_morphology(motion_mask)

        # Find motion contours and regions
        contours, regions = self._analyze_motion_regions(motion_mask, frame)

        # Calculate motion intensity
        motion_pixels = np.sum(motion_mask > 0)
        total_pixels = motion_mask.size
        intensity = (motion_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        # Update motion tracking
        timestamp = self._calculate_timestamp(frame_number)
        has_motion = intensity > self.motion_intensity_threshold

        if has_motion:
            self.motion_timeline.append((timestamp, intensity))

            # Track motion objects
            if self.analysis_mode in ["Forensic Analysis", "Real-time Monitoring"]:
                self._track_motion_objects(regions, frame_number)

            # Update heatmap
            self._update_motion_heatmap(motion_mask)

        # Store frame data for later retrieval
        self.frame_motion_data[frame_number] = {
            'mask': motion_mask.copy(),
            'regions': regions,
            'contours': contours,
            'intensity': intensity,
            'timestamp': timestamp
        }

        # Calculate optical flow for advanced analysis
        if self.analysis_mode == "Forensic Analysis" and self.prev_frame is not None:
            flow = calculate_optical_flow(self.prev_frame, gray_frame)
            self.optical_flow_history.append(flow)

        self.prev_frame = gray_frame
        self.frame_count += 1

        return {
            'has_motion': has_motion,
            'timestamp': timestamp,
            'intensity': intensity,
            'regions': regions,
            'contours': contours,
            'motion_mask': motion_mask
        }

    def _adaptive_background_subtraction(self, gray_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform adaptive background subtraction with variance-based thresholding
        """
        motion_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        block_differences = np.zeros((self.blocks_y, self.blocks_x), dtype=np.float32)

        for by in range(self.blocks_y):
            for bx in range(self.blocks_x):
                y_start, x_start = by * self.block_size, bx * self.block_size
                y_end, x_end = y_start + self.block_size, x_start + self.block_size

                block = gray_frame[y_start:y_end, x_start:x_end]
                if block.size == 0:
                    continue

                block_mean = np.mean(block)
                block_var = np.var(block)

                # Calculate normalized difference
                bg_mean = self.background_model[by, bx]
                bg_var = self.background_variance[by, bx]

                # Adaptive threshold based on variance
                adaptive_threshold = self.threshold * np.sqrt(bg_var / 10)
                diff = abs(block_mean - bg_mean)
                block_differences[by, bx] = diff

                if diff > adaptive_threshold:
                    motion_mask[y_start:y_end, x_start:x_end] = 255

                    # Slower adaptation for motion areas
                    alpha = self.learning_rate * 0.1
                else:
                    # Normal adaptation for static areas
                    alpha = self.learning_rate

                # Update background model
                self.background_model[by, bx] = (1 - alpha) * bg_mean + alpha * block_mean
                self.background_variance[by, bx] = (1 - alpha) * bg_var + alpha * block_var

        return motion_mask, block_differences

    def _apply_morphology(self, motion_mask: np.ndarray) -> np.ndarray:
        """
        kasih morphological operations baut clean up motion mask
        """
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel_small)

        # Close gaps in motion regions
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel_large)

        # Final dilation for better connectivity
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.dilate(motion_mask, kernel_dilate, iterations=1)

        return motion_mask

    def _analyze_motion_regions(self, motion_mask: np.ndarray, frame: np.ndarray) -> Tuple[List, List[Dict]]:
        """
        Analyse motion regions sama extrasi detailed information
        """
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        valid_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            valid_contours.append(contour)

            # Calculate region properties
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate region confidence based on density
            region_mask = np.zeros(motion_mask.shape, dtype=np.uint8)
            cv2.drawContours(region_mask, [contour], -1, 255, -1)
            motion_density = np.sum(motion_mask[y:y + h, x:x + w] > 0) / (w * h) if w * h > 0 else 0

            # Extract color histogram for tracking
            roi = frame[y:y + h, x:x + w]
            hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            regions.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area,
                'confidence': motion_density,
                'histogram': hist.tolist(),
                'contour_points': contour.squeeze().tolist() if len(contour) > 0 else []
            })

        return valid_contours, regions

    def _track_motion_objects(self, regions: List[Dict], frame_number: int) -> None:
        """
        Track motion objects across frames
        """
        # Simple tracking based on distance and appearance
        current_centers = [(r['center'][0], r['center'][1]) for r in regions]

        if not self.motion_tracks:
            # Initialize tracks
            for i, region in enumerate(regions):
                self.motion_tracks[self.next_track_id] = [(frame_number, region['center'][0], region['center'][1])]
                self.next_track_id += 1
        else:
            # Match with existing tracks
            unmatched_regions = list(range(len(regions)))

            for track_id, track in list(self.motion_tracks.items()):
                if len(track) > 0:
                    last_frame, last_x, last_y = track[-1]

                    # Skip if track is too old
                    if frame_number - last_frame > 10:
                        continue

                    # Find closest region
                    min_dist = float('inf')
                    best_match = -1

                    for i in unmatched_regions:
                        dist = distance.euclidean((last_x, last_y), current_centers[i])
                        if dist < min_dist and dist < 50:  # Max 50 pixel movement
                            min_dist = dist
                            best_match = i

                    if best_match >= 0:
                        # Update track
                        self.motion_tracks[track_id].append(
                            (frame_number, current_centers[best_match][0], current_centers[best_match][1])
                        )
                        unmatched_regions.remove(best_match)

            # Create new tracks for unmatched regions
            for i in unmatched_regions:
                self.motion_tracks[self.next_track_id] = [
                    (frame_number, current_centers[i][0], current_centers[i][1])
                ]
                self.next_track_id += 1

    def _update_motion_heatmap(self, motion_mask: np.ndarray) -> None:
        """
        Update cumulative motion heatmap
        """
        # Add current motion to heatmap with decay
        self.motion_heatmap = self.motion_heatmap * 0.995  # Decay factor
        self.motion_heatmap += (motion_mask > 0).astype(np.float32) * 0.1

        # Clip values
        self.motion_heatmap = np.clip(self.motion_heatmap, 0, 100)

    def _calculate_timestamp(self, frame_number: int) -> datetime:
        """
        Calculate timestamp for a given frame
        """
        seconds_elapsed = frame_number / self.fps
        return self.start_datetime + timedelta(seconds=seconds_elapsed)

    def visualize_motion_frame(self, frame: np.ndarray, regions: List[Dict],
                               contours: List, show_boxes: bool = True,
                               show_trails: bool = True) -> np.ndarray:
        """
        Visualize motion detection results on a frame
        """
        vis_frame = frame.copy()

        # Draw contours
        if contours:
            cv2.drawContours(vis_frame, contours, -1, (0, 255, 0), 2)

        # Draw bounding boxes and info
        if show_boxes and regions:
            for region in regions:
                x, y, w, h = region['bbox']
                cx, cy = region['center']

                # Draw bounding box
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Draw center point
                cv2.circle(vis_frame, (cx, cy), 5, (0, 0, 255), -1)

                # Add text info
                info_text = f"Area: {region['area']:.0f}"
                cv2.putText(vis_frame, info_text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(vis_frame, info_text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Apply motion trails if enabled
        if show_trails and hasattr(self, 'motion_accumulator'):
            trail_overlay = create_ghost_trail(vis_frame, self.motion_accumulator)
            vis_frame = cv2.addWeighted(vis_frame, 0.7, trail_overlay, 0.3, 0)

        return vis_frame

    def visualize_motion_vectors(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """
        Visualize optical flow motion vectors
        """
        vis_frame = frame.copy()

        if len(self.optical_flow_history) > 0:
            flow = self.optical_flow_history[-1]

            # Sample points for vector visualization
            step = 20
            for y in range(0, frame.shape[0], step):
                for x in range(0, frame.shape[1], step):
                    fx, fy = flow[y, x]

                    # Only show significant motion
                    if np.sqrt(fx ** 2 + fy ** 2) > 1:
                        cv2.arrowedLine(vis_frame, (x, y),
                                        (int(x + fx), int(y + fy)),
                                        (0, 255, 255), 2, tipLength=0.3)

        return vis_frame

    def get_motion_timeline(self) -> List[Tuple[datetime, float]]:
        """
        Get the complete motion timeline
        """
        return self.motion_timeline.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics
        """
        if not self.motion_timeline:
            return {
                'total_events': 0,
                'total_motion_duration': 0,
                'motion_coverage': 0,
                'avg_intensity': 0,
                'max_intensity': 0,
                'intensity_std': 0,
                'first_motion': None,
                'last_motion': None,
                'activity_periods': 0
            }

        intensities = [event[1] for event in self.motion_timeline]
        timestamps = [event[0] for event in self.motion_timeline]

        # Calculate activity periods
        activity_periods = 1
        for i in range(1, len(timestamps)):
            if (timestamps[i] - timestamps[i - 1]).total_seconds() > 10:
                activity_periods += 1

        # Calculate motion coverage
        motion_coverage = (len(self.motion_timeline) / self.frame_count * 100) if self.frame_count > 0 else 0

        return {
            'total_events': len(self.motion_timeline),
            'total_motion_duration': (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0,
            'motion_coverage': motion_coverage,
            'avg_intensity': np.mean(intensities),
            'max_intensity': np.max(intensities),
            'intensity_std': np.std(intensities),
            'first_motion': timestamps[0] if timestamps else None,
            'last_motion': timestamps[-1] if timestamps else None,
            'activity_periods': activity_periods
        }

    def get_motion_heatmap(self) -> Optional[np.ndarray]:
        """
        Get the cumulative motion heatmap
        """
        if self.motion_heatmap is None:
            return None

        # Normalize to 0-100 range
        heatmap = self.motion_heatmap.copy()
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = (heatmap / max_val) * 100

        return heatmap

    def identify_high_activity_regions(self, heatmap: np.ndarray,
                                       threshold: float = 50) -> List[Dict]:
        """
        Identify regions with high motion activity
        """
        # Threshold the heatmap
        binary = (heatmap > threshold).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Minimum region size
                continue

            # Get region properties
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Get peak activity in region
                mask = np.zeros(heatmap.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                peak_activity = np.max(heatmap[mask > 0])

                regions.append({
                    'center': (cx, cy),
                    'size': area,
                    'peak_activity': peak_activity,
                    'contour': contour
                })

        return sorted(regions, key=lambda x: x['peak_activity'], reverse=True)

    def generate_analysis_video(self, input_path: str, motion_data: List[Dict],
                                output_path: str = None, show_timeline: bool = True,
                                show_intensity: bool = True, show_regions: bool = True) -> str:
        """
        Generate video with analysis overlay
        """
        if output_path is None:
            output_path = input_path.replace('.mp4', '_analysis.mp4')

        # Open video
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        motion_dict = {m['frame']: m for m in motion_data}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Add overlays if this frame has motion
            if frame_count in motion_dict:
                motion_info = motion_dict[frame_count]

                # Visualize motion regions
                if show_regions:
                    frame = self.visualize_motion_frame(
                        frame,
                        motion_info['regions'],
                        motion_info['contours'],
                        show_boxes=True,
                        show_trails=False
                    )

                # Add intensity bar
                if show_intensity:
                    intensity = motion_info['intensity']
                    bar_height = int(height * 0.05)
                    bar_width = int(width * 0.3)
                    bar_x = width - bar_width - 20
                    bar_y = 20

                    # Background
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + bar_width, bar_y + bar_height),
                                  (0, 0, 0), -1)

                    # Intensity bar
                    fill_width = int(bar_width * (intensity / 100))
                    color = (0, 255, 0) if intensity < 30 else (0, 255, 255) if intensity < 60 else (0, 0, 255)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + fill_width, bar_y + bar_height),
                                  color, -1)

                    # Text
                    cv2.putText(frame, f"Motion: {intensity:.1f}%",
                                (bar_x, bar_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Add timestamp
                if show_timeline:
                    timestamp_str = motion_info['timestamp'].strftime('%H:%M:%S.%f')[:-3]
                    cv2.putText(frame, timestamp_str,
                                (20, height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, timestamp_str,
                                (20, height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

        return output_path