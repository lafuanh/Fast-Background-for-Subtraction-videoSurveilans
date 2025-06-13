# utils.py

import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import io
import base64


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


def extract_frame_at_time(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """
    Extract a specific frame from video.

    Args:
        video_path: Path to video file
        frame_number: Frame number to extract

    Returns:
        Frame as numpy array or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def calculate_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """
    Calculate dense optical flow between two frames.

    Args:
        prev_gray: Previous frame in grayscale
        curr_gray: Current frame in grayscale

    Returns:
        Optical flow field
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow


def create_ghost_trail(composite: np.ndarray, trail_accumulator: np.ndarray) -> np.ndarray:
    """
    Create an enhanced ghost trail effect with multiple glow layers.

    Args:
        composite: Base composite image
        trail_accumulator: Accumulated trail data

    Returns:
        Enhanced composite with ghost trail effect
    """
    # Extract trail intensity
    if len(trail_accumulator.shape) == 3:
        trail_intensity = np.max(trail_accumulator, axis=2)
    else:
        trail_intensity = trail_accumulator

    max_intensity = np.max(trail_intensity)

    if max_intensity > 0:
        # Create multiple glow layers
        glow_configs = [
            {'kernel': 25, 'alpha': 0.3, 'color': [0.4, 0.8, 1.0]},  # Outer cyan glow
            {'kernel': 15, 'alpha': 0.5, 'color': [0.2, 0.6, 1.0]},  # Middle blue glow
            {'kernel': 7, 'alpha': 0.7, 'color': [0.1, 0.4, 1.0]},  # Inner bright blue
            {'kernel': 3, 'alpha': 1.0, 'color': [0.0, 0.2, 1.0]}  # Core
        ]

        final_glow = np.zeros_like(composite, dtype=np.float32)

        for config in glow_configs:
            kernel_size = config['kernel']
            alpha = config['alpha']
            color = config['color']

            # Create Gaussian blur
            glow = cv2.GaussianBlur(trail_intensity, (kernel_size, kernel_size), 0)

            # Normalize and apply color
            if np.max(glow) > 0:
                glow_normalized = (glow / np.max(glow) * 255).astype(np.uint8)

                # Create colored glow
                glow_colored = np.zeros_like(composite, dtype=np.float32)
                for i in range(3):
                    glow_colored[:, :, i] = glow_normalized * color[i]

                # Accumulate with alpha
                final_glow += glow_colored * alpha

        # Blend with original
        composite = composite.astype(np.float32)
        composite = cv2.addWeighted(composite, 0.8, final_glow, 0.2, 0)
        composite = np.clip(composite, 0, 255)

    return composite.astype(np.uint8)


def generate_motion_report_pdf(analysis_data: Dict, output_path: str) -> None:
    """
    Generate a comprehensive PDF report of motion analysis.

    Args:
        analysis_data: Dictionary containing all analysis results
        output_path: Path to save PDF report
    """
    with PdfPages(output_path) as pdf:
        # Page 1: Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Motion Analysis Report', fontsize=16, fontweight='bold')

        # Timeline plot
        motion_timeline = analysis_data['motion_timeline']
        if motion_timeline:
            timestamps = [event[0] for event in motion_timeline]
            intensities = [event[1] for event in motion_timeline]

            ax1.plot(timestamps, intensities, 'b-', linewidth=2)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Motion Intensity (%)')
            ax1.set_title('Motion Intensity Timeline')
            ax1.grid(True, alpha=0.3)

        # Statistics
        stats = analysis_data['statistics']
        stats_text = f"""Total Events: {stats['total_events']}
Motion Duration: {stats['total_motion_duration']:.1f}s
Average Intensity: {stats['avg_intensity']:.1f}%
Peak Intensity: {stats['max_intensity']:.1f}%
Motion Coverage: {stats['motion_coverage']:.1f}%
Activity Periods: {stats['activity_periods']}"""

        ax2.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                 transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.5",
                                                    facecolor="lightgray"))
        ax2.set_title('Analysis Statistics')
        ax2.axis('off')

        # Heatmap
        heatmap = analysis_data.get('heatmap')
        if heatmap is not None:
            im = ax3.imshow(heatmap, cmap='hot', aspect='auto')
            ax3.set_title('Motion Activity Heatmap')
            ax3.set_xlabel('X Position')
            ax3.set_ylabel('Y Position')
            plt.colorbar(im, ax=ax3, label='Activity Level (%)')

        # Motion distribution by hour (if applicable)
        if motion_timeline:
            hours = [event[0].hour for event in motion_timeline]
            hour_counts = np.bincount(hours, minlength=24)

            ax4.bar(range(24), hour_counts, color='skyblue', edgecolor='navy')
            ax4.set_xlabel('Hour of Day')
            ax4.set_ylabel('Motion Events')
            ax4.set_title('Motion Distribution by Hour')
            ax4.set_xticks(range(0, 24, 4))
            ax4.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 2: Detailed motion regions
        if 'motion_data' in analysis_data and analysis_data['motion_data']:
            fig, axes = plt.subplots(3, 3, figsize=(11, 8.5))
            fig.suptitle('Key Motion Events', fontsize=16, fontweight='bold')

            # Sample up to 9 key events
            motion_events = analysis_data['motion_data']
            sample_indices = np.linspace(0, len(motion_events) - 1,
                                         min(9, len(motion_events)), dtype=int)

            for idx, (ax, event_idx) in enumerate(zip(axes.flat, sample_indices)):
                event = motion_events[event_idx]

                # Create visualization placeholder
                ax.text(0.5, 0.5, f"Frame {event['frame']}\n"
                                  f"{event['timestamp'].strftime('%H:%M:%S')}\n"
                                  f"Intensity: {event['intensity']:.1f}%\n"
                                  f"Regions: {len(event['regions'])}",
                        ha='center', va='center', fontsize=10,
                        transform=ax.transAxes)

                ax.set_title(f"Event {event_idx + 1}", fontsize=10)
                ax.axis('off')

            # Hide unused subplots
            for idx in range(len(sample_indices), 9):
                axes.flat[idx].axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()


def create_motion_visualization_grid(frames: List[np.ndarray],
                                     motion_masks: List[np.ndarray],
                                     timestamps: List[datetime],
                                     grid_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
    """
    Create a grid visualization of frames with motion overlays.

    Args:
        frames: List of original frames
        motion_masks: List of motion masks
        timestamps: List of timestamps
        grid_size: Grid dimensions (rows, cols)

    Returns:
        Combined grid image
    """
    rows, cols = grid_size

    if not frames:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # Get frame dimensions
    h, w = frames[0].shape[:2]

    # Create grid canvas
    grid_h = h * rows + 10 * (rows - 1)
    grid_w = w * cols + 10 * (cols - 1)
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 128

    # Place frames in grid
    for idx in range(min(len(frames), rows * cols)):
        row = idx // cols
        col = idx % cols

        y_start = row * (h + 10)
        x_start = col * (w + 10)

        # Create composite frame
        composite = frames[idx].copy()

        # Overlay motion mask
        if idx < len(motion_masks):
            mask = motion_masks[idx]
            mask_color = np.zeros_like(composite)
            mask_color[:, :, 2] = mask  # Red channel
            composite = cv2.addWeighted(composite, 0.7, mask_color, 0.3, 0)

        # Add timestamp
        if idx < len(timestamps):
            timestamp_str = timestamps[idx].strftime('%H:%M:%S')
            cv2.putText(composite, timestamp_str, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(composite, timestamp_str, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

        # Place in canvas
        canvas[y_start:y_start + h, x_start:x_start + w] = composite

    return canvas


def analyze_motion_patterns(motion_timeline: List[Tuple[datetime, float]],
                            window_size: int = 60) -> Dict[str, Any]:
    """
    Analyze motion patterns over time windows.

    Args:
        motion_timeline: List of (timestamp, intensity) tuples
        window_size: Window size in seconds for pattern analysis

    Returns:
        Dictionary containing pattern analysis results
    """
    if not motion_timeline:
        return {
            'patterns': [],
            'peak_periods': [],
            'quiet_periods': [],
            'activity_score': 0
        }

    # Sort timeline
    sorted_timeline = sorted(motion_timeline, key=lambda x: x[0])

    # Create time windows
    start_time = sorted_timeline[0][0]
    end_time = sorted_timeline[-1][0]
    total_duration = (end_time - start_time).total_seconds()

    windows = []
    current_time = start_time

    while current_time < end_time:
        window_end = current_time + timedelta(seconds=window_size)

        # Get events in this window
        window_events = [
            (t, i) for t, i in sorted_timeline
            if current_time <= t < window_end
        ]

        if window_events:
            avg_intensity = np.mean([i for _, i in window_events])
            event_count = len(window_events)
            event_density = event_count / window_size

            windows.append({
                'start': current_time,
                'end': window_end,
                'event_count': event_count,
                'avg_intensity': avg_intensity,
                'event_density': event_density,
                'max_intensity': max(i for _, i in window_events)
            })

        current_time = window_end

    # Identify patterns
    if windows:
        avg_density = np.mean([w['event_density'] for w in windows])

        peak_periods = [
            w for w in windows
            if w['event_density'] > avg_density * 1.5
        ]

        quiet_periods = [
            w for w in windows
            if w['event_density'] < avg_density * 0.5
        ]

        # Calculate activity score (0-100)
        activity_score = min(100, avg_density * 1000)
    else:
        peak_periods = []
        quiet_periods = []
        activity_score = 0

    return {
        'patterns': windows,
        'peak_periods': peak_periods,
        'quiet_periods': quiet_periods,
        'activity_score': activity_score,
        'total_duration': total_duration
    }


def generate_forensic_timeline(motion_data: List[Dict],
                               video_info: Dict) -> List[Dict]:
    """
    Generate a detailed forensic timeline with all events.

    Args:
        motion_data: List of motion detection data
        video_info: Video metadata

    Returns:
        List of forensic timeline entries
    """
    timeline = []

    for event in motion_data:
        # Calculate exact timestamp
        frame_time = event['frame'] / video_info['fps']

        # Create forensic entry
        entry = {
            'timestamp': event['timestamp'],
            'frame_number': event['frame'],
            'time_offset': frame_time,
            'motion_intensity': event['intensity'],
            'regions_detected': len(event['regions']),
            'total_motion_area': sum(r['area'] for r in event['regions']),
            'primary_location': None,
            'motion_type': classify_motion_type(event['regions'])
        }

        # Determine primary motion location
        if event['regions']:
            largest_region = max(event['regions'], key=lambda r: r['area'])
            entry['primary_location'] = {
                'x': largest_region['center'][0],
                'y': largest_region['center'][1],
                'width': largest_region['bbox'][2],
                'height': largest_region['bbox'][3]
            }

        timeline.append(entry)

    return timeline


def classify_motion_type(regions: List[Dict]) -> str:
    """
    Classify the type of motion based on region characteristics.

    Args:
        regions: List of motion regions

    Returns:
        Motion type classification
    """
    if not regions:
        return "none"

    # Calculate motion characteristics
    total_area = sum(r['area'] for r in regions)
    num_regions = len(regions)

    # Get bounding box of all motion
    if num_regions == 1:
        # Single object motion
        region = regions[0]
        aspect_ratio = region['bbox'][2] / region['bbox'][3] if region['bbox'][3] > 0 else 1

        if aspect_ratio > 2 or aspect_ratio < 0.5:
            return "linear_motion"
        else:
            return "localized_motion"

    elif num_regions <= 3:
        # Multiple objects or fragmented motion
        return "multi_object_motion"

    else:
        # Many regions - could be noise or complex scene
        if total_area < 1000:
            return "scattered_noise"
        else:
            return "complex_motion"


def calculate_motion_metrics(motion_timeline: List[Tuple[datetime, float]],
                             motion_data: List[Dict]) -> Dict[str, float]:
    """
    Calculate advanced motion metrics for analysis.

    Args:
        motion_timeline: Timeline of motion events
        motion_data: Detailed motion data

    Returns:
        Dictionary of calculated metrics
    """
    if not motion_timeline:
        return {
            'consistency_score': 0,
            'periodicity_score': 0,
            'coverage_uniformity': 0,
            'intensity_variance': 0,
            'motion_complexity': 0
        }

    # Extract data
    timestamps = [t for t, _ in motion_timeline]
    intensities = [i for _, i in motion_timeline]

    # Calculate time intervals
    intervals = []
    for i in range(1, len(timestamps)):
        interval = (timestamps[i] - timestamps[i - 1]).total_seconds()
        intervals.append(interval)

    # Consistency score (how regular the motion is)
    if intervals:
        interval_std = np.std(intervals)
        interval_mean = np.mean(intervals)
        consistency_score = max(0, 100 - (interval_std / interval_mean * 100)) if interval_mean > 0 else 0
    else:
        consistency_score = 0

    # Periodicity score (detect repeating patterns)
    periodicity_score = detect_periodicity(intervals) if len(intervals) > 10 else 0

    # Coverage uniformity (how evenly distributed motion is)
    if motion_data:
        x_positions = []
        y_positions = []

        for event in motion_data:
            for region in event['regions']:
                x_positions.append(region['center'][0])
                y_positions.append(region['center'][1])

        if x_positions and y_positions:
            x_std = np.std(x_positions)
            y_std = np.std(y_positions)

            # Normalize by frame dimensions (assumed 640x480 if not available)
            coverage_uniformity = 100 - min(100, (x_std / 320 + y_std / 240) * 50)
        else:
            coverage_uniformity = 0
    else:
        coverage_uniformity = 0

    # Intensity variance
    intensity_variance = np.var(intensities) if intensities else 0

    # Motion complexity (based on number of regions per event)
    if motion_data:
        regions_per_event = [len(event['regions']) for event in motion_data]
        motion_complexity = np.mean(regions_per_event) * 10  # Scale to 0-100
        motion_complexity = min(100, motion_complexity)
    else:
        motion_complexity = 0

    return {
        'consistency_score': consistency_score,
        'periodicity_score': periodicity_score,
        'coverage_uniformity': coverage_uniformity,
        'intensity_variance': intensity_variance,
        'motion_complexity': motion_complexity
    }


def detect_periodicity(intervals: List[float], max_period: int = 100) -> float:
    """
    Detect periodic patterns in motion intervals.

    Args:
        intervals: List of time intervals between events
        max_period: Maximum period to check

    Returns:
        Periodicity score (0-100)
    """
    if len(intervals) < 10:
        return 0

    # Use autocorrelation to detect periodicity
    intervals_array = np.array(intervals)

    # Normalize
    intervals_normalized = (intervals_array - np.mean(intervals_array)) / np.std(intervals_array)

    # Calculate autocorrelation
    autocorr = np.correlate(intervals_normalized, intervals_normalized, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]

    # Find peaks in autocorrelation
    peaks = []
    for i in range(1, min(len(autocorr) - 1, max_period)):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            peaks.append((i, autocorr[i]))

    if peaks:
        # Find strongest peak
        strongest_peak = max(peaks, key=lambda x: x[1])

        # Score based on peak strength
        periodicity_score = min(100, strongest_peak[1] * 100)
    else:
        periodicity_score = 0

    return periodicity_score


def export_motion_data_json(analysis_data: Dict, output_path: str) -> None:
    """
    Export motion analysis data to JSON format with proper serialization.

    Args:
        analysis_data: Complete analysis data
        output_path: Path to save JSON file
    """
    # Prepare data for JSON serialization
    export_data = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'video_info': analysis_data.get('video_info', {}),
            'analysis_parameters': {
                'algorithm': 'FBS-ABL',
                'version': '2.0'
            }
        },
        'statistics': analysis_data.get('statistics', {}),
        'motion_events': []
    }

    # Convert motion events
    for event in analysis_data.get('motion_data', []):
        json_event = {
            'frame': event['frame'],
            'timestamp': event['timestamp'].isoformat(),
            'intensity': event['intensity'],
            'regions': []
        }

        for region in event['regions']:
            json_region = {
                'center': region['center'],
                'bbox': region['bbox'],
                'area': region['area'],
                'confidence': region['confidence']
            }
            json_event['regions'].append(json_region)

        export_data['motion_events'].append(json_event)

    # Convert numpy arrays to lists
    if 'heatmap' in analysis_data and analysis_data['heatmap'] is not None:
        export_data['heatmap'] = analysis_data['heatmap'].tolist()

    # Write to file
    import json
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)