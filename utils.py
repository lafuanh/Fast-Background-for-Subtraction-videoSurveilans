# utils.py

import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple


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
    Create an enhanced ghost trail effect with a soft glow and better visibility.

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
        # Create multiple levels of glow for better ghost effect
        glow_levels = [
            (21, 0.4, [0.3, 0.8, 0.5]),  # Outer glow - cyan-ish
            (15, 0.6, [0.2, 0.6, 0.9]),  # Middle glow - blue-ish
            (9, 0.8, [0.1, 0.4, 1.0])  # Inner glow - bright blue
        ]

        final_glow = np.zeros_like(composite, dtype=np.float32)

        for kernel_size, alpha, color_weights in glow_levels:
            # Create glow with current parameters
            glow = cv2.GaussianBlur(trail_intensity, (kernel_size, kernel_size), 0)
            glow = cv2.GaussianBlur(glow, (kernel_size, kernel_size), 0)

            # Normalize glow to the range [0, 255]
            max_glow = np.max(glow)
            if max_glow > 0:
                glow_normalized = (glow / max_glow * 255).astype(np.uint8)

                # Create colored glow
                glow_colored = np.zeros_like(composite, dtype=np.float32)
                glow_colored[:, :, 0] = glow_normalized * color_weights[0]  # Blue
                glow_colored[:, :, 1] = glow_normalized * color_weights[1]  # Green
                glow_colored[:, :, 2] = glow_normalized * color_weights[2]  # Red

                # Accumulate glow layers
                final_glow += glow_colored * alpha

        # Blend the accumulated glow with the composite image
        composite = composite.astype(np.float32)
        composite = cv2.addWeighted(composite, 1.0, final_glow, 0.6, 0)
        composite = np.clip(composite, 0, 255)

    return composite.astype(np.uint8)


def format_motion_timeline_report(motion_timeline: List[Tuple[datetime, float]],
                                  video_start: datetime,
                                  video_duration: float) -> str:
    """
    Format motion timeline data into a comprehensive report.

    Args:
        motion_timeline: List of motion events with timestamps and intensities
        video_start: Video start timestamp
        video_duration: Video duration in seconds

    Returns:
        Formatted report string
    """
    if not motion_timeline:
        return "No motion detected during the video period."

    # Sort events by timestamp
    sorted_timeline = sorted(motion_timeline, key=lambda x: x[0])

    # Calculate statistics
    total_events = len(sorted_timeline)
    max_intensity = max(event[1] for event in sorted_timeline)
    avg_intensity = sum(event[1] for event in sorted_timeline) / total_events

    # Group events by time periods (useful for long videos)
    time_periods = {}
    for timestamp, intensity in sorted_timeline:
        period_key = timestamp.strftime('%H:%M')  # Group by minute
        if period_key not in time_periods:
            time_periods[period_key] = []
        time_periods[period_key].append((timestamp, intensity))

    # Build report
    report_lines = [
        "=== MOTION DETECTION ANALYSIS REPORT ===",
        f"Analysis Period: {video_start.strftime('%Y-%m-%d %H:%M:%S')} - {(video_start + timedelta(seconds=video_duration)).strftime('%H:%M:%S')}",
        f"Total Motion Events: {total_events}",
        f"Peak Motion Intensity: {max_intensity:.1f}%",
        f"Average Motion Intensity: {avg_intensity:.1f}%",
        "",
        "=== DETAILED TIMELINE ===",
    ]

    for period, events in sorted(time_periods.items()):
        if len(events) == 1:
            timestamp, intensity = events[0]
            report_lines.append(f"{timestamp.strftime('%H:%M:%S')} - Motion detected (Intensity: {intensity:.1f}%)")
        else:
            first_event = min(events, key=lambda x: x[0])
            last_event = max(events, key=lambda x: x[0])
            max_period_intensity = max(event[1] for event in events)
            report_lines.append(
                f"{first_event[0].strftime('%H:%M:%S')} - {last_event[0].strftime('%H:%M:%S')} - {len(events)} motion events (Peak: {max_period_intensity:.1f}%)")

    return "\n".join(report_lines)


def calculate_motion_statistics(motion_timeline: List[Tuple[datetime, float]]) -> dict:
    """
    Calculate comprehensive statistics from motion timeline data.

    Args:
        motion_timeline: List of motion events with timestamps and intensities

    Returns:
        Dictionary containing various motion statistics
    """
    if not motion_timeline:
        return {
            'total_events': 0,
            'duration_with_motion': 0,
            'max_intensity': 0,
            'avg_intensity': 0,
            'motion_periods': []
        }

    sorted_timeline = sorted(motion_timeline, key=lambda x: x[0])

    # Basic statistics
    total_events = len(sorted_timeline)
    intensities = [event[1] for event in sorted_timeline]
    max_intensity = max(intensities)
    avg_intensity = sum(intensities) / total_events

    # Calculate motion periods (continuous motion segments)
    motion_periods = []
    current_period_start = sorted_timeline[0][0]
    current_period_end = sorted_timeline[0][0]

    for i in range(1, len(sorted_timeline)):
        current_time = sorted_timeline[i][0]
        prev_time = sorted_timeline[i - 1][0]

        # If gap is more than 30 seconds, consider it a new period
        if (current_time - prev_time).total_seconds() > 30:
            motion_periods.append((current_period_start, current_period_end))
            current_period_start = current_time
        current_period_end = current_time

    # Add the last period
    motion_periods.append((current_period_start, current_period_end))

    # Calculate total duration with motion
    total_motion_duration = sum(
        (end - start).total_seconds() for start, end in motion_periods
    )

    return {
        'total_events': total_events,
        'duration_with_motion': total_motion_duration,
        'max_intensity': max_intensity,
        'avg_intensity': avg_intensity,
        'motion_periods': motion_periods,
        'first_motion': sorted_timeline[0][0],
        'last_motion': sorted_timeline[-1][0]
    }