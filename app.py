# app.py

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from fbs_abl import FBSProcessor
from utils import resize_frame_if_needed

# Page configuration
st.set_page_config(
    page_title="Visualisasi Jejak Gerakan - Movement Trail Detection",
    page_icon="👻",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.stTitle {
    color: #1E88E5;
    font-size: 2.5rem !important;
    text-align: center;
    margin-bottom: 1rem;
}
.info-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1E88E5;
    margin-bottom: 1rem;
}
.result-container {
    border: 2px solid #1E88E5;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-top: 1rem;
}
.timestamp-info {
    background-color: #e8f5e8;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #4CAF50;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("👻 Visualisasi Jejak Gerakan")
st.markdown("**Movement Trail Detection with Timestamp Analysis**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Video start time configuration
    st.subheader("📅 Video Timeline")
    start_date = st.date_input("Video Start Date", datetime.now().date())
    start_time = st.time_input("Video Start Time", datetime.now().time())

    # Block size selection
    block_size = st.slider(
        "Block Size (pixels)",
        min_value=4,
        max_value=16,
        value=8,
        step=4,
        help="Smaller blocks = more detail but slower processing"
    )

    # Threshold selection
    threshold = st.slider(
        "Motion Threshold",
        min_value=10,
        max_value=50,
        value=25,
        help="Lower = more sensitive to movement"
    )

    # Trail intensity
    trail_opacity = st.slider(
        "Trail Opacity",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Opacity of movement trails"
    )

    # Max resolution
    max_width = st.selectbox(
        "Maximum Width (pixels)",
        options=[480, 720, 1080, 1920],
        index=1,
        help="Videos will be resized if larger"
    )

    # Frame skip
    frame_skip = st.slider(
        "Process every N frames",
        min_value=1,
        max_value=5,
        value=2,
        help="Skip frames for faster processing"
    )

    st.markdown("---")
    st.info(
        "💡 **Tips:**\n"
        "- Set accurate start time for precise timestamps\n"
        "- Use lower resolution for faster processing\n"
        "- Increase frame skip for long videos\n"
        "- Adjust threshold based on movement speed"
    )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Video")
    st.markdown("Supported formats: MP4, AVI, MOV")
    st.markdown('</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov'],
        help="Maximum recommended duration: 30 seconds"
    )

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 🎯 How it works")
    st.markdown(
        "1. **Upload** a video file\n"
        "2. **Set** video start time for accurate timestamps\n"
        "3. **Algorithm** detects movement with time tracking\n"
        "4. **Visualization** shows trails with exact timestamps\n"
        "5. **Download** result with motion timeline"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Process video when uploaded
if uploaded_file is not None:
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        # Video information
        cap = cv2.VideoCapture(tmp_file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        # Calculate video timeline
        video_start_datetime = datetime.combine(start_date, start_time)
        video_end_datetime = video_start_datetime + timedelta(seconds=duration)

        # Display video info
        st.markdown("### 📊 Video Information")
        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("Resolution", f"{width}×{height}")
        with info_cols[1]:
            st.metric("Duration", f"{duration:.1f}s")
        with info_cols[2]:
            st.metric("FPS", fps)
        with info_cols[3]:
            st.metric("Frames", total_frames)

        # Display timeline info
        st.markdown('<div class="timestamp-info">', unsafe_allow_html=True)
        st.markdown("### ⏰ Video Timeline")
        timeline_cols = st.columns(2)
        with timeline_cols[0]:
            st.write(f"**Start Time:** {video_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        with timeline_cols[1]:
            st.write(f"**End Time:** {video_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Process button
        if st.button("🚀 Process Video", type="primary", use_container_width=True):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize processor with timestamp info
            processor = FBSProcessor(
                block_size=block_size,
                threshold=threshold,
                trail_opacity=trail_opacity,
                start_datetime=video_start_datetime,
                fps=fps
            )

            # Process video
            cap = cv2.VideoCapture(tmp_file_path)
            frame_count = 0
            processed_frames = 0

            status_text.text("Initializing background model...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames if specified
                if frame_count % frame_skip == 0:
                    # Resize if needed
                    frame = resize_frame_if_needed(frame, max_width)

                    # Process frame with timestamp
                    processor.process_frame(frame, frame_count)
                    processed_frames += 1

                # Update progress
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}...")

                frame_count += 1

            cap.release()

            # Get final result
            status_text.text("Generating movement trail visualization...")
            result_image = processor.get_trail_visualization()
            motion_timeline = processor.get_motion_timeline()

            if result_image is not None:
                # Display result
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown("### ✅ Movement Trail Visualization")
                st.image(result_image, channels="BGR", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Display motion timeline
                if motion_timeline:
                    st.markdown("### 📋 Motion Detection Timeline")
                    st.markdown('<div class="timestamp-info">', unsafe_allow_html=True)

                    # Group motion events by minute for better readability
                    motion_summary = {}
                    for timestamp, intensity in motion_timeline:
                        minute_key = timestamp.strftime('%Y-%m-%d %H:%M')
                        if minute_key not in motion_summary:
                            motion_summary[minute_key] = []
                        motion_summary[minute_key].append((timestamp, intensity))

                    # Display summary
                    if motion_summary:
                        st.write("**🚨 Motion Detected at:**")
                        for minute, events in sorted(motion_summary.items()):
                            max_intensity = max(event[1] for event in events)
                            first_event = min(events, key=lambda x: x[0])
                            last_event = max(events, key=lambda x: x[0])

                            if len(events) == 1:
                                st.write(
                                    f"• **{first_event[0].strftime('%H:%M:%S')}** - Intensity: {max_intensity:.1f}%")
                            else:
                                st.write(
                                    f"• **{first_event[0].strftime('%H:%M:%S')} - {last_event[0].strftime('%H:%M:%S')}** - Peak Intensity: {max_intensity:.1f}% ({len(events)} events)")
                    else:
                        st.write("**✅ No significant motion detected in the video**")

                    st.markdown('</div>', unsafe_allow_html=True)

                # Convert to downloadable format
                _, buffer = cv2.imencode('.png', result_image)

                # Create downloadable timeline report
                timeline_report = f"""Motion Detection Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Video Duration: {duration:.1f} seconds
Total Frames: {total_frames}
Processed Frames: {processed_frames}

Video Timeline:
Start: {video_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}
End: {video_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}

Motion Events:
"""

                if motion_timeline:
                    for timestamp, intensity in motion_timeline:
                        timeline_report += f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} - Intensity: {intensity:.1f}%\n"
                else:
                    timeline_report += "No significant motion detected.\n"

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Result Image",
                        data=buffer.tobytes(),
                        file_name=f"movement_trail_{video_start_datetime.strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )

                with col2:
                    st.download_button(
                        label="📊 Download Timeline Report",
                        data=timeline_report.encode('utf-8'),
                        file_name=f"motion_report_{video_start_datetime.strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                # Statistics
                st.markdown("### 📈 Processing Statistics")
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Frames Processed", processed_frames)
                with stat_cols[1]:
                    st.metric("Frames Skipped", frame_count - processed_frames)
                with stat_cols[2]:
                    st.metric("Processing Ratio", f"{(processed_frames / frame_count) * 100:.1f}%")
                with stat_cols[3]:
                    motion_events = len(motion_timeline) if motion_timeline else 0
                    st.metric("Motion Events", motion_events)

                # Clear progress
                progress_bar.empty()
                status_text.empty()
            else:
                st.error("❌ Failed to generate visualization. Please try again.")

    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

else:
    # Show sample usage
    st.markdown("### 🎬 Sample Usage")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔒 Security Surveillance**")
        st.markdown("Track intruder movements with exact timestamps for forensic analysis")

    with col2:
        st.markdown("**🚗 Traffic Monitoring**")
        st.markdown("Analyze vehicle patterns with precise timing data")

    with col3:
        st.markdown("**⚽ Sports Analysis**")
        st.markdown("Study player movements with timeline correlation")

# Footer
st.markdown("---")
st.markdown(
    "🔧 **Technical Details:** This application uses Fast Background Subtraction "
    "with Adaptive Block Learning (FBS-ABL) algorithm enhanced with timestamp tracking "
    "for forensic and security analysis applications."
)