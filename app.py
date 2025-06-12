# app.py
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
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
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("👻 Visualisasi Jejak Gerakan")
st.markdown("**Movement Trail Detection using Fast Background Subtraction**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

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
        "2. **Algorithm** detects movement using block-based subtraction\n"
        "3. **Visualization** shows accumulated movement trails\n"
        "4. **Download** the result image"
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

        # Process button
        if st.button("🚀 Process Video", type="primary", use_container_width=True):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize processor
            processor = FBSProcessor(
                block_size=block_size,
                threshold=threshold,
                trail_opacity=trail_opacity
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

                    # Process frame
                    processor.process_frame(frame)
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

            if result_image is not None:
                # Display result
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown("### ✅ Movement Trail Visualization")
                st.image(result_image, channels="BGR", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Convert to downloadable format
                _, buffer = cv2.imencode('.png', result_image)

                # Download button
                st.download_button(
                    label="📥 Download Result Image",
                    data=buffer.tobytes(),
                    file_name="movement_trail.png",
                    mime="image/png",
                    use_container_width=True
                )

                # Statistics
                st.markdown("### 📈 Processing Statistics")
                stat_cols = st.columns(3)
                with stat_cols[0]:
                    st.metric("Frames Processed", processed_frames)
                with stat_cols[1]:
                    st.metric("Frames Skipped", frame_count - processed_frames)
                with stat_cols[2]:
                    st.metric("Processing Ratio", f"{(processed_frames / frame_count) * 100:.1f}%")

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
        st.markdown("**1. Indoor Surveillance**")
        st.markdown("Track movement patterns in hallways or rooms")

    with col2:
        st.markdown("**2. Traffic Analysis**")
        st.markdown("Visualize vehicle flow on roads")

    with col3:
        st.markdown("**3. Sports Analysis**")
        st.markdown("Analyze player movement patterns")

    # Footer
    st.markdown("---")
    st.markdown(
        "🔧 **Technical Details:** This application uses a simplified Fast Background Subtraction "
        "with Adaptive Block Learning (FBS-ABL) algorithm optimized for real-time processing "
        "on limited hardware."
    )