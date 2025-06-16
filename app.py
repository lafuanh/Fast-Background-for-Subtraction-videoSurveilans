# app.py

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from fbs_abl import FBSProcessor
from utils import resize_frame_if_needed, extract_frame_at_time
import json
from typing import Dict, List, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="Motion Analysis System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Update CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --tertiary-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --dark-gradient: linear-gradient(135deg, #0c0c0c 0%, #121C47 50%, #9d174d 100%);
        --glass-bg: rgba(255, 255, 255, 0.12);
        --glass-border: rgba(255, 255, 255, 0.18);
        --text-primary: #1a202c;
        --text-secondary: #4a5568;
        --shadow-lg: 0 20px 40px rgba(0, 0, 0, 0.1);
        --shadow-xl: 0 25px 50px rgba(0, 0, 0, 0.15);
        --border-radius: 20px;
        --border-radius-sm: 12px;
    }
    
    .main {
        padding-top: 0rem;
        font-family: 'Poppins', 'Inter';
        letter-spacing: -0.02em;
    }
    
    .stApp {
        background: var(--dark-gradient);
        min-height: 100vh;
        position: relative;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 119, 198, 0.2) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    .main-header {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 3rem 2rem;
        border-radius: var(--border-radius);
        margin: 1rem 0 2rem 0;
        box-shadow: var(--shadow-xl);
        text-align: center;
        border: 1px solid var(--glass-border);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: var(--primary-gradient);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .main-title {
        color: var(--text-primary);
        font-size: 3.5rem !important;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .main-subtitle {
        color: var(--text-secondary);
        font-size: 1.25rem;
        font-weight: 400;
        margin: 0;
        opacity: 0.9;
        font-family: 'Inter', sans-serif;
    }
    
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: var(--border-radius);
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow-lg);
        margin-bottom: 1.5rem;
        color: #D6DCF5;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-xl);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .metric-card {
        background: var(--primary-gradient);
        color: white;
        padding: 0.5rem 0.2rem;
        border-radius: var(--border-radius);
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: rgba(255, 255, 255, 0.3);
    }
    
    .metric-card:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.5);
    }
    
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        text-shadow: 0 2px 3px rgba(0, 0, 0, 0.2);
    }
    
    .metric-label {
        font-size: 0.6rem;
        opacity: 0.95;
        font-weight: 400;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.95rem;
        backdrop-filter: blur(10px);
        border: 1px solid;
        transition: all 0.3s ease;
    }
    
    .status-success {
        background: rgba(72, 187, 120, 0.15);
        color: #2f855a;
        border-color: rgba(72, 187, 120, 0.3);
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.2);
    }
    
    .status-processing {
        background: rgba(246, 173, 85, 0.15);
        color: #c05621;
        border-color: rgba(246, 173, 85, 0.3);
        box-shadow: 0 4px 15px rgba(246, 173, 85, 0.2);
    }
    
    .timeline-container {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-xl);
        border: 1px solid var(--glass-border);
        position: relative;
        overflow: hidden;
    }
    
    .timeline-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        animation: rotate 20s linear infinite;
        z-index: -1;
    }
    
    @keyframes rotate {
        100% { transform: rotate(360deg); }
    }
    
    .tab-header {
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        border-radius: var(--border-radius-sm);
        padding: 0.75rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--glass-border);
    }
    
    .sidebar .stSelectbox > div > div {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border-radius: var(--border-radius-sm);
        border: 1px solid var(--glass-border);
        transition: all 0.3s ease;
        color: #9d174d;
    }
    
    .sidebar .stSelectbox > div > div:hover {
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }
    
    .progress-container {
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        padding: 3rem 2rem;
        border-radius: var(--border-radius);
        margin: 1.5rem 0;
        text-align: center;
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow-lg);
    }
    
    .video-info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .info-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: var(--border-radius-sm);
        text-align: center;
        border: 1px solid var(--glass-border);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .info-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--primary-gradient);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .info-card:hover::before {
        transform: scaleX(1);
    }
    
    .section-title {
        color: #5468BD;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.55rem;
        position: relative;
        padding-left: 1rem;
    }
    
    .section-title::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 30px;
        background: var(--primary-gradient);
        border-radius: 2px;
    }
    
    .export-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 1rem;
    }
    
    .export-card {
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        padding: 2rem 1.5rem;
        border-radius: var(--border-radius);
        text-align: center;
        border: 2px solid var(--glass-border);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .export-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--primary-gradient);
        opacity: 0;
        transition: opacity 0.3s ease;
        z-index: -1;
    }
    
    .export-card:hover {
        border-color: #667eea;
        transform: translateY(-6px) scale(1.02);
        box-shadow: var(--shadow-xl);
        color: white;
    }
    
    .export-card:hover::before {
        opacity: 0.9;
    }
    
    .export-card h3 {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        transition: color 0.3s ease;
    }
    
    .export-card p {
        opacity: 0.8;
        font-size: 0.95rem;
        line-height: 1.5;
        transition: opacity 0.3s ease;
    }
    
    .export-card:hover p {
        opacity: 1;
    }
    
    /* Sidebar Improvements */
    .sidebar .element-container {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border-radius: var(--border-radius-sm);
        margin-bottom: 1rem;
        border: 1px solid var(--glass-border);
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fcf, #6a5acd);
    }
    
    /* Button Animations */
    .stButton > button {
        background: var(--primary-gradient) !important;
        border: none !important;
        border-radius: 50px !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Metrics */
    .stMetric {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: var(--border-radius-sm);
        border: 1px solid var(--glass-border);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem !important;
        }
        
        .metric-value {
            font-size: 2rem;
        }
        
        .export-grid {
            grid-template-columns: 1fr;
        }
        
        .video-info-grid {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        }
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'selected_frame' not in st.session_state:
    st.session_state.selected_frame = None
if 'motion_regions' not in st.session_state:
    st.session_state.motion_regions = {}


# Main header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">Visualisasi Jejak Gerakan Analysis System</h1>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("Analysis Configuration")

    # Analysis mode selection
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Quick Analysis", "Forensic Analysis", "Real-time Monitoring"],
        help="Select the depth of analysis"
    )

    # Video timeline settings
    st.subheader("Video Timeline")

    start_date = st.date_input("Video Start Date", datetime.now().date())
    start_time = st.time_input("Video Start Time", datetime.now().time())
    st.caption("(waktu auto ganti sesuai input video)")
    # Algorithm parameters 
    with st.expander("Algorithm Parameters", expanded=True):
        st.markdown("**Detection Settings**")
        
        col1, col2 = st.columns(2)
        with col1:
            block_size = st.slider(
                "Block Size",
                min_value=4, max_value=32, value=8, step=4,
                help="Smaller blocks = more detail but slower"
            )
        with col2:
            threshold = st.slider(
                "Sensitivity",
                min_value=5, max_value=50, value=20,
                help="Lower = more sensitive detection"
            )

        st.markdown("**Learning Parameters**")
        col1, col2 = st.columns(2)
        with col1:
            learning_rate = st.slider(
                "Learning Rate",
                min_value=0.001, max_value=0.1, value=0.05, format="%.3f",
                help="How quickly the background model adapts"
            )
        with col2:
            min_contour_area = st.slider(
                "Min. Area",
                min_value=50, max_value=1000, value=200,
                help="Filter out small movements (pixels²)"
            )

    # Visualization settings
    with st.expander("Visualization Settings"):
        col1, col2 = st.columns(2)
        with col1:
            show_motion_boxes = st.checkbox("📦 Motion Boxes", value=True)
            show_motion_trails = st.checkbox("✨ Motion Trails", value=True)
        with col2:
            show_heatmap = st.checkbox("🔥 Heat Map", value=False)
            trail_decay_rate = st.slider("Trail Decay", 0.9, 0.99, 0.95, format="%.2f")

    # Export settings
    st.markdown("---")
    st.markdown("### Export Options")
    export_format = st.selectbox(
        "Export Format",
        ["📄 JSON Report", "📊 CSV Data", "📋 PDF Report", "🎬 Video with Overlay"]
    )

# Main content area
main_container = st.container()

with main_container:
     # Video upload section 
    st.markdown('<div class="glass-card">Visualisasi Intelligent Video Surveillance with Real-time Motion Detection & Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="section-title">Video Input</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drop your video file here or click to browse",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Maximum recommended size: 500MB",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown('<div class="section-title">📊 Quick Stats</div>', unsafe_allow_html=True)
        
        if st.session_state.analysis_data:
            stats = st.session_state.analysis_data.get('statistics', {})
            
            # Custom metric cards
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats.get('total_events', 0)}</div>
                <div class="metric-label">Motion Events</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb, #f5576c);">
                <div class="metric-value">{stats.get('max_intensity', 0):.1f}%</div>
                <div class="metric-label">Peak Intensity</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe, #00f2fe);">
                <div class="metric-value">{stats.get('motion_coverage', 0):.1f}%</div>
                <div class="metric-label">Coverage</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Upload a video to see live statistics")

    st.markdown('</div>', unsafe_allow_html=True)

# Process video when uploaded
if uploaded_file is not None:
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
        st.session_state.video_path = tmp_file_path

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

    # Display video information
    st.markdown("###Video Information")
    info_cols = st.columns(5)
    with info_cols[0]:
        st.metric("Resolution", f"{width}×{height}")
    with info_cols[1]:
        st.metric("Duration", f"{duration:.1f}s")
    with info_cols[2]:
        st.metric("FPS", fps)
    with info_cols[3]:
        st.metric("Total Frames", total_frames)
    with info_cols[4]:
        st.metric("Bitrate", f"{os.path.getsize(tmp_file_path) * 8 / duration / 1000:.0f} kbps")

    # Process button with different analysis modes
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Analysis", type="primary", use_container_width=True):
            # Progress container
            #st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            st.markdown("### 🔄 Processing Video...")

            # Initialize processor with advanced parameters
            processor = FBSProcessor(
                block_size=block_size,
                threshold=threshold,
                learning_rate=learning_rate,
                min_contour_area=min_contour_area,
                start_datetime=video_start_datetime,
                fps=fps,
                analysis_mode=analysis_mode
            )

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            ta_text = st.empty()

            # Process video
            cap = cv2.VideoCapture(tmp_file_path)
            frame_count = 0
            start_time_process = datetime.now()

            # Store frames for later retrieval
            key_frames = {}
            motion_data = []

            status_text.markdown('<div class="status-indicator status-processing">Initializing motion detection...</div>', unsafe_allow_html=True)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                motion_info = processor.process_frame_advanced(frame, frame_count)

                # Store motion data
                if motion_info['has_motion']:
                    motion_data.append({
                        'frame': frame_count,
                        'timestamp': motion_info['timestamp'],
                        'intensity': motion_info['intensity'],
                        'regions': motion_info['regions'],
                        'contours': motion_info['contours']
                    })

                    # Store key frames (every 10th motion frame)
                    if len(motion_data) % 10 == 0:
                        key_frames[frame_count] = frame.copy()

                # Update progress
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                
                # Calculate ETA
                if frame_count > 0:
                    elapsed = (datetime.now() - start_time_process).total_seconds()
                    eta = (elapsed / frame_count) * (total_frames - frame_count)
                    ta_text.markdown(f"⏱ {eta:.0f}s | Motion events detected: {len(motion_data)}")

                    status_text.markdown(f'<div class="status-indicator status-processing"> Frame {frame_count:,}/{total_frames:,} ({progress*100:.1f}%)</div>', unsafe_allow_html=True)

                frame_count += 1

            cap.release()

            # Store analysis results
            st.session_state.processor = processor
            st.session_state.analysis_data = {
                'motion_timeline': processor.get_motion_timeline(),
                'motion_data': motion_data,
                'key_frames': key_frames,
                'statistics': processor.get_statistics(),
                'heatmap': processor.get_motion_heatmap(),
                'video_info': {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'duration': duration,
                    'total_frames': total_frames
                }
            }

            # Success message
            progress_bar.empty()
            status_text.markdown('<div class="status-indicator status-success">✅ Analysis completed successfully!</div>', unsafe_allow_html=True)
            ta_text.empty()
            st.balloons()
            st.markdown('</div>', unsafe_allow_html=True)
            st.rerun()

# Display analysis results if available
if st.session_state.analysis_data:
    st.markdown('<div class="timeline-container">', unsafe_allow_html=True)
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Timeline Analysis",
        "Frame Analysis",
        "Motion Heatmap",
        "Statistical Analysis",
        "Export Results"
    ])

    with tab1:
        st.markdown('<div class="section-title">📈 Interactive Motion Timeline</div>', unsafe_allow_html=True)

        # Prepare timeline data
        motion_timeline = st.session_state.analysis_data['motion_timeline']
        if motion_timeline:
            # Create DataFrame for plotting
            df = pd.DataFrame(motion_timeline, columns=['timestamp', 'intensity'])
            df['seconds'] = [(t - df['timestamp'].iloc[0]).total_seconds() for t in df['timestamp']]

            # Create interactive timeline plot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Motion Intensity Over Time", "Motion Events Distribution"),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )

            # Main timeline plot
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['intensity'],
                    mode='lines+markers',
                    name='Motion Intensity',
                    line=dict(color='#1976d2', width=2),
                    marker=dict(size=6),
                    hovertemplate='<b>Time:</b> %{x|%H:%M:%S}<br><b>Intensity:</b> %{y:.1f}%<extra></extra>',
                    fill='tonexty'
                ),
                row=1, col=1
            )

            # Add threshold line
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Detection Threshold",
                row=1, col=1
            )

            # Motion events histogram
            fig.add_trace(
                go.Histogram(
                    x=df['timestamp'],
                    name='Event Frequency',
                    marker_color='#155255',
                    nbinsx=50
                ),
                row=2, col=1
            )

            fig.update_layout(
                height=600,
                showlegend=True,
                hovermode='x unified',
                xaxis_rangeslider_visible=False,
                font=dict(family="Inter, poppins")
            )

            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Intensity (%)", row=1, col=1)
            fig.update_yaxes(title_text="Events", row=2, col=1)

            # Display plot
            selected_point = st.plotly_chart(fig, use_container_width=True, key="timeline_plot")

            # Frame selection based on timeline
            st.markdown("### Select Time Range for Detailed Analysis")
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                start_time_select = st.time_input(
                    "Start Time",
                    value=df['timestamp'].iloc[0].time() if len(df) > 0 else datetime.now().time()
                )

            with col2:
                end_time_select = st.time_input(
                    "End Time",
                    value=df['timestamp'].iloc[-1].time() if len(df) > 0 else datetime.now().time()
                )

            with col3:
                if st.button(" Analyze Range", use_container_width=True):
                    selected_data = df[
                        (df['timestamp'].dt.time >= start_time_select) &
                        (df['timestamp'].dt.time <= end_time_select)
                    ]
                    
                    if not selected_data.empty:
                        st.success(f" Found {len(selected_data)} motion events in selected range")

                    # Show detailed stats for selected range
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Events", len(selected_data))
                    with col2:
                        st.metric("Avg Intensity", f"{selected_data['intensity'].mean():.1f}%")
                    with col3:
                        st.metric("Max Intensity", f"{selected_data['intensity'].max():.1f}%")
                    with col4:
                        st.metric("Duration",
                                  f"{selected_data['seconds'].iloc[-1] - selected_data['seconds'].iloc[0]:.1f}s")

    with tab2:
        st.markdown('<div class="section-title"> Frame-by-Frame Analysis</div>', unsafe_allow_html=True)

        motion_data = st.session_state.analysis_data['motion_data']

        if motion_data:
            # Frame selector
            selected_event = st.selectbox(
                "Select Motion Event",
                range(len(motion_data)),
                format_func=lambda
                    x: f"Frame {motion_data[x]['frame']} - {motion_data[x]['timestamp'].strftime('%H:%M:%S')} - Intensity: {motion_data[x]['intensity']:.1f}%"
            )

            if selected_event is not None:
                event_data = motion_data[selected_event]
                frame_number = event_data['frame']

                # Extract frame from video
                frame = extract_frame_at_time(st.session_state.video_path, frame_number)

                if frame is not None:
                    # Create visualization columns
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Original Frame")
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                    with col2:
                        st.markdown("#### Motion Detection Result")
                        # Process frame to show motion
                        if st.session_state.processor:
                            motion_viz = st.session_state.processor.visualize_motion_frame(
                                frame,
                                event_data['regions'],
                                event_data['contours'],
                                show_boxes=show_motion_boxes,
                                show_trails=show_motion_trails
                            )
                            st.image(cv2.cvtColor(motion_viz, cv2.COLOR_BGR2RGB), use_container_width=True)

                    # Show additional analysis
                    st.markdown("#### Motion Regions Analysis")

                    regions_df = pd.DataFrame([
                        {
                            'Region': i + 1,
                            'Area': region['area'],
                            'Center X': region['center'][0],
                            'Center Y': region['center'][1],
                            'Confidence': region['confidence']
                        }
                        for i, region in enumerate(event_data['regions'])
                    ])

                    if not regions_df.empty:
                        st.dataframe(regions_df, use_container_width=True)

                        # Motion vector visualization
                        if st.checkbox("Show Motion Vectors"):
                            vector_viz = st.session_state.processor.visualize_motion_vectors(
                                frame,
                                frame_number
                            )
                            st.image(cv2.cvtColor(vector_viz, cv2.COLOR_BGR2RGB), use_container_width=True)

    with tab3:
        st.markdown('<div class="section-title"> Motion Heatmap Analysis</div>', unsafe_allow_html=True)

        heatmap = st.session_state.analysis_data.get('heatmap')

        if heatmap is not None:
            # Create heatmap visualization
            fig = go.Figure(data=go.Heatmap(
                z=heatmap,
                colorscale='Hot',
                reversescale=True,
                hovertemplate='X: %{x}<br>Y: %{y}<br>Activity: %{z:.1f}%<extra></extra>'
            ))

            fig.update_layout(
                title="Cumulative Motion Activity Heatmap",
                xaxis_title="X Position",
                yaxis_title="Y Position",
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            # 3D surface plot option
            if st.checkbox("Show 3D Surface Plot"):
                fig_3d = go.Figure(data=[go.Surface(
                    z=heatmap,
                    colorscale='Hot',
                    reversescale=True
                )])

                fig_3d.update_layout(
                    title="3D Motion Activity Surface",
                    scene=dict(
                        xaxis_title="X Position",
                        yaxis_title="Y Position",
                        zaxis_title="Activity Level"
                    ),
                    height=700
                )

                st.plotly_chart(fig_3d, use_container_width=True)

            # Region of Interest (ROI) analysis
            st.markdown("### Region of Interest Analysis")

            col1, col2 = st.columns(2)
            with col1:
                roi_threshold = st.slider(
                    "Activity Threshold",
                    min_value=0,
                    max_value=100,
                    value=50,
                    help="Identify regions with activity above this threshold"
                )

            with col2:
                if st.button("Identify High Activity Regions"):
                    high_activity_regions = st.session_state.processor.identify_high_activity_regions(
                        heatmap,
                        threshold=roi_threshold
                    )

                    if high_activity_regions:
                        st.success(f"Found {len(high_activity_regions)} high activity regions")

                        # Display regions
                        for i, region in enumerate(high_activity_regions):
                            st.write(f"**Region {i + 1}:** Center: ({region['center'][0]}, {region['center'][1]}), "
                                     f"Size: {region['size']}px², Peak Activity: {region['peak_activity']:.1f}%")

    with tab4:
        st.markdown('<div class="section-title">Statistical Analysis</div>', unsafe_allow_html=True)

        stats = st.session_state.analysis_data['statistics']

        # Overall statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Motion Statistics")
            st.metric("Total Motion Events", stats['total_events'])
            st.metric("Motion Duration", f"{stats['total_motion_duration']:.1f}s")
            st.metric("Motion Coverage", f"{stats['motion_coverage']:.1f}%")

        with col2:
            st.markdown("#### Intensity Statistics")
            st.metric("Average Intensity", f"{stats['avg_intensity']:.1f}%")
            st.metric("Peak Intensity", f"{stats['max_intensity']:.1f}%")
            st.metric("Std Deviation", f"{stats['intensity_std']:.1f}%")

        with col3:
            st.markdown("#### Temporal Statistics")
            st.metric("First Motion", stats['first_motion'].strftime('%H:%M:%S') if stats['first_motion'] else "N/A")
            st.metric("Last Motion", stats['last_motion'].strftime('%H:%M:%S') if stats['last_motion'] else "N/A")
            st.metric("Activity Periods", stats['activity_periods'])

        # Detailed charts
        st.markdown("### Motion Pattern Analysis")

        # Motion intensity distribution
        motion_timeline = st.session_state.analysis_data['motion_timeline']
        if motion_timeline:
            intensities = [event[1] for event in motion_timeline]

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=intensities,
                nbinsx=30,
                name='Intensity Distribution',
                marker_color='#1976d2'
            ))

            fig_dist.update_layout(
                title="Motion Intensity Distribution",
                xaxis_title="Intensity (%)",
                yaxis_title="Frequency",
                height=400
            )

            st.plotly_chart(fig_dist, use_container_width=True)

        # Motion patterns by time
        if st.checkbox("Show Hourly Motion Patterns"):
            df = pd.DataFrame(motion_timeline, columns=['timestamp', 'intensity'])
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute

            hourly_stats = df.groupby('hour').agg({
                'intensity': ['mean', 'max', 'count']
            }).round(1)

            fig_hourly = go.Figure()

            fig_hourly.add_trace(go.Bar(
                x=hourly_stats.index,
                y=hourly_stats[('intensity', 'count')],
                name='Event Count',
                yaxis='y',
                marker_color='#42a5f5'
            ))

            fig_hourly.add_trace(go.Scatter(
                x=hourly_stats.index,
                y=hourly_stats[('intensity', 'mean')],
                name='Avg Intensity',
                yaxis='y2',
                line=dict(color="#b0281f", width=3)
            ))

            fig_hourly.update_layout(
                title="Motion Patterns by Hour",
                xaxis_title="Hour of Day",
                yaxis=dict(title="Event Count", side='left'),
                yaxis2=dict(title="Average Intensity (%)", side='right', overlaying='y'),
                height=400,
                hovermode='x unified'
            )

            st.plotly_chart(fig_hourly, use_container_width=True)

    with tab5:
        st.markdown('<div class="section-title">📤 Export Analysis Results</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="export-grid">
            <div class="export-card">
                <h3>📄 JSON Report</h3>
                <p>Detailed analysis data in JSON format</p>
            </div>
            <div class="export-card">
                <h3>📊 CSV Data</h3>
                <p>Motion events in spreadsheet format</p>
            </div>
            <div class="export-card">
                <h3>📋 PDF Report</h3>
                <p>Professional analysis report</p>
            </div>
            <div class="export-card">
                <h3>🎬 Analysis Video</h3>
                <p>Video with motion overlay</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Prepare export data
        export_data = {
            'video_info': st.session_state.analysis_data['video_info'],
            'statistics': st.session_state.analysis_data['statistics'],
            'motion_events': [
                {
                    'frame': event['frame'],
                    'timestamp': event['timestamp'].isoformat(),
                    'intensity': event['intensity'],
                    'regions_count': len(event['regions'])
                }
                for event in st.session_state.analysis_data['motion_data']
            ]
        }

        col1, col2 = st.columns(2)

        with col1:
            if export_format == "JSON":
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="Download JSON Report",
                    data=json_str,
                    file_name=f"motion_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

            elif export_format == "CSV":
                # Convert to CSV format
                df = pd.DataFrame(export_data['motion_events'])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV Report",
                    data=csv,
                    file_name=f"motion_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col2:
            if st.button("Generate Analysis Video", use_container_width=True):
                with st.spinner("Generating video with analysis overlay..."):
                    # Generate video with overlays
                    output_path = st.session_state.processor.generate_analysis_video(
                        st.session_state.video_path,
                        st.session_state.analysis_data['motion_data'],
                        show_timeline=True,
                        show_intensity=True,
                        show_regions=show_motion_boxes
                    )

                    if output_path and os.path.exists(output_path):
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="Download Analysis Video",
                                data=f.read(),
                                file_name=f"analysis_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                        os.remove(output_path)

# footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.1); border-radius: 15px; margin-top: 2rem; color: #D6DCF5; font-family: 'Inter', sans-serif;">
    <p> Advanced Motion Analysis System</p>
    <p>Powered by FBS-ABL Algorithm | Version 3.2 | Last updated: {datetime.now().strftime('%Y-%m-%d')}</p>
</div>
""", unsafe_allow_html=True)