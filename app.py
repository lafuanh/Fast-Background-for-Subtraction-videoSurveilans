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
    page_title="Visualisasi Jejak Gerakan - Analysis System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced UI
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
.analysis-section {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
    margin-bottom: 1rem;
}
.frame-viewer {
    border: 2px solid #1E88E5;
    border-radius: 0.5rem;
    padding: 0.5rem;
    background-color: #ffffff;
}
.motion-stats {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1976d2;
}
.timeline-container {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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

# Title and description
st.title("Visualisasi Jejak Gerakan Analysis System")
st.markdown("**Visualisasi Jejak Gerakan dengan Fast Background Subtraction dengan Interactive Timeline Analysis**")

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
    # Advanced algorithm parameters
    with st.expander("🔧 Algorithm Parameters", expanded=True):
        block_size = st.slider(
            "Block Size (pixels)",
            min_value=4,
            max_value=32,
            value=8,
            step=4,
            help="Smaller blocks = more detail but slower"
        )

        threshold = st.slider(
            "Motion Threshold",
            min_value=5,
            max_value=50,
            value=20,
            help="Lower = more sensitive detection"
        )

        learning_rate = st.slider(
            "Background Learning Rate",
            min_value=0.001,
            max_value=0.1,
            value=0.05,
            format="%.3f",
            help=" How quickly the background model adapts"
        )

        min_contour_area = st.slider(
            "Minimum Motion Area",
            min_value=50,
            max_value=1000,
            value=200,
            help="Filter out small movements (pixels²)"
        )

    # Visualization settings
    with st.expander("Visualization Settings"):
        show_motion_boxes = st.checkbox("Show Motion Bounding Boxes", value=True)
        show_motion_trails = st.checkbox("Show Motion Trails", value=True)
        show_heatmap = st.checkbox("Show Motion Heatmap", value=False)
        trail_decay_rate = st.slider(
            "Trail Decay Rate",
            min_value=0.9,
            max_value=0.99,
            value=0.95,
            format="%.2f"
        )

    # Export settings
    st.markdown("---")
    st.subheader("📤 Export Options")
    export_format = st.selectbox(
        "Export Format",
        ["JSON", "CSV", "PDF Report", "Video with Overlay"]
    )

# Main content area
main_container = st.container()

with main_container:
    # File upload section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("### Video Input")
        uploaded_file = st.file_uploader(
            "masukan video untuk dianalysis",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Maximum recommended size: 500MB"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="motion-stats">', unsafe_allow_html=True)
        st.markdown("### Quick Stats")
        if st.session_state.analysis_data:
            stats = st.session_state.analysis_data.get('statistics', {})
            st.metric("Motion Events", stats.get('total_events', 0))
            st.metric("Peak Intensity", f"{stats.get('max_intensity', 0):.1f}%")
            st.metric("Coverage", f"{stats.get('motion_coverage', 0):.1f}%")
        else:
            st.info("Upload a video to see statistics")
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
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
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

            # Process video
            cap = cv2.VideoCapture(tmp_file_path)
            frame_count = 0

            # Store frames for later retrieval
            key_frames = {}
            motion_data = []

            status_text.text("Initializing advanced motion detection...")

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
                status_text.text(
                    f"Analyzing frame {frame_count}/{total_frames} - Detected {len(motion_data)} motion events")

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

            # Clear progress
            progress_bar.empty()
            status_text.empty()
            st.success("Analysis complete!!!")
            st.rerun()

# Display analysis results if available
if st.session_state.analysis_data:
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Timeline Analysis",
        "Frame Analysis",
        "Motion Heatmap",
        "Statistical Analysis",
        "Export Results"
    ])

    with tab1:
        st.markdown("### Interactive Motion Timeline")

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
                    hovertemplate='<b>Time:</b> %{x|%H:%M:%S}<br><b>Intensity:</b> %{y:.1f}%<extra></extra>'
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
                    marker_color='#42a5f5',
                    nbinsx=50
                ),
                row=2, col=1
            )

            fig.update_layout(
                height=600,
                showlegend=True,
                hovermode='x unified',
                xaxis_rangeslider_visible=False
            )

            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Intensity (%)", row=1, col=1)
            fig.update_yaxes(title_text="Events", row=2, col=1)

            # Display plot
            selected_point = st.plotly_chart(fig, use_container_width=True, key="timeline_plot")

            # Frame selection based on timeline
            st.markdown("### Select Time Range for Detailed Analysis")
            col1, col2 = st.columns(2)

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

            if st.button("Analyze Selected Range"):
                # Filter data for selected range
                selected_data = df[
                    (df['timestamp'].dt.time >= start_time_select) &
                    (df['timestamp'].dt.time <= end_time_select)
                    ]

                if not selected_data.empty:
                    st.info(f"Found {len(selected_data)} motion events in selected range")

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
        st.markdown("### Frame-by-Frame Analysis")

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
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

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
                            st.image(cv2.cvtColor(motion_viz, cv2.COLOR_BGR2RGB), use_column_width=True)

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
                            st.image(cv2.cvtColor(vector_viz, cv2.COLOR_BGR2RGB), use_column_width=True)

    with tab3:
        st.markdown("### Motion Heatmap Analysis")

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
        st.markdown("### Statistical Analysis")

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
                line=dict(color='#f44336', width=3)
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
        st.markdown("### Export Analysis Results")

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

# Footer
st.markdown("---")
st.markdown(
    "**Advanced Motion Analysis System** | "
    "Powered by FBS-ABL Algorithm | "
    f"Version 2.0 | Last updated: {datetime.now().strftime('%Y-%m-%d')}"
)