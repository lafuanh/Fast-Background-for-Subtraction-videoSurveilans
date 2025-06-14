# Advanced Motion Analysis System 🚀

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0+-red)
![License](https://img.shields.io/badge/License-Educational/Research-green)

A cool tool to turn basic motion detection into a forensic analysis beast! Get interactive timelines, frame-by-frame breakdowns, heatmaps, and stats. Perfect for security, traffic, or wildlife monitoring! 🕵️‍♂️🚗🦒

Demo: https://fast-background-for-subtraction-videosurveilans.streamlit.app/
---

## What's Inside? 

### Key Features
- **Interactive Timeline**: See motion intensity, click to jump to events, and zoom into time ranges.
- **Frame-by-Frame**: Check original frames, motion boxes, and optical flow vibes.
- **Motion Heatmap**: Spot high-activity zones with 3D plot options.
- **Stats Galore**: Hourly patterns, motion consistency, and periodicity scores.
- **Export Goodies**: JSON, CSV, PDF reports, or videos with overlays.

---

## Installation 

```bash
# Grab the goodies
pip install streamlit>=1.28.0 opencv-python>=4.8.0 numpy>=1.24.0 pandas>=2.0.0 plotly>=5.17.0 matplotlib>=3.7.0 scipy>=1.10.0 Pillow>=10.0.0
```

# Fire it up
```bash
streamlit run app.py
```

## How to Use
- Upload Video: Toss in MP4, AVI, MOV, or MKV files.
- Set Timeline: Add the real recording time for spot-on timestamps.
- Pick Mode: Quick Analysis, Forensic Analysis, or Real-time
- Tweak Settings: Play with block size, motion threshold, etc.
- Explore: Dive into timeline, stats, or heatmap tabs.

## Tips 
- Security: Use Forensic mode, export timelines for reports.
- Traffic: Enable motion trails, check hourly patterns.
- Wildlife: Lower thresholds for small critters, track paths.

## Troubleshooting 
- Slow?: Lower resolution, skip frames, or use Quick mode.
- False Positives?: Bump up motion threshold or area.
- Missing Motion?: Lower threshold, check video quality.

## System Requirements 
Python 3.8+
4GB RAM (8GB is better)
GPU for big videos (optional)


## License 
Free for educational/research use. No warranties, just fun!
