Beam Eye Tracker – Attention Analytics & Heatmap Video Generator

This Python script uses the Beam Eye Tracker SDK to collect real-time gaze data and transform it into quantifiable attention metrics and a visual heatmap video of user focus across the screen.
The program records eye-tracking data, detects valid and lost tracking states, and analyzes attention through multiple dimensions such as fixation stability, gaze dispersion, distraction periods, and engagement over time.

Key Features
🎯 Attention Scoring (0–100) based on gaze validity and consistency
👁️ Real-time gaze heatmap video with dwell-time visualization
🚨 Distraction detection with timestamps and duration analysis
📊 Spatial analysis (gaze dispersion, fixation stability, focus quality)
⏱️ Temporal engagement tracking over configurable time intervals
📁 Automatic session management with participant folders
📄 Exported JSON and human-readable text reports


Outputs
  MP4 heatmap video of gaze activity
  Detailed analytics report (JSON)
  Formatted attention report (TXT)


Requirements
  Python 3.x (Preferrably 3.11.7)
  Beam Eye Tracker SDK (installed, running, and calibrated)
  numpy, opencv-python, scipy, pandas, matplotlib, pillow


Use Cases
  Attention and focus research
  Educational and cognitive studies
  UX/UI usability testing
  Human–computer interaction (HCI) analysis
