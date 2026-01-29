# Beam Eye Tracker – Attention Analytics & Heatmap Video Generator

A Python-based attention analytics system built on the **Beam Eye Tracker SDK** that captures real-time gaze data, generates heatmap videos, and produces **quantifiable metrics** for attention, focus, and distraction.

This tool is designed for **research, education, UX analysis, and human–computer interaction (HCI)** studies where objective measurement of visual attention is required.

---

## 📌 Features

- 🎯 **Attention Score (0–100)** based on gaze validity and tracking consistency  
- 👁️ **Real-time gaze heatmap video** with dwell-time visualization  
- 🚨 **Distraction detection** with timestamps and duration analysis  
- 📊 **Spatial attention analysis**
  - Gaze dispersion  
  - Fixation stability  
  - Focus quality score  
- ⏱️ **Temporal engagement tracking** over time intervals  
- 📁 Automatic participant/session folder management  
- 📄 Exported **JSON analytics report** and **human-readable TXT report**  
- 🎥 MP4 video output with overlays (time, gaze count, attention score)

---

## 🧠 How It Works

1. The Beam Eye Tracker SDK streams gaze data in real time.
2. Valid and invalid gaze samples are recorded and timestamped.
3. Gaze points are accumulated and converted into a heatmap.
4. Attention metrics are calculated using:
   - Tracking validity
   - Gaze dispersion
   - Fixation stability
   - Engagement over time
5. Results are saved as:
   - Heatmap video (`.mp4`)
   - Structured analytics (`.json`)
   - Readable attention report (`.txt`)

---

## 📂 Output Structure

attention_analysis/
└── Participant_1/
├── heatmap_YYYYMMDD_HHMMSS.mp4
├── analytics_YYYYMMDD_HHMMSS.json
└── report_YYYYMMDD_HHMMSS.txt

yaml
Copy code

Each recording session is automatically assigned a new participant folder.

---

## 📊 Attention Metrics Explained

| Metric | Description |
|------|------------|
| Attention Score | Overall attention level (0–100) |
| Validity Rate | Percentage of valid gaze samples |
| Distraction Periods | Continuous loss of tracking beyond a threshold |
| Gaze Dispersion | How spread out gaze points are |
| Fixation Stability | How steady the gaze is over time |
| Focus Quality | Combined spatial attention metric |
| Engagement Timeline | Attention level across time intervals |

---

## 🛠 Requirements

### Software
- Python **3.8+**
- Beam Eye Tracker SDK
- Beam Eye Tracker app **running and calibrated**

### Python Libraries
```bash
pip install numpy opencv-python scipy pandas matplotlib pillow
🚀 Installation
Install the Beam Eye Tracker SDK

Ensure the Beam Eye Tracker app is running and calibrated

Clone this repository:

bash
Copy code
git clone https://github.com/yourusername/beam-attention-analytics.git
cd beam-attention-analytics
Install dependencies:

bash
Copy code
pip install -r requirements.txt
▶️ Usage
Run the script:

bash
Copy code
python beam_attention_analytics.py
You will be prompted to:

Enter your screen resolution

Set the recording duration (seconds)

Begin eye-tracking and recording

Press Ctrl + C to stop early.

-------------------------------------
📈 Example Console Output
yaml
Copy code
Overall Attention Score:     92.3/100
Validity Rate:               98.1%
Total Distractions:          1
Time Distracted:             3.2%
Focus Quality Score:         88.7/100

-------------------------------------
🎓 Use Cases
Educational attention monitoring

Cognitive and behavioral research

UX / UI usability testing

Human–Computer Interaction (HCI)

Assistive technology research

Student focus and engagement studies

⚠️ Important Notes
The Beam Eye Tracker must be calibrated before recording.

Lighting conditions and face visibility affect accuracy.

Attention metrics are indicators, not medical diagnoses.

🧪 Limitations
Requires Beam Eye Tracker hardware and SDK

Heatmap accuracy depends on sampling rate and calibration quality

Not intended for clinical or diagnostic use

📜 License
This project is provided for educational and research purposes.
Please ensure compliance with the Beam Eye Tracker SDK license when using or distributing this software.
