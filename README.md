# Driver Drowsiness Detection System

Real-time drowsiness detection using a laptop webcam — no special hardware required. Detects eye closure and yawning, triggers audio/SMS alerts, and generates AI-powered post-trip safety reports.

Built during internship at **Future Kmuniti Pvt. Ltd., Jaipur** (Jan–Apr 2026) under Mr. Abhishek Maher (CTO).

---

## Features

- Real-time face landmark detection via MediaPipe FaceMesh (468 points)
- EAR (Eye Aspect Ratio) + MAR (Mouth Aspect Ratio) based drowsiness logic
- CNN model (TensorFlow/Keras) for eye/mouth state classification
- Audio alarm on drowsiness detection
- Auto SMS alert to emergency contact via Twilio (max once per 5 min)
- Real-time safety tips sidebar on dashboard
- Post-trip report with drowsy episodes, avg EAR/MAR, session duration
- AI feedback via Meta Llama 3.1 8B (Groq API) — personalized safety coaching
- Export report as CSV or JSON

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python |
| Face landmarks | MediaPipe FaceMesh |
| Video capture | OpenCV |
| AI classification | CNN (TensorFlow/Keras) |
| Dashboard | Streamlit |
| SMS alerts | Twilio API |
| AI feedback | Meta Llama 3.1 8B via Groq API |
| Export | CSV, JSON |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/saurav-jorwal/drowsiness-detection.git
cd drowsiness-detection
```

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the model

The CNN model file (`drowiness_new7.h5`) is not included in the repo due to size.  
Download it from: **[HuggingFace / Google Drive link here]**  
Place it in the root project directory.

### 5. Configure secrets

Create a `.env` file in the root directory (use `.env.example` as template):

```
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+1xxxxxxxxxx
EMERGENCY_CONTACT=+91xxxxxxxxxx
GROQ_API_KEY=your_key
```

### 6. Run

```bash
streamlit run app.py
```

---

## How It Works

1. Webcam captures face at ~25–30 FPS
2. MediaPipe maps 468 facial landmarks per frame
3. EAR calculated — if below 0.22 for 10+ consecutive frames → drowsy
4. MAR calculated — if above 0.90 for 3+ yawns → drowsy
5. CNN model confirms eye/mouth state (Open / Closed / Yawn / No Yawn)
6. On drowsiness: alarm plays, SMS sent, safety tips shown
7. Post-trip: report generated → sent to Llama 3.1 for AI coaching feedback

---

## Project Structure

```
drowsiness-detection/
├── app.py                  # Main Streamlit app
├── drowiness_new7.h5       # CNN model (download separately)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Note

This app requires a physical webcam and must be run locally. It cannot be fully hosted on cloud platforms due to webcam dependency.

---

## Acknowledgements

- Base drowsiness detection logic inspired by open-source community work
- Extended with: Twilio SMS alerts, safety sidebar UI, and Llama 3.1 AI feedback
- Internship project — Future Kmuniti Pvt. Ltd., Jaipur
