# Driver Drowsiness Detection System

A real-time driver drowsiness detection system using computer vision and deep learning. The system monitors driver's eyes and mouth to detect signs of drowsiness and fatigue, providing timely alerts to prevent accidents.

## Dataset
[Kaggle Drowsiness Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)

## Features

- Real-time face detection and landmark tracking using MediaPipe
- Eye state classification (Open/Closed) using CNN
- Yawn detection
- Eye Aspect Ratio (EAR) monitoring
- Mouth Aspect Ratio (MAR) monitoring
- Audio alerts for drowsiness detection
- Streamlit web interface with live monitoring
- Trip reports with statistics and AI-powered insights
- Data visualization of drowsiness patterns
- Export functionality for trip data (CSV & JSON)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SANKETSINGH6555/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For AI insights, create a `.env` file and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Use the web interface to:
   - Start/Stop trip monitoring
   - View real-time drowsiness detection
   - Generate trip reports
   - Download trip data

## Model Architecture

The system uses a CNN model trained on a custom dataset to classify:
- Eye states (Open/Closed)
- Yawn detection
- Face landmark detection using MediaPipe

## Project Structure

```
|── drowsiness_new7.h5      # Trained CNN-model
|── driver-drowsiness_notebook.ipynb
├── app.py                  # Streamlit web application
├── detect_drowsiness.py    # Core detection logic
├── requirements.txt        # Project dependencies
├── assets/                 # Sound alerts and cascades
├── dataset/                # Contains the dataset 
└── reports/               # Generated trip reports
```

## Technical Details

- **Eye Aspect Ratio (EAR)**: Measures eye openness
- **Mouth Aspect Ratio (MAR)**: Detects yawning
- **Alert Conditions**:
  - Consecutive frames of closed eyes
  - Frequent yawning patterns
  - Combined drowsiness indicators

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request



