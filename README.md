# StrokeSense

**Experimental Multimodal Screening of Stroke Warning Signs Using Facial, Speech, and Movement Analysis**

StrokeSense is a university research prototype that combines three machine-learning modules:

- **Face:** Facial landmark and asymmetry analysis.
- **Speech:** Acoustic feature analysis for dysarthria-related classification.
- **Movement:** Pose-based movement completeness classification.
- **Fusion:** Experimental combination of the three model scores.

> **Research disclaimer:** This prototype is not a medical diagnostic tool. The combined score is not a validated probability of stroke.

## Project Structure

```text
STROKE_PROJECT/
├── src/
│   ├── face/           # Facial inference
│   ├── speech/         # Speech inference
│   ├── movement/       # Movement inference
│   └── fusion/         # Multimodal pipeline
├── experiments/
│   ├── face/           # Face training and experiments
│   ├── speech/         # Speech training and experiments
│   └── movement/       # Movement training and experiments
├── models/             # Trained models and MediaPipe tasks
├── tests/              # Integration tests
├── data/               # Local datasets (not committed)
├── outputs/            # Generated results (not committed)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11
- Dependencies listed in requirements.txt

## Installation

From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run Backend

```bash
python -m src.fusion.run_pipeline \
  --image path/to/face.jpg \
  --audio path/to/speech.wav \
  --video path/to/movement.mp4
```

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Frontend Integration

The Streamlit frontend can call run_pipeline() from src.fusion.run_pipeline.

## Limitations

This is an experimental research prototype, not a medical diagnostic system. The fusion score is not a validated stroke probability.
