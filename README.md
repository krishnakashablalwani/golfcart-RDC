# Golf Cart Face Recognition System

Production-grade face recognition for 5,000+ users using ArcFace embeddings plus MediaPipe 3D landmarks, backed by MongoDB.

## Project Structure (top-level)

```
register_face.py          # Register a user (captures 50 samples, normalized embedding)
recognize_face.py         # Real-time recognizer (ArcFace + 3D landmarks)
modules/                  # Core modules (MongoDB, config, parsers)
scripts/                  # DB utilities (clear/import/check)
tests/                    # Debug and test scripts
data/                     # Student info and mappings
Samples/                  # Captured faces (created at runtime)
Detections/               # Detection logs (created at runtime)
requirements.txt
```

## Quick Start (Windows, Git Bash)

```bash
# 1) Activate venv
source .venv/Scripts/activate

# 2) Register a user (50 samples, normalized)
python register_face.py

# 3) Run recognition (3D landmarks + embeddings)
python recognize_face.py

# Quit: press Q in the window
```

## Mode
- `recognize_face.py`: ArcFace + MediaPipe 3D landmarks (70% landmarks / 30% embedding), generous landmark scaling, cooldown disabled, threshold 0.55.

## Configuration
- GPU (default): `USE_GPU=1` (CUDAExecutionProvider). Set `USE_GPU=0` to force CPU.
- Model: Uses ArcFace `buffalo_sc` (MobileFaceNet, 512-D) with ONNX Runtime.
- Database: MongoDB at `localhost:27017` (see `modules/database.py`).
- Thresholds/weights: see `recognize_face.py` for landmark weight and similarity scaling.

## Operational Tips
- Re-register if lighting/angle changed; embeddings will align to current appearance.
- Landmarks auto-accumulate per user; scores rise after a few matches.
- Camera index: default 0; change in the script constructor if needed.
- For large scale, run GPU-enabled ONNX Runtime on each node; one RTX 3060-class GPU can handle multiple streams comfortably.

## Common Tasks
```bash
# Clear all face data
python scripts/clear_database.py

# Import students from Excel
python scripts/manage_database.py  # choose option 1

# Check embeddings
python scripts/check_embeddings.py
```

## Troubleshooting
- If recognition shows UNKNOWN: ensure registration was just taken under similar lighting; re-register if needed.
- If camera fails: try indices 0/1/2; check permissions; verify OpenCV GUI support.
- If accuracy is low: confirm embeddings are normalized (they are in current scripts), and ensure at least 50 good samples per user.