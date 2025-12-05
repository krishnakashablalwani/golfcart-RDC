# Golf Cart Face Recognition - Raspberry Pi Setup

## Quick Setup on Raspberry Pi

### 1. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 (if not already installed)
sudo apt install python3.11 python3.11-venv python3-pip -y

# Install system libraries
sudo apt install -y \
    libopencv-dev python3-opencv \
    libatlas-base-dev \
    mongodb
```

### 2. Clone and Setup

```bash
# Clone repository
git clone https://github.com/krishnakashablalwani/golfcart-RDC.git
cd golfcart-RDC

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 3. Start MongoDB

```bash
# Start MongoDB service
sudo systemctl start mongodb
sudo systemctl enable mongodb
```

### 4. Import Students

```bash
# Make sure Student information.xlsx is in data/ folder
python scripts/manage_database.py
# Choose option 1: Import all students
```

### 5. Register Students

```bash
python register_face.py
```

**Instructions:**
- Press 'C' to capture (15 times per student)
- Vary angles between captures
- Press 'Q' to skip

### 6. Run Recognition

```bash
python recognize_face.py
```

**Controls:**
- 'Q' to quit
- 'R' to reload cache

## Camera Configuration

If camera not working, try different index:
```python
# In register_face_deepface.py or recognize_face_deepface.py
camera_index=0  # Try 0, 1, or 2
```

## Performance Tips for Raspberry Pi

1. **Lower resolution** if slow:
   ```python
   high_res_width=1280
   high_res_height=720
   ```

2. **Process fewer frames**:
   ```python
   process_every_n_frames=3  # or 4
   ```

3. **Use USB camera** instead of Pi Camera for better performance

## Troubleshooting

### Camera Issues
```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera
python -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```

### Memory Issues
```bash
# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### MongoDB Not Starting
```bash
sudo systemctl status mongodb
sudo journalctl -u mongodb -n 50
```

## Files Structure

```
golfcart-RDC/
├── register_face_deepface.py    # Register students
├── recognize_face_deepface.py   # Real-time recognition
├── modules/                      # Core code
├── scripts/                      # Utilities
└── data/                        # Student Excel file
```

## Quick Commands

```bash
# Clear all data and start fresh
python scripts/clear_database.py

# Check system
python scripts/test_installation.py

# Manage database
python scripts/manage_database.py
```

---

Ready to deploy on Raspberry Pi! 🚀
