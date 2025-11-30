# Raspberry Pi Installation Troubleshooting

## Network Issues During Installation

### Problem: TensorFlow Download Failing
If you see errors like "incomplete-download" or "not enough bytes were received":

**Solution 1: Install in Stages**
```bash
# Activate virtual environment first
source .venv/bin/activate

# Install one package at a time with retries
pip install --retries 10 --timeout 600 numpy
pip install --retries 10 --timeout 600 tensorflow
pip install --retries 10 tf-keras
pip install --retries 10 deepface
pip install --retries 10 opencv-python-headless
pip install pymongo pandas openpyxl python-dotenv
```

**Solution 2: Download Wheel File Manually**
If network is unstable, download on a stable connection and transfer:

1. On a computer with stable internet:
```bash
# Download TensorFlow wheel for ARM64
wget https://files.pythonhosted.org/packages/ea/4c/c1aa90c5cc92e9f7f9c78421e121ef25bae7d378f8d1d4cbad46c6308836/tensorflow-2.20.0-cp313-cp313-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

2. Transfer to Raspberry Pi:
```bash
scp tensorflow-2.20.0-*.whl golfcart@raspberrypi.local:~/
```

3. Install from local file on Pi:
```bash
source ~/golfcart-RDC/.venv/bin/activate
pip install ~/tensorflow-2.20.0-*.whl
pip install tf-keras deepface opencv-python-headless
pip install pymongo pandas openpyxl python-dotenv numpy
```

**Solution 3: Use Different Network/Time**
- Try during off-peak hours when network is less congested
- Connect Pi via Ethernet instead of WiFi
- Try a different internet connection if available

### Problem: Virtual Environment Issues
If venv creation fails or packages conflict:

```bash
# Remove and recreate
rm -rf ~/.cache/pip
rm -rf ~/golfcart-RDC/.venv
python3 -m venv ~/golfcart-RDC/.venv --system-site-packages
source ~/golfcart-RDC/.venv/bin/activate
```

### Problem: Docker MongoDB Issues
If Docker installation or MongoDB container fails:

**Check Docker Status:**
```bash
sudo systemctl status docker
sudo usermod -aG docker $USER
newgrp docker
```

**Start MongoDB Manually:**
```bash
docker pull mongo:latest
docker run -d --name mongodb --restart unless-stopped -p 27017:27017 mongo:latest
```

**Check MongoDB Running:**
```bash
docker ps
docker logs mongodb
```

## Camera Issues

### Check Camera Device
```bash
# List video devices
v4l2-ctl --list-devices

# Test camera
libcamera-hello
```

### Camera Permissions
```bash
# Add user to video group
sudo usermod -aG video $USER
```

## Performance Optimization

If system is slow or runs out of memory:

### Increase Swap Space
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Lower Camera Resolution
Edit `modules/config.py`:
```python
CAMERA_WIDTH = 1280  # Instead of 1920
CAMERA_HEIGHT = 720  # Instead of 1080
```

### Skip Frames in Recognition
```python
# Process every 2nd frame instead of every frame
if frame_count % 2 == 0:
    detect_and_recognize()
```

## Quick Verification

After installation, verify everything works:

```bash
cd ~/golfcart-RDC
source .venv/bin/activate
python scripts/test_installation.py
```

Expected output:
```
✓ All imports successful
✓ DeepFace available
✓ Camera accessible
✓ MongoDB connected
✓ Custom modules working
```

## Getting Help

If issues persist:
1. Check system logs: `journalctl -xe`
2. Check Python errors: Run with `python -v`
3. Check disk space: `df -h`
4. Check memory: `free -h`
