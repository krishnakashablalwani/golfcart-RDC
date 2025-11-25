# High-Performance PC Setup Guide - Golf Cart Face Recognition

This guide details the setup for the high-performance Linux unit using dual Global Shutter cameras.

## 1. Hardware Components

*   **Cameras (x2)**: Arducam OV9782 (Color) Global Shutter USB.
    *   *Why*: Global shutter eliminates "jello effect" when the cart is moving.
*   **Bandwidth Fix**: PCIe x1 to USB 3.0 Expansion Card.
    *   *Purpose*: To provide a dedicated USB controller for the second camera to prevent bandwidth saturation.
*   **CPU**: Intel Core i3-12100 (12th Gen).
*   **Motherboard**: H610M Micro-ATX.
*   **RAM**: 16GB DDR4.
*   **Storage**: 1TB NVMe SSD (Gen 3).
*   **OS**: Ubuntu Linux 22.04 LTS or 24.04 LTS.

## 2. Hardware Assembly & Bandwidth Management

1.  **Install Expansion Card**: Insert the PCIe x1 to USB 3.0 card into the motherboard.
2.  **Camera Connection Strategy**:
    *   **Camera 1**: Plug into a **Motherboard** USB 3.0 port (Rear I/O).
    *   **Camera 2**: Plug into the **PCIe Expansion Card** USB port.
    *   *Reasoning*: Dual uncompressed video streams can saturate a single USB controller. Splitting them ensures stable frame rates.

## 3. Operating System Installation

1.  **Create Bootable USB**: Use [Rufus](https://rufus.ie/) or Etcher to flash **Ubuntu 22.04 LTS Desktop** to a USB drive.
2.  **Install Ubuntu**:
    *   Boot from USB.
    *   Select "Minimal Installation" (removes bloatware).
    *   Install third-party drivers (for Wi-Fi/Graphics).
    *   Create user: `admin` (or your preference).

## 4. System Configuration

Open a terminal and run:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system tools and v4l-utils (for camera management)
sudo apt install -y python3-pip python3-venv python3-opencv git v4l-utils curl
```

### Verify Cameras
Check if both cameras are detected and identify their paths:
```bash
v4l2-ctl --list-devices
```
You should see two entries for Arducam. Note the `/dev/videoX` numbers (usually `video0` and `video2` or similar).

## 5. Database Setup (Docker)

Using Docker ensures the database version is consistent and easy to manage.

1.  **Install Docker**:
    ```bash
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    # Log out and log back in here!
    ```

2.  **Start MongoDB**:
    ```bash
    docker run -d \
      -p 27017:27017 \
      --name mongo \
      --restart always \
      mongo:latest
    ```

## 6. Application Setup

1.  **Setup Directory**:
    ```bash
    mkdir -p ~/golf-cart-system
    # Copy your project files here
    cd ~/golf-cart-system
    ```

2.  **Python Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## 7. Multi-Camera Configuration

Since you have two cameras, you need to configure the system to use them.

1.  **Identify Camera Indexes**:
    Run a quick python script to check which index corresponds to which camera:
    ```python
    import cv2
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            cap.release()
    ```

2.  **Environment Configuration**:
    You may need to run two instances of the recognition script (one for front, one for back) or modify the code to handle two streams.
    
    **Option A: Two Services (Easiest without code changes)**
    Create two `.env` files:
    *   `.env.front`: `CAMERA_INDEX=0`
    *   `.env.back`: `CAMERA_INDEX=2` (or whatever the second index is)

## 8. Deployment

Create Systemd services to run the recognition automatically.

### Service 1 (Front Camera)
`sudo nano /etc/systemd/system/cam-front.service`

```ini
[Unit]
Description=Golf Cart Cam Front
After=network.target docker.service

[Service]
User=admin
WorkingDirectory=/home/admin/golf-cart-system
# Load specific env file or pass env var directly
Environment="CAMERA_INDEX=0"
ExecStart=/home/admin/golf-cart-system/venv/bin/python recognize_face.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Service 2 (Back Camera)
`sudo nano /etc/systemd/system/cam-back.service`

```ini
[Unit]
Description=Golf Cart Cam Back
After=network.target docker.service

[Service]
User=admin
WorkingDirectory=/home/admin/golf-cart-system
Environment="CAMERA_INDEX=2" 
# Ensure this index matches your second camera
ExecStart=/home/admin/golf-cart-system/venv/bin/python recognize_face.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Enable Services
```bash
sudo systemctl daemon-reload
sudo systemctl enable cam-front.service
sudo systemctl enable cam-back.service
sudo systemctl start cam-front.service
sudo systemctl start cam-back.service
```
