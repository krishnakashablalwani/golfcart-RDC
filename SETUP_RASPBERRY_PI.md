# Raspberry Pi Setup Guide - Golf Cart Face Recognition

This guide details the setup process for deploying the Face Recognition system on a Raspberry Pi 4B or 5.

## 1. Hardware Components

*   **Single Board Computer**: Raspberry Pi 4B (4GB/8GB) or Raspberry Pi 5.
*   **Storage**: 32GB+ MicroSD Card (Class 10 / A1 rated for fast I/O).
*   **Camera**: Raspberry Pi Camera Module (v2/v3) or High-Quality USB Webcam.
*   **Power**: Official USB-C Power Supply (ensure 3A+ for Pi 4, 5A for Pi 5).
*   **Connectivity**: Wi-Fi or Ethernet connection.
*   **Cooling**: Heatsink and Fan case (Critical for running AI models continuously).

## 2. Operating System Installation

1.  **Download Imager**: Install [Raspberry Pi Imager](https://www.raspberrypi.com/software/) on your computer.
2.  **Select OS**:
    *   Click "Choose OS".
    *   Navigate to **Raspberry Pi OS (other)** -> **Raspberry Pi OS (64-bit)**.
    *   *Note: 64-bit is required for better performance with OpenCV and MongoDB.*
3.  **Configure Settings** (Gear Icon):
    *   Set Hostname: `golfcart-pi`
    *   Enable SSH (Password authentication).
    *   Set Username/Password (e.g., `pi`/`securepassword`).
    *   Configure Wi-Fi (SSID and Password).
4.  **Write**: Select your SD card and click "Write".
    
## 3. Initial Configuration

1.  Insert SD card into the Pi and power it on.
2.  Connect via SSH (or open a terminal if using a monitor):
    ```bash
    ssh pi@golfcart-pi.local
    ```
3.  **Update System**:
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```
4.  **Enable Camera**:
    *   Run `sudo raspi-config`.
    *   Go to **Interface Options**.
    *   Enable **Legacy Camera** (if using older Pi Camera) or ensure **I2C** is on.
    *   *For USB Cameras*: No specific config needed, just plug it in.

## 4. Install Dependencies

### System Libraries
Install Python system dependencies and atlas library for NumPy:
```bash
sudo apt install -y python3-pip python3-venv python3-opencv libatlas-base-dev git
```

### Database (MongoDB)
We recommend running MongoDB in Docker for ease of use on the Pi.

1.  **Install Docker**:
    ```bash
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    ```
    *Log out and log back in for group changes to take effect.*

2.  **Run MongoDB**:
    ```bash
    docker run -d \
      -p 27017:27017 \
      --name mongo \
      --restart always \
      mongo:latest
    ```

## 5. Application Setup

1.  **Clone/Copy Code**:
    Transfer your project files to `/home/pi/golf-cart-recognition`.

2.  **Create Virtual Environment**:
    ```bash
    cd ~/golf-cart-recognition
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables**:
    Create a `.env` file:
    ```bash
    nano .env
    ```
    Add:
    ```ini
    MONGODB_URI=mongodb://localhost:27017/
    CAMERA_INDEX=0
    ```

## 6. Deployment (Auto-start)

To ensure the system starts when the Golf Cart is turned on, create a Systemd service.

1.  **Create Service File**:
    ```bash
    sudo nano /etc/systemd/system/face-recognition.service
    ```

2.  **Add Configuration**:
    ```ini
    [Unit]
    Description=Golf Cart Face Recognition
    After=network.target docker.service

    [Service]
    User=pi
    WorkingDirectory=/home/pi/golf-cart-recognition
    ExecStart=/home/pi/golf-cart-recognition/venv/bin/python recognize_face.py
    Restart=always
    RestartSec=10

    [Install]
    WantedBy=multi-user.target
    ```

3.  **Enable and Start**:
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable face-recognition.service
    sudo systemctl start face-recognition.service
    ```

4.  **Check Status**:
    ```bash
    sudo systemctl status face-recognition.service
    ```
