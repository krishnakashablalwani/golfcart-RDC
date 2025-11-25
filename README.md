# Golf Cart Face Recognition System

This project implements a face recognition system for a golf cart using a Raspberry Pi. It identifies registered users (students) via a camera and logs their presence.

## Hardware Requirements

*   **Raspberry Pi**: Model 4B (4GB/8GB RAM) or Raspberry Pi 5 recommended for best performance with OpenCV DNN.
*   **Camera**: Raspberry Pi Camera Module (v2 or v3) or a high-quality USB Webcam.
*   **Power Supply**: Reliable USB-C power supply (ensure it can handle the Pi + Camera load).
*   **MicroSD Card**: 32GB+ (Class 10 / A1 rated).
*   **Display (Optional)**: A small HDMI screen if you want to see the live feed/feedback.

## Step 1: Raspberry Pi Setup

1.  **Install OS**:
    *   Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
    *   Choose **Raspberry Pi OS (64-bit)**. The 64-bit version is crucial for better performance with OpenCV and MongoDB.
    *   Write it to your SD card.
2.  **Initial Boot**:
    *   Insert SD card, connect monitor/keyboard/mouse, and power on.
    *   Follow the setup wizard (connect to Wi-Fi, set username/password).
3.  **Enable Camera**:
    *   Open terminal: `sudo raspi-config`
    *   Go to **Interface Options** -> **Legacy Camera** (if using older camera lib) or ensure **I2C/Camera** interfaces are enabled.
    *   *Note: Modern Bullseye/Bookworm OS uses libcamera by default. OpenCV usually works with standard video nodes (`/dev/video0`).*

## Step 2: Install System Dependencies

Open a terminal on the Pi and run:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv python3-opencv libatlas-base-dev
```

### Install MongoDB (Database)
The system uses MongoDB to store registered faces.

**Option A: Local MongoDB (Recommended for offline use)**
1.  Follow the official instructions to install MongoDB Community Edition for Debian/Raspberry Pi OS.
    *   *Note: MongoDB binaries for Pi (ARM64) might require specific versions. An easier alternative for Pi is often using Docker.*
2.  **Alternative**: Use a lightweight Docker container for MongoDB:
    ```bash
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    # Log out and back in
    docker run -d -p 27017:27017 --name mongo --restart always mongo:latest
    ```

**Option B: Cloud MongoDB (Atlas)**
1.  Create a free account on [MongoDB Atlas](https://www.mongodb.com/atlas).
2.  Get your connection string (e.g., `mongodb+srv://user:pass@cluster...`).
3.  You will use this in the `.env` file later.

## Step 3: Project Setup

1.  **Copy the Code**:
    *   Clone this repository or copy the files (`register_face.py`, `recognize_face.py`, `requirements.txt`, etc.) to a folder, e.g., `/home/pi/golf-cart-face-rec`.

2.  **Create Virtual Environment**:
    ```bash
    cd /home/pi/golf-cart-face-rec
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Python Libraries**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**:
    *   Create a `.env` file:
        ```bash
        nano .env
        ```
    *   Paste the configuration (adjust as needed):
        ```ini
        MONGODB_URI=mongodb://localhost:27017/
        DB_NAME=face_recognition_db
        COLLECTION_NAME=registered_faces_ultra
        CAMERA_INDEX=0
        # Email settings (optional)
        SMTP_SERVER=smtp.gmail.com
        SMTP_PORT=587
        SENDER_EMAIL=your_email@gmail.com
        SENDER_PASSWORD=your_app_password
        RECIPIENT_EMAIL=admin@example.com
        ```
    *   Save and exit (`Ctrl+X`, `Y`, `Enter`).

## Step 4: Usage

### 1. Registering Users
You need to register faces before the system can recognize them.
1.  Connect a keyboard/mouse.
2.  Run:
    ```bash
    source .venv/bin/activate
    python register_face.py
    ```
3.  Press **'C'** to capture a face.
4.  Enter the **Roll Number**. The system will look up the name in `students.xlsx` (ensure this file is in the folder).
5.  The system captures 5 samples and saves them.

### 2. Running Recognition
To start the detection system:
```bash
source .venv/bin/activate
# Start the HOD Dashboard (in background or separate terminal)
python hod_server.py &
# Start the Recognition System
python recognize_face.py
```
*   Access the HOD Dashboard at `http://<pi-ip-address>:5000`.
*   Press **'Q'** to quit recognition.

## Step 5: Auto-Start on Boot

To make the recognition system run automatically when the Golf Cart (Pi) turns on:

1.  **Create a Systemd Service for HOD Server**:
    ```bash
    sudo nano /etc/systemd/system/hod-server.service
    ```
    Content:
    ```ini
    [Unit]
    Description=HOD Dashboard Server
    After=network.target

    [Service]
    User=pi
    WorkingDirectory=/home/pi/golf-cart-face-rec
    ExecStart=/home/pi/golf-cart-face-rec/.venv/bin/python /home/pi/golf-cart-face-rec/hod_server.py
    Restart=always

    [Install]
    WantedBy=multi-user.target
    ```

2.  **Create a Systemd Service for Recognition**:
    ```bash
    sudo nano /etc/systemd/system/face-rec.service
    ```
    Content:
    ```ini
    [Unit]
    Description=Golf Cart Face Recognition
    After=network.target video.target hod-server.service

    [Service]
    Type=simple
    User=pi
    WorkingDirectory=/home/pi/golf-cart-face-rec
    Environment=DISPLAY=:0
    Environment=XAUTHORITY=/home/pi/.Xauthority
    ExecStart=/home/pi/golf-cart-face-rec/.venv/bin/python /home/pi/golf-cart-face-rec/recognize_face.py
    Restart=always
    RestartSec=5

    [Install]
    WantedBy=multi-user.target
    ```

3.  **Enable Services**:
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable hod-server.service face-rec.service
    sudo systemctl start hod-server.service face-rec.service
    ```

*Note: If running "headless" (no monitor), `cv2.imshow` might cause errors. You may need to modify `recognize_face.py` to comment out `cv2.imshow` and `cv2.waitKey` lines if no screen is attached.*
