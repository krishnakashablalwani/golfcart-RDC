# SETUP_PYTHON_311.md

## Optimal Setup for Best Accuracy (Python 3.11 + Azure + DeepFace)

### Why Python 3.11?
- TensorFlow 2.15+ (required by DeepFace) has stable wheels for Python 3.11
- Python 3.13 lacks prebuilt TensorFlow/DeepFace wheels on Windows
- Hybrid Azure + local gives maximum accuracy and reliability

### Step 1: Install Python 3.11

**Windows:**
```bash
# Download Python 3.11.9 from python.org
# Or use winget:
winget install Python.Python.3.11

# Verify:
py -3.11 --version
```

**Linux (Raspberry Pi):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### Step 2: Create New Virtual Environment

```bash
# Remove old venv (backup if needed)
rm -rf .venv

# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate

# Linux / Mac / Git Bash
python3.11 -m venv .venv --system-site-packages
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip setuptools wheel

# Install in order (important for dependency resolution)
pip install numpy==1.24.4
pip install opencv-python==4.9.0.80
pip install tensorflow==2.15.0
pip install keras==2.15.0
pip install tf-keras==2.15.1
pip install deepface==0.0.96
pip install pymongo==4.8.3
pip install pandas==2.2.3
pip install openpyxl==3.1.5
pip install python-dotenv==1.0.1
pip install requests==2.32.3

# Raspberry Pi only (if using Pi Camera):
# pip install picamera2
```

### Step 4: Configure Azure Face API

1. Go to [Azure Portal](https://portal.azure.com)
2. Create a new **Face** resource
3. Copy **Endpoint** and **Key** from resource
4. Create `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

5. Edit `.env` and set:
```env
USE_AZURE_FACE=1
AZURE_FACE_ENDPOINT=https://YOUR_REGION.api.cognitive.microsoft.com
AZURE_FACE_KEY=your_32_char_hex_key
AZURE_FACE_PERSON_GROUP_ID=students-group
```

### Step 5: Verify Installation

```bash
python scripts/test_installation.py
```

Expected output: All tests pass ✅

### Step 6: Register Students (Hybrid Mode)

```bash
python register_face.py
```

- Captures 15 high-quality samples locally
- Stores average embedding in MongoDB
- **Uploads best sample to Azure** (if USE_AZURE_FACE=1)
- Triggers Azure training after registration

### Step 7: Run Recognition

```bash
python recognize_face.py
```

**Recognition Flow:**
1. **Primary:** Azure Face API identify (≥95% confidence)
2. **Fallback:** Local DeepFace Facenet512 (if Azure unavailable or low confidence)
3. **Display:** Green box + roll number for ≥95% confidence only

### Accuracy Expectations

| Method | Confidence Range | Use Case |
|--------|-----------------|----------|
| **Azure recognition_04** | 95-99% | Primary production use |
| **DeepFace Facenet512** | 85-95% | Fallback / offline mode |
| **Combined Hybrid** | 95-99% | Best reliability + accuracy |

### Troubleshooting

**TensorFlow install fails:**
```bash
# Check Python version
python --version  # Must be 3.11.x

# Try with pip cache clear
pip cache purge
pip install tensorflow==2.15.0 --no-cache-dir
```

**Azure quota exceeded:**
- Free tier: 20 transactions/min, 30K/month
- Standard: Pay-as-you-go, higher limits
- Check quota: Azure Portal > Face resource > Metrics

**Recognition confidence low (<95%):**
- Re-register with better lighting
- Ensure 15 varied angle samples
- Check camera focus and resolution
- Verify Azure training completed

### Cost Estimate (Azure Face API)

- **Free tier:** 30,000 calls/month (sufficient for ~300 recognitions/day)
- **Standard tier:** $1.00 per 1,000 transactions
- **5000 students × 2 recognitions/day = 10K calls/day = ~$300/month standard**
- **Recommendation:** Start with free tier + local fallback for cost optimization

### Next Steps

1. Complete Python 3.11 setup
2. Configure Azure credentials
3. Register 5-10 test students
4. Validate ≥95% confidence on recognition
5. Scale to full student database
6. Monitor Azure usage and costs

---

For questions or issues, see main [README.md](README.md) or [docs/](docs/)
