#!/usr/bin/env bash
# Setup Python 3.11 virtual environment and install all dependencies

set -e

echo "============================================================"
echo "Setting up Python 3.11 environment for Golf Cart Face Recognition"
echo "============================================================"

# Check Python 3.11 available
if ! py -3.11 --version &>/dev/null; then
    echo "❌ Python 3.11 not found"
    echo "Please install Python 3.11.x from python.org"
    exit 1
fi

echo "✅ Found Python 3.11"
py -3.11 --version

# Remove old venv
if [ -d ".venv" ]; then
    echo ""
    echo "Removing old virtual environment..."
    rm -rf .venv
    echo "✅ Old venv removed"
fi

# Create new venv
echo ""
echo "Creating Python 3.11 virtual environment..."
py -3.11 -m venv .venv
echo "✅ Virtual environment created"

# Activate
echo ""
echo "Activating virtual environment..."
source .venv/Scripts/activate

# Upgrade pip
echo ""
echo "Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel -q

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install -q opencv-python==4.9.0.80
pip install -q numpy==1.26.4
pip install -q pymongo==4.6.3
pip install -q pandas==2.0.3
pip install -q openpyxl==3.1.5
pip install -q python-dotenv==1.0.1
pip install -q requests==2.31.0

echo "✅ Core packages installed"

# Ask about TensorFlow/DeepFace
echo ""
echo "============================================================"
echo "Optional: Install TensorFlow + DeepFace for local recognition?"
echo "(Required for hybrid mode; not needed if using Azure-only)"
read -p "Install TensorFlow/DeepFace? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Installing TensorFlow and DeepFace (this may take a few minutes)..."
    pip install tensorflow==2.15.0
    pip install keras==2.15.0
    pip install tf-keras==2.15.1
    pip install deepface==0.0.96
    echo "✅ TensorFlow + DeepFace installed"
else
    echo "⏭️  Skipping TensorFlow/DeepFace (Azure-only mode)"
fi

# Test installation
echo ""
echo "============================================================"
echo "Testing installation..."
echo "============================================================"
python scripts/test_installation.py

echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Virtual environment: .venv (Python 3.11)"
echo ""
echo "Next steps:"
echo "  1. Configure Azure: cp .env.example .env (then edit .env)"
echo "  2. Test Azure: python scripts/test_azure_face.py"
echo "  3. Register: python register_face.py"
echo "  4. Recognize: python recognize_face.py"
echo ""
echo "To activate venv later: source .venv/Scripts/activate"
echo "============================================================"
