# Golf Cart Face Recognition System

High-resolution face recognition system for monitoring 5,000+ students using DeepFace and MongoDB.

## 📁 Project Structure

```
Golf cart/
├── register_face_deepface.py    # Main: Register students
├── recognize_face_deepface.py   # Main: Real-time recognition
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment config template
│
├── modules/                     # Core application modules
│   ├── database.py             # MongoDB interface
│   ├── excel_parser.py         # Excel student data parser
│   └── config.py               # Configuration settings
│
├── scripts/                     # Utility scripts
│   ├── clear_database.py       # Clear all face data
│   ├── manage_database.py      # Database management CLI
│   ├── check_embeddings.py     # Verify stored embeddings
│   └── test_installation.py    # System verification
│
├── data/                        # Data files
│   ├── Student information.xlsx # Student records
│   └── departments.txt         # Department mappings
│
├── docs/                        # Documentation
│   ├── README.md               # Comprehensive guide
│   ├── START.md                # Quick start guide
│   ├── QUICK_REFERENCE.md      # Command reference
│   └── ...                     # More docs
│
├── Samples/                     # Face samples (created at runtime)
│   └── Year/Dept/RollNumber/
│
└── Detections/                  # Detection logs (created at runtime)
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source .venv/Scripts/activate  # Git Bash on Windows
# or
.venv\Scripts\activate         # CMD on Windows

# Verify installation
python scripts/test_installation.py
```

### 2. Import Students

```bash
# Import from Excel
python scripts/manage_database.py
# Choose option 1: Import all students from Excel
```

### 3. Register Students

```bash
# Start registration
python register_face_deepface.py

# Instructions:
# - Press 'C' to capture (15 times per student)
# - Vary angles slightly between captures
# - Press 'Q' to skip to next student
```

### 4. Run Recognition

```bash
# Start real-time recognition
python recognize_face_deepface.py

# Controls:
# - 'Q' to quit
# - 'R' to reload cache
```

## 📚 Documentation

- **[START.md](docs/START.md)** - Quick start guide
- **[README.md](docs/README.md)** - Full documentation
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Command reference
- **[INSTALLATION_SUCCESS.md](docs/INSTALLATION_SUCCESS.md)** - Setup verification

## 🛠️ Key Features

- **High Resolution**: 1920x1080 camera capture
- **Scalable**: Optimized for 5,000+ students
- **Accurate**: DeepFace with Facenet512 model
- **Persistent**: MongoDB storage with caching
- **Organized**: Samples stored by Year/Department/Roll Number

## 📋 Common Tasks

### Clear Database
```bash
python scripts/clear_database.py
```

### Check Embeddings
```bash
python scripts/check_embeddings.py
```

### Database Management
```bash
python scripts/manage_database.py
```

### Search Students
```bash
python scripts/manage_database.py
# Choose option 5: Search student
```

## 🔧 Configuration

Edit `modules/config.py` for:
- Camera settings
- Recognition thresholds
- MongoDB connection
- File paths

## 📝 Requirements

- Python 3.13+
- MongoDB (localhost:27017)
- Webcam
- See `requirements.txt` for packages

## 🐛 Troubleshooting

### Camera Issues
- Try different camera index (0, 1, 2)
- Check camera permissions
- See [SETUP_PC_LINUX.md](docs/SETUP_PC_LINUX.md)

### Recognition Not Working
- Ensure 15 samples per student
- Check distance threshold
- Reload cache with 'R' key

### Import Errors
- Verify virtual environment is activated
- Check `modules/` folder structure
- Ensure MongoDB is running

## 📄 License

MIT License - See LICENSE file

## 👥 Authors

Golf Cart Face Recognition Team

---

For detailed documentation, see [docs/README.md](docs/README.md)
