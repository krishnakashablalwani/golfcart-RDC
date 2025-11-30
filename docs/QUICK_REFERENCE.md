# Golf Cart Face Recognition - Quick Reference

## 🚀 Daily Operations

### Starting the System

```bash
# 1. Activate virtual environment
source .venv/Scripts/activate  # Windows
source .venv/bin/activate      # Linux

# 2. Start recognition
python recognize_face.py
```

### Keyboard Controls
- **Q** - Quit recognition
- **R** - Reload database (after new registrations)

---

## 👤 Register New Student

### Single Student
```bash
python register_face.py
# Choose option 1
# Enter roll number: 2451-25-733-075
# System captures 15 samples automatically
```

### Multiple Students
```bash
python register_face.py
# Choose option 2
# Enter roll numbers (one per line)
# Press Enter on empty line to start
```

**Important:** 
- Student must be in Excel file first
- Takes ~2-3 minutes per student
- Look at camera, keep face centered
- Slight angle variations are good

---

## 📊 Check Status

```bash
python manage_database.py
# Option 4 - View statistics
```

Shows:
- Total students
- Students with face samples
- Recent detections
- Students by department

---

## 🔍 Find Student

```bash
python manage_database.py
# Option 5 - Search student
# Enter name or roll number
```

---

## ⚠️ Troubleshooting

### Camera Not Working
```bash
# Check camera index in .env
CAMERA_INDEX=0  # Try 0, 1, 2...
```

### System Slow
- Close other programs
- Restart recognition system
- Check disk space

### Student Not Found
- Verify roll number format: `2451-25-733-075`
- Check Excel file has student
- Try database search first

### Recognition Not Working
- Press 'R' to reload database
- Check lighting conditions
- Verify camera is not blocked

---

## 📁 File Locations

| Data | Location |
|------|----------|
| Face Samples | `Samples/Year/Dept/RollNumber/` |
| Detections | `Detections/` |
| Student Excel | `Student information.xlsx` |
| Departments | `departments.txt` |
| Config | `.env` |

---

## 🔄 Common Commands

### Start MongoDB (if stopped)
```bash
docker start mongo
```

### Check MongoDB Status
```bash
docker ps
```

### View Recent Detections
```bash
python manage_database.py
# Option 4 - Statistics
# Scroll to "Recent Detections"
```

### List Unregistered Students
```bash
python manage_database.py
# Option 6 - List unregistered
```

---

## 📞 Emergency Contacts

System Admin: [Add contact]
IT Support: [Add contact]
HOD: [Add contact]

---

## ⚡ Quick Tips

1. **Register students during class hours** - easier to find them
2. **Good lighting is critical** - natural light is best
3. **Keep camera lens clean** - check daily
4. **Monitor disk space** - each detection saves an image
5. **Backup weekly** - use database export feature

---

## 🎯 Roll Number Format

**Format:** `COLLEGE-YY-DEPT-ROLL`

**Example:** `2451-25-733-075`
- 2451 = College code
- 25 = Year (2025)
- 733 = Dept (CSE)
- 075 = Class roll

**Must match exactly!**

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Registration Time | 2-3 min/student |
| Recognition Speed | 10-15 FPS |
| Detection Distance | 3-5 meters |
| Students in DB | Up to 5000 |
| Samples per Student | 15 |

---

## 🔐 Access Levels

| User | Can Do |
|------|--------|
| Operator | Run recognition, register students |
| Admin | All + database management |
| Super Admin | All + system configuration |

---

**Keep this card handy for daily operations!**
