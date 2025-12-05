# Temporary script to update InsightFace initialization in both files

import re

def fix_insightface_init(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the InsightFace initialization block
    old_pattern = r'''# Load InsightFace with error handling
import insightface
_insightface = None
_retinaface = None
providers = \["CPUExecutionProvider"\]

try:
    if USE_GPU in \("1", "true", "True"\):
        providers = \["CUDAExecutionProvider", "CPUExecutionProvider"\]
    
    # Get model \(will download if not present\) - use buffalo_l which is available
    _insightface = insightface\.model_zoo\.get_model\("buffalo_l"\)
    if _insightface:
        _insightface\.prepare\(ctx_id=0 if providers\[0\] == "CUDAExecutionProvider" else -1\)
        print\(f"✓ InsightFace loaded: buffalo_l model with \{providers\[0\]\}"\)
except Exception as e:
    print\(f"⚠ InsightFace initialization error: \{e\}"\)
    print\("Attempting fallback initialization\.\.\."\)
    _insightface = None'''
    
    new_content = '''# Load InsightFace ArcFace model directly (lighter memory footprint)
import insightface
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
import os.path as osp
_insightface = None
_retinaface = None
providers = ["CPUExecutionProvider"]

try:
    if USE_GPU in ("1", "true", "True"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    # Use standalone ArcFace model from buffalo_sc (MobileFaceNet - lighter)
    model_path = osp.join(osp.expanduser('~'), '.insightface', 'models', 'buffalo_sc', 'w600k_mbf.onnx')
    if osp.exists(model_path):
        _insightface = ArcFaceONNX(model_file=model_path)
        _insightface.prepare(ctx_id=0 if providers[0] == "CUDAExecutionProvider" else -1)
        print(f"✓ InsightFace loaded: w600k_mbf (MobileFaceNet) with {providers[0]}")
    else:
        print(f"⚠ Model file not found at {model_path}")
        print("Please ensure buffalo_sc model is downloaded")
        _insightface = None
except Exception as e:
    print(f"⚠ InsightFace initialization error: {e}")
    print("Attempting fallback initialization...")
    _insightface = None'''
    
    updated_content = re.sub(old_pattern, new_content, content, flags=re.MULTILINE | re.DOTALL)
    
    if updated_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"✓ Updated {filepath}")
        return True
    else:
        print(f"✗ Pattern not found in {filepath}")
        return False

# Fix both files
fix_insightface_init('register_face_deepface.py')
fix_insightface_init('recognize_face_deepface.py')
print("\n✓ All files updated!")
