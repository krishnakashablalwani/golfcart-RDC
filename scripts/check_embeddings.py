import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.database import db
import numpy as np

embs = list(db.face_embeddings.find({}))
print(f'Total registered: {len(embs)}')

for e in embs:
    print(f'\n{e["roll_number"]}:')
    print(f'  Samples: {e.get("num_samples", 0)}')
    emb_list = e.get("embedding", [])
    print(f'  Embedding length: {len(emb_list)}')
    print(f'  Embedding type: {type(emb_list)}')
    
    if len(emb_list) > 0:
        emb_arr = np.array(emb_list)
        print(f'  Mean: {emb_arr.mean():.4f}')
        print(f'  Std: {emb_arr.std():.4f}')
        print(f'  Min: {emb_arr.min():.4f}')
        print(f'  Max: {emb_arr.max():.4f}')
        print(f'  First 5 values: {emb_arr[:5]}')
