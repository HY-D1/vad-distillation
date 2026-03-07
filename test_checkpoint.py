import torch
import os

# Check one checkpoint
checkpoint_path = 'outputs/production_cuda/checkpoints/fold_F01_latest_best.pt'
print(f'Checking: {checkpoint_path}')
print(f'Exists: {os.path.exists(checkpoint_path)}')

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print('SUCCESS: Checkpoint loaded!')
    print(f'Keys: {list(checkpoint.keys())}')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
