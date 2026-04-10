#!/bin/bash
# Setup script for TurboQuant RS benchmark
set -e

echo "Installing core dependencies..."
pip install -r requirements.txt

echo ""
echo "Installing torchgeo (without lightning extras to avoid jsonnet build failure)..."
pip install torchgeo --no-deps

echo ""
echo "Verifying key imports..."
python -c "
import numpy, scipy, matplotlib, torch, faiss, tqdm
print(f'numpy {numpy.__version__}')
print(f'scipy {scipy.__version__}')
print(f'torch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
try:
    import torchgeo
    print(f'torchgeo {torchgeo.__version__}')
except Exception as e:
    print(f'torchgeo import warning (may still work for datasets): {e}')
print('All core imports OK')
"

echo ""
echo "Ready. Run: python sanity_check.py"
