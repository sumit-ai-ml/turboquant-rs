#!/bin/bash
# Setup script for TurboQuant RS benchmark
set -e

echo "Installing jsonnet-binary first (avoids C++ build failure)..."
pip install jsonnet-binary

echo ""
echo "Installing remaining dependencies..."
pip install -r requirements.txt

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
print('All core imports OK')
"

echo ""
echo "Ready. Run: python sanity_check.py"
