mkdir temp_dir
cd temp_dir
python -m venv test_env
source test_env/bin/activate
module load cuda
pip install torch
pip install --upgrade setuptools

cat > test_gpu.py << EOF

import torch
import os
for i in range(4):
        os.environ["CUDA_DEVICE_ORDER"]=f"{i}"
        assert torch.cuda.is_available() == True, f"Cuda is NOT available for GPU number {i}"
print("Complete: all GPUs are working")

EOF

python test_gpu.py


