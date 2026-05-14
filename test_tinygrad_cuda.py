import os
os.environ['CUDA'] = '1'
from tinygrad import Tensor, Device
print("tinygrad Device.DEFAULT:", Device.DEFAULT)
try:
    x = Tensor([1.0, 2.0], device="CUDA")
    print("CUDA tensor OK:", x.numpy())
except Exception as e:
    print("CUDA failed:", e)

# Try without setting CUDA=1
from tinygrad.helpers import getenv
print("BEAM:", getenv("BEAM", 0))

# Check what devices are available
from tinygrad import Device as D
print("Available:", [d for d in ["CUDA", "GPU", "CPU", "CLANG", "LLVM"] if True])
