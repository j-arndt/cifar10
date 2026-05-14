import subprocess
# Find out what libcuda.so tinygrad is about to load
import ctypes.util
cuda_lib = ctypes.util.find_library('cuda')
print("ctypes finds libcuda at:", cuda_lib)

# Check the actual file
import subprocess
r = subprocess.run(['ldd', '/lib/x86_64-linux-gnu/libcuda.so'], capture_output=True, text=True)
print("ldd libcuda.so:", r.stdout[:500])

# Check the wsl version
r2 = subprocess.run(['ldd', '/usr/lib/wsl/lib/libcuda.so.1'], capture_output=True, text=True)
print("ldd wsl libcuda.so.1:", r2.stdout[:200])

# try patching ctypes to load the WSL version
import ctypes
try:
    cuda_wsl = ctypes.CDLL('/usr/lib/wsl/lib/libcuda.so.1')
    result = ctypes.c_int(-1)
    cuda_wsl.cuInit(0)
    print("cuInit with WSL lib: SUCCESS")
except Exception as e:
    print("cuInit with WSL lib:", e)
