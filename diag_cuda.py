import subprocess, os

# Check libcuda.so locations
r = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
for line in r.stdout.split('\n'):
    if 'libcuda' in line:
        print("ldconfig:", line.strip())

# Check WSL lib
for f in ['/usr/lib/wsl/lib/libcuda.so.1', '/usr/lib/wsl/lib/libcuda.so']:
    print(f"WSL lib {f}:", os.path.exists(f))

# Check where tinygrad tries to load from
import tinygrad.runtime.ops_cuda as ops
import inspect
src = inspect.getsource(ops)
for line in src.split('\n'):
    if 'libcuda' in line.lower() or 'ctypes' in line.lower() and 'cuda' in line.lower():
        print("tinygrad loading:", line.strip())
