import tinygrad
print("tinygrad version:", tinygrad.__version__)
from tinygrad import Tensor, TinyJit, Device, GlobalCounters
from tinygrad.helpers import getenv, BEAM
print("Device:", Device.DEFAULT)
print("CUDA available:", Device.DEFAULT == "CUDA" or "CUDA" in Device._devices if hasattr(Device, '_devices') else "unknown")
# Check if extra is available
try:
    from extra.lr_scheduler import OneCycleLR
    print("extra.lr_scheduler: OK")
except ImportError as e:
    print("extra.lr_scheduler: MISSING -", e)
try:
    from extra.bench_log import BenchEvent
    print("extra.bench_log: OK")
except ImportError as e:
    print("extra.bench_log: MISSING -", e)
from tinygrad.nn import optim
from tinygrad.nn.state import get_state_dict
print("nn.optim + state: OK")
from tinygrad import dtypes
from tinygrad.helpers import Context, WINO, colored, prod
print("helpers: OK")
