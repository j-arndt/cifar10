import torch
import triton
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "triton:", triton.__version__)

# Quick compile test
m = torch.nn.Conv2d(3,64,3,padding=1).cuda()
m2 = torch.compile(m, mode='max-autotune')
out = m2(torch.randn(4,3,32,32).cuda())
print("torch.compile max-autotune: OK shape", out.shape)
print("gcc available: OK")
