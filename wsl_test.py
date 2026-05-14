import torch
print('torch version:', torch.__version__)
m = torch.nn.Conv2d(3,64,3,padding=1).cuda()
m2 = torch.compile(m, mode='max-autotune')
out = m2(torch.randn(4,3,32,32).cuda())
print('torch.compile max-autotune OK, shape', out.shape)
