import os
os.environ['CUDA'] = '1'
from tinygrad import Tensor, Device
print("Device.DEFAULT:", Device.DEFAULT)
x = Tensor.zeros(3).cuda()
x.realize()
print("tinygrad CUDA OK:", x.numpy())
