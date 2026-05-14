import os
os.environ['CUDA'] = '1'
from tinygrad import Tensor, Device
print("Device.DEFAULT:", Device.DEFAULT)
x = Tensor.zeros(3, device="CUDA")
x.realize()
print("tinygrad CUDA tensor OK:", x.numpy())
