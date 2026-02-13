import torch

# > Using cpu device

import torch

print("CUDA available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

def test():
    print("hello")
  
