import torch
print("CUDA:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
