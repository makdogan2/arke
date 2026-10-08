import torch

print("torch:", torch.__version__)
print("cuda var mı:", torch.cuda.is_available())
print("kart:", torch.cuda.get_device_name(0))
print("mimari:", torch.cuda.get_device_capability(0))

x = torch.randn(8192, 8192, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("çarpım bitti:", y.mean().item())