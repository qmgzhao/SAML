import torch

# 假设的维度大小
b, e, r, i = 2, 3, 4, 5

# 初始化张量
route_prob = torch.rand(b, e)  # [b, e]
lora_A_stacked = torch.rand(e, r, i)  # [e, r, i]
print(f"route_prob: {route_prob}")
print(f"lora_A_stacked: {lora_A_stacked}")

result1 = (route_prob @ lora_A_stacked.view(r, e, i)).view(b, r, i)
print(f"result1: {result1}")

result1_1 = (lora_A_stacked.view(r, i, e) @ route_prob.view(e, b)).view(b, r, i)
print(f"result1_1: {result1_1}")

result1_2 = (route_prob @ lora_A_stacked.view(e, r * i)).view(b, r, i)
print(f"result1_2: {result1_2}")



result2 = torch.sum(route_prob.view(b, e, 1, 1) * lora_A_stacked.unsqueeze(0), dim=1)
print(f"result2: {result2}")

result3 = torch.einsum('be,eri->bri', route_prob, lora_A_stacked)
print(f"result3: {result3}")