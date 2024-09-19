import torch

# 设置随机种子以确保结果可重复
torch.manual_seed(0)

# 假设的维度大小
b, e, r, i = 2, 3, 4, 5

# 初始化张量，并设置 requires_grad=True 以便计算梯度
route_prob = torch.rand(b, e, requires_grad=True)  # [b, e]
lora_A_stacked = torch.rand(e, r, i, requires_grad=True)  # [e, r, i]

# 打印初始张量
print("route_prob:", route_prob)
print("lora_A_stacked:", lora_A_stacked)

# **第一种计算方式**

# 清零梯度
route_prob.grad = None
lora_A_stacked.grad = None

# 前向计算
result1_2 = (route_prob @ lora_A_stacked.view(e, r * i)).view(b, r, i)

# 定义损失函数（这里我们简单地将结果求和）
loss1 = result1_2.sum()

# 反向传播
loss1.backward()

# 保存梯度
route_prob_grad1 = route_prob.grad.clone()
lora_A_stacked_grad1 = lora_A_stacked.grad.clone()

# **第二种计算方式**

# 清零梯度
route_prob.grad = None
lora_A_stacked.grad = None

# 前向计算
result2 = torch.sum(route_prob.view(b, e, 1, 1) * lora_A_stacked.unsqueeze(0), dim=1)

# 定义损失函数
loss2 = result2.sum()

# 反向传播
loss2.backward()

# 保存梯度
route_prob_grad2 = route_prob.grad.clone()
lora_A_stacked_grad2 = lora_A_stacked.grad.clone()

# **第三种计算方式**

# 清零梯度
route_prob.grad = None
lora_A_stacked.grad = None

# 前向计算
result3 = torch.einsum('be,eri->bri', route_prob, lora_A_stacked)

# 定义损失函数
loss3 = result3.sum()

# 反向传播
loss3.backward()

# 保存梯度
route_prob_grad3 = route_prob.grad.clone()
lora_A_stacked_grad3 = lora_A_stacked.grad.clone()

# **比较梯度**

# 比较 route_prob 的梯度
print("route_prob 梯度是否相等（方法1和方法2）:", torch.allclose(route_prob_grad1, route_prob_grad2))
print("route_prob 梯度是否相等（方法1和方法3）:", torch.allclose(route_prob_grad1, route_prob_grad3))

# 比较 lora_A_stacked 的梯度
print("lora_A_stacked 梯度是否相等（方法1和方法2）:", torch.allclose(lora_A_stacked_grad1, lora_A_stacked_grad2))
print("lora_A_stacked 梯度是否相等（方法1和方法3）:", torch.allclose(lora_A_stacked_grad1, lora_A_stacked_grad3))
