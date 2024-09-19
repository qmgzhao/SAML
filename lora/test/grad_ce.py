import torch
import torch.nn.functional as F

# 设置随机种子以确保结果可重复
torch.manual_seed(0)

# 假设的维度大小
b, e, num_classes = 2, 3, 5  # b: batch size, e: number of experts, num_classes: number of classes

# 初始化张量，并设置 requires_grad=True 以便计算梯度
route_prob = torch.rand(b, e, requires_grad=True)  # [b, e]
lora_A_stacked = torch.rand(e, num_classes, requires_grad=True)  # [e, num_classes]

# 打印初始张量
print("route_prob:", route_prob)
print("lora_A_stacked:", lora_A_stacked)

# **第一种计算方式**

# 清零梯度
route_prob.grad = None
lora_A_stacked.grad = None

# 前向计算
result1 = route_prob @ lora_A_stacked  # [b, e] @ [e, num_classes] -> [b, num_classes]

# 定义目标标签，随机生成
labels = torch.randint(0, num_classes, (b,), dtype=torch.long)  # [b]

# 定义损失函数（交叉熵损失）
loss1 = F.cross_entropy(result1, labels)

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
result2 = torch.sum(route_prob.view(b, e, 1) * lora_A_stacked.unsqueeze(0), dim=1)  # [b, num_classes]

# 定义损失函数
loss2 = F.cross_entropy(result2, labels)

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
result3 = torch.einsum('be,ec->bc', route_prob, lora_A_stacked)  # [b, num_classes]

# 定义损失函数
loss3 = F.cross_entropy(result3, labels)

# 反向传播
loss3.backward()

# 保存梯度
route_prob_grad3 = route_prob.grad.clone()
lora_A_stacked_grad3 = lora_A_stacked.grad.clone()

# **比较梯度**

# 比较 route_prob 的梯度
print("route_prob 梯度是否相等（方法1和方法2）:", torch.allclose(route_prob_grad1, route_prob_grad2))
print("route_prob 梯度是否相等（方法1和方法3）:", torch.allclose(route_prob_grad1, route_prob_grad3))
print(f"route_prob_grad1: {route_prob_grad1}")
print(f"route_prob_grad2: {route_prob_grad2}")
print(f"route_prob_grad3: {route_prob_grad3}")


# 比较 lora_A_stacked 的梯度
print("lora_A_stacked 梯度是否相等（方法1和方法2）:", torch.allclose(lora_A_stacked_grad1, lora_A_stacked_grad2))
print("lora_A_stacked 梯度是否相等（方法1和方法3）:", torch.allclose(lora_A_stacked_grad1, lora_A_stacked_grad3))
print(f"lora_A_stacked_grad1: {lora_A_stacked_grad1}")
print(f"lora_A_stacked_grad2: {lora_A_stacked_grad2}")
print(f"lora_A_stacked_grad3: {lora_A_stacked_grad3}")