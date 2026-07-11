import torch

#----#
def rot_by_head(q, k, v, num_rot):
    # q, k, v: (B, num_heads, N, head_dim)
    # print("q.shape:",q.shape)
    B, num_heads, N, head_dim = q.shape # (128, 6, 256, 64)
    H = W = int(N ** 0.5)
    
    # 1. 合并处理 q, k, v -> (3, B, num_heads, N, head_dim)
    combined = torch.stack([q, k, v])
    
    # 2. 变形为 7 维以进行批量旋转
    # 维度索引: 0:3, 1:B, 2:num_rot, 3:heads_per_group, 4:H, 5:W, 6:head_dim
    heads_per_group = num_heads // num_rot
    combined = combined.view(3, B, num_rot, heads_per_group, H, W, head_dim)

    c = 4 // num_rot
    results = []

    for i in range(num_rot):
        if i == 0:
            results.append(combined[:, :, i]) # (3, B, heads_per_group, H, W, head_dim)
        else:
            k_rot = c * i
            # k_rot=i
            # 修正点：在 6 维张量中，H 和 W 分别是 index 3 和 4
            # 原代码 dims=(3, 2) 对应 H, W 旋转，这里对应 (4, 3)
            rotated = torch.rot90(combined[:, :, i], k=k_rot, dims=(5, 4)) # 这里的 5, 4 是相对于 (3, B, heads_per_group, H, W, head_dim) 的 H, W
            # 纠正：对于 combined[:, :, i] 来说，shape 是 (3, B, heads_per_group, H, W, head_dim)
            # H 是 dim 3, W 是 dim 4。所以 dims 应为 (4, 3)
            rotated = torch.rot90(combined[:, :, i], k=k_rot, dims=(4, 3))
            results.append(rotated)
            
    # 3. 合并回原始形状
    # cat 之后 shape: (3, B, num_heads, H, W, head_dim)
    out = torch.cat(results, dim=2) 
    out = out.reshape(3, B, num_heads, N, head_dim) # 使用 reshape 比 view 更稳健
    
    return out[0], out[1], out[2]

def recover(x_rot, num_rot):
    # x_rot: (B, num_heads, N, head_dim)
    B, num_heads, N, head_dim = x_rot.shape
    H = W = int(N ** 0.5)

    heads_per_group = num_heads // num_rot
    # 维度索引: 0:B, 1:num_rot, 2:heads_per_group, 3:H, 4:W, 5:head_dim
    x_grouped = x_rot.view(B, num_rot, heads_per_group, H, W, head_dim)

    c = 4 // num_rot
    results = []

    for i in range(num_rot):
        if i == 0:
            results.append(x_grouped[:, i])
        else:
            k_rot = - c * i
            # 在 5 维张量 (B, heads_per_group, H, W, head_dim) 中，H 是 2, W 是 3
            rotated = torch.rot90(x_grouped[:, i], k=k_rot, dims=(3, 2))
            results.append(rotated)
            
    # 合并并还原
    out = torch.cat(results, dim=1)
    return out.reshape(B, num_heads, N, head_dim)
#----#