import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from utils import compute_score
import pdb

# CW L2 attack
def cw_l2_attack(X, model, c=0.1, lr=0.01, iters=100, targeted=False):
    encoder = model.vae.encode
    clean_latents = encoder(X).latent_dist.mean

    def f(x):
        latents = encoder(x).latent_dist.mean
        if targeted:
            return latents.norm()
        else:
            return -torch.norm(latents - clean_latents.detach(), p=2, dim=-1)
    
    w = torch.zeros_like(X, requires_grad=True).cuda()
    pbar = tqdm(range(iters))
    optimizer = optim.Adam([w], lr=lr)

    for step in pbar:
        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, X)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2
        pbar.set_description(f"Loss: {cost.item():.5f} | loss1: {loss1.item():.5f} | loss2: {loss2.item():.5f}")
        # pdb.set_trace()

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
    X_adv = 1/2*(nn.Tanh()(w) + 1)
    return X_adv

# Encoder attack - Targeted / Untargeted
def encoder_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1, targeted=False):
    """
    Processing encoder attack using l_inf norm
    Params:
        X - image tensor we hope to protect
        model - the targeted edit model
        eps - attack budget
        step_size - attack step size
        iters - attack iterations
        clamp_min - min value for the image pixels
        clamp_max - max value for the image pixels
    Return:
        X_adv - image tensor for the protected image
    """
    encoder = model.vae.encode
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).half().cuda(), min=clamp_min, max=clamp_max)
    if not targeted:
        loss_fn = nn.MSELoss()
        clean_latent = encoder(X).latent_dist.mean
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_(True)
        latent = encoder(X_adv).latent_dist.mean
        if targeted:
            loss = latent.norm()
            grad, = torch.autograd.grad(loss, [X_adv])
            X_adv = X_adv - grad.detach().sign() * actual_step_size
        else:
            loss = loss_fn(latent, clean_latent)
            grad, = torch.autograd.grad(loss, [X_adv])
            X_adv = X_adv + grad.detach().sign() * actual_step_size

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        pbar.set_postfix(norm_2=(X_adv - X).norm().item(), norm_inf=(X_adv - X).abs().max().item())

    return X_adv

def vae_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1):
    """
    Processing encoder attack using l_inf norm
    Params:
        X - image tensor we hope to protect
        model - the targeted edit model
        eps - attack budget
        step_size - attack step size
        iters - attack iterations
        clamp_min - min value for the image pixels
        clamp_max - max value for the image pixels
    Return:
        X_adv - image tensor for the protected image
    """
    vae = model.vae
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).half().cuda(), min=clamp_min, max=clamp_max)
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_()
        image = vae(X_adv).sample

        loss = (image).norm()
        grad, = torch.autograd.grad(loss, [X_adv])
        X_adv = X_adv - grad.detach().sign() * actual_step_size

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

    return X_adv

# def facelock(X, model, aligner, fr_model, lpips_fn, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1):
#     X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).to(X.device), min=clamp_min, max=clamp_max).half()
#     pbar = tqdm(range(iters))
    
#     vae = model.vae
#     X_adv.requires_grad_(True)
#     clean_latent = vae.encode(X).latent_dist.mean

#     for i in pbar:
#         # actual_step_size = step_size
#         actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        
#         latent = vae.encode(X_adv).latent_dist.mean
#         image = vae.decode(latent).sample.clip(-1, 1)

#         loss_cvl = compute_score(image.float(), X.float(), aligner=aligner, fr_model=fr_model)
#         loss_encoder = F.mse_loss(latent, clean_latent)
#         loss_lpips = lpips_fn(image, X)
#         loss = -loss_cvl * (1 if i >= iters * 0.35 else 0.0) + loss_encoder * 0.2 + loss_lpips * (1 if i > iters * 0.25 else 0.0)
#         grad, = torch.autograd.grad(loss, [X_adv])
#         X_adv = X_adv + grad.detach().sign() * actual_step_size

#         X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
#         X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
#         X_adv.grad = None

#         pbar.set_postfix(loss_cvl=loss_cvl.item(), loss_encoder=loss_encoder.item(), loss_lpips=loss_lpips.item(), loss=loss.item())

#     return X_adv

# ### facelock 算法改进
def facelock(X, model, aligner, fr_model, lpips_fn, eps=0.03, step_size=0.01, iters=100, 
                      clamp_min=-1, clamp_max=1, alpha=0.2, beta=1.0, gamma=1.0):
    """
    改进版facelock算法，引入多种优化技术提高性能
    
    参数:
        X - 需要保护的图像张量
        model - 目标编辑模型
        aligner - 人脸对齐模型
        fr_model - 人脸识别模型
        lpips_fn - LPIPS感知相似度函数
        eps - 攻击预算
        step_size - 攻击步长基准值
        iters - 攻击迭代次数
        clamp_min/clamp_max - 图像像素值范围
        alpha - 编码器损失基础权重
        beta - LPIPS损失基础权重
        gamma - CVL损失基础权重
    
    返回:
        X_adv - 受保护的图像张量
    """
    # 初始化设置
    vae = model.vae
    device = X.device
    
    # 1. 多样化初始化策略
    # 使用三种不同初始化方法并行处理，最后选择最佳结果
    init_methods = [
        lambda: X.clone().detach(),  # 原始图像初始化
        lambda: torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).to(device), 
                           min=clamp_min, max=clamp_max).half(),  # 随机噪声初始化
        lambda: torch.clamp(X.clone().detach() + eps * torch.sign(torch.randn_like(X)), 
                           min=clamp_min, max=clamp_max).half()  # 有符号高斯噪声初始化
    ]
    
    clean_latent = vae.encode(X).latent_dist.mean
    
    # 获取初始CVL分数作为基准
    with torch.no_grad():
        initial_image = vae.decode(clean_latent).sample.clip(clamp_min, clamp_max)
        initial_cvl = compute_score(initial_image.float(), X.float(), aligner=aligner, fr_model=fr_model)
    
    best_score = float('-inf')
    best_X_adv = None
    best_lpips = float('inf')
    
    # 每种初始化方法尝试一次
    for init_idx, init_method in enumerate(init_methods):
        X_adv = init_method()
        X_adv.requires_grad_(True)
        
        # 2. 使用记忆增强机制
        # 记录优化过程中的最佳结果和历史轨迹
        history_best_X_adv = X_adv.clone()
        stagnation_counter = 0
        last_loss_cvl = initial_cvl
        best_iter_score = float('-inf')
        
        # 3. 实现周期性退火策略
        # 使用余弦退火调度器动态调整学习率
        import math
        cycle_size = iters // 3  # 定义周期大小
        
        pbar = tqdm(range(iters))
        for i in pbar:
            # 周期性学习率调整
            relative_position = (i % cycle_size) / cycle_size
            current_lr = step_size * (0.5 + 0.5 * math.cos(math.pi * relative_position))
            
            latent = vae.encode(X_adv).latent_dist.mean
            image = vae.decode(latent).sample.clip(clamp_min, clamp_max)
            
            # 4. 计算各类损失
            loss_cvl = compute_score(image.float(), X.float(), aligner=aligner, fr_model=fr_model)
            loss_encoder = F.mse_loss(latent, clean_latent)
            loss_lpips = lpips_fn(image, X)
            
            # 5. 自适应权重调整策略
            # 根据当前损失值状态动态调整各损失权重
            progress = i / iters  # 当前进度比例
            
            # CVL权重自适应调整：当CVL不再下降时增加权重
            cvl_change_rate = (loss_cvl / last_loss_cvl) if last_loss_cvl > 0 else 1.0
            w_cvl = gamma * min(2.0, max(0.5, 1.5 - cvl_change_rate)) if progress > 0.2 else 0.0
            last_loss_cvl = loss_cvl.item()
            
            # 其他损失权重随进度调整
            w_lpips = beta * min(1.0, progress * 3) if progress > 0.15 else 0.0
            w_encoder = alpha * (1.0 - 0.5 * progress)  # 随着迭代进行逐渐减小编码器权重
            
            # 6. 损失函数平衡器
            # 确保各损失项数值在相近范围内，防止某一损失主导优化过程
            losses = [loss_cvl.item(), loss_encoder.item(), loss_lpips.item()]
            if i > 0 and i % 10 == 0:  # 每10次迭代重新平衡一次权重
                max_loss = max(losses)
                if max_loss > 0:
                    scale_encoder = max_loss / max(1e-8, losses[1])
                    scale_lpips = max_loss / max(1e-8, losses[2])
                    # 应用缩放，但限制在合理范围内
                    w_encoder = min(alpha * 2, w_encoder * min(1.5, scale_encoder))
                    w_lpips = min(beta * 2, w_lpips * min(1.5, scale_lpips))
            
            # 综合损失计算
            loss = -w_cvl * loss_cvl + w_encoder * loss_encoder + w_lpips * loss_lpips
            
            # 计算梯度并更新
            grad, = torch.autograd.grad(loss, [X_adv])
            
            # 7. 人脸关键区域重点保护（简化版）
            # 如果有人脸关键点信息，可以对关键区域施加差异化扰动
            if hasattr(aligner, 'get_landmarks') and i % 5 == 0:  # 每5次迭代更新一次
                try:
                    with torch.no_grad():
                        # 创建简单的注意力图，对眼睛鼻子区域加权
                        attention_map = torch.ones_like(X_adv)
                        landmarks = aligner.get_landmarks(image.float())
                        if landmarks is not None:
                            for lm in landmarks:
                                # 这里简化处理，实际应用需要更精确的关键点映射
                                h, w = image.shape[2:]
                                for point in lm[:5]:  # 假设前5个点是眼睛和鼻子区域
                                    y, x = int(point[0] * h), int(point[1] * w)
                                    if 0 <= y < h and 0 <= x < w:
                                        # 在关键点周围区域增加权重
                                        radius = 5
                                        for dy in range(-radius, radius+1):
                                            for dx in range(-radius, radius+1):
                                                if 0 <= y+dy < h and 0 <= x+dx < w:
                                                    attention_map[:, :, y+dy, x+dx] = 1.5
                        # 应用注意力图
                        grad = grad * attention_map
                except:
                    pass  # 如果获取关键点失败，就使用原始梯度
            
            # 使用当前学习率更新
            X_adv = X_adv + grad.detach().sign() * current_lr
            
            # 应用扰动约束
            X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
            X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
            X_adv.grad = None
            
            # 8. 记忆增强机制
            # 记录最佳分数和检测停滞
            current_score = loss_cvl.item()
            current_lpips = loss_lpips.item()
            
            # 评估当前结果是否为最佳
            if current_score > best_iter_score:
                best_iter_score = current_score
                history_best_X_adv = X_adv.clone()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # 如果连续10次迭代无改进，则进行恢复和扰动
            if stagnation_counter > 10:
                # 恢复到历史最佳，并添加小扰动重新开始探索
                X_adv = history_best_X_adv.clone() + (torch.rand_like(X_adv) * eps * 0.1).to(device)
                X_adv = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
                stagnation_counter = 0
                pbar.set_description(f"重启优化 (Init {init_idx+1}/{len(init_methods)})")
            
            # 9. 早停机制
            # 如果已经达到足够好的保护效果且视觉质量良好，提前结束
            if i > iters * 0.5 and current_score < 0.3 and current_lpips < 0.1:
                pbar.set_description(f"提前停止: 已达到保护目标 (Init {init_idx+1}/{len(init_methods)})")
                break
                
            pbar.set_postfix(
                loss_cvl=loss_cvl.item(), 
                loss_encoder=loss_encoder.item(), 
                loss_lpips=loss_lpips.item(),
                lr=current_lr,
                stagnation=stagnation_counter
            )
        
        # 10. 评估当前初始化方法结果
        # 根据保护效果和视觉质量综合评分
        with torch.no_grad():
            final_image = vae.decode(vae.encode(X_adv).latent_dist.mean).sample.clip(clamp_min, clamp_max)
            final_score = compute_score(final_image.float(), X.float(), aligner=aligner, fr_model=fr_model)
            final_lpips = lpips_fn(final_image, X).item()

            final_score_value = final_score.item() if isinstance(final_score, torch.Tensor) else final_score
            final_lpips_value = final_lpips.item() if isinstance(final_lpips, torch.Tensor) else final_lpips
            
            # 综合考虑保护效果和视觉质量
            combined_score = final_score_value - 0.5 * final_lpips_value
            
            if combined_score > best_score or (abs(combined_score - best_score) < 0.05 and final_lpips < best_lpips):
                best_score = combined_score
                best_X_adv = X_adv.clone()
                best_lpips = final_lpips
                print(f"找到更好的结果: CVL={final_score.item():.4f}, LPIPS={final_lpips:.4f}, 初始化方法 {init_idx+1}")
    
    # 如果所有方法都失败，使用最后一个结果
    if best_X_adv is None:
        best_X_adv = X_adv
        
    return best_X_adv
