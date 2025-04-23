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
import math  # 添加数学模块导入
from pytorch_msssim import ssim

# CW L2 attack
def cw_l2_attack(X, model, c=0.1, lr=0.01, iters=100, targeted=False, lambda_ssim=0.1):
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
    
    # 导入SSIM模块（需安装pytorch-msssim）
    from pytorch_msssim import ssim

    for step in pbar:
        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, X)
        loss2 = torch.sum(c*f(a))
        
        # 新增SSIM损失
        loss_ssim = 1 - ssim(a, X, data_range=2.0, nonnegative_ssim=True)
        loss_ssim = loss_ssim * a.shape[0]  # 批量平均
        
        cost = loss1 + loss2 + lambda_ssim * loss_ssim

        pbar.set_description(f"Loss: {cost.item():.5f} | SSIM: {loss_ssim.item():.5f}")
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
    X_adv = 1/2*(nn.Tanh()(w) + 1)
    return X_adv

# Encoder attack - Targeted / Untargeted
def encoder_attack(X, model, eps=0.03, lr=0.01, iters=100, clamp_min=-1, clamp_max=1, targeted=False, lambda_ssim=0.1):
    encoder = model.vae.encode
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda(), min=clamp_min, max=clamp_max)
    X_adv.requires_grad_(True)
    
    optimizer = optim.Adam([X_adv], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters)
    
    clean_latent = encoder(X).latent_dist.mean if not targeted else None

    pbar = tqdm(range(iters))
    for i in pbar:
        latent = encoder(X_adv).latent_dist.mean
        if targeted:
            loss_latent = latent.norm()
        else:
            loss_latent = F.mse_loss(latent, clean_latent)
        
        # 计算SSIM损失
        from pytorch_msssim import ssim
        image = model.vae.decode(latent).sample
        loss_ssim = 1 - ssim(image, X, data_range=2.0).mean()
        
        total_loss = loss_latent + lambda_ssim * loss_ssim

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.data = torch.min(torch.max(X_adv, X - eps), X + eps)
        
        pbar.set_postfix(Latent=loss_latent.item(), SSIM=loss_ssim.item())

    return X_adv.detach()


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

def facelock(X, model, aligner, fr_model, lpips_fn, eps=0.03, step_size=0.01, iters=100,
             clamp_min=-1, clamp_max=1, alpha=0.2, beta=1.0, gamma=1.0, delta_ssim=0.5):
    """
    改进版facelock算法，增强SSIM损失与动态权重调整
    """
    vae = model.vae
    device = X.device
    clean_latent = vae.encode(X).latent_dist.mean

    # 1. 多样化初始化策略（修正数据类型）
    init_methods = [
        lambda: X.clone().detach().float(),
        lambda: torch.clamp(
            X.clone().detach() + (torch.rand_like(X)*2*eps - eps).to(device),
            min=clamp_min, max=clamp_max
        ).float(),
        lambda: torch.clamp(
            X.clone().detach() + eps * torch.sign(torch.randn_like(X)),
            min=clamp_min, max=clamp_max
        ).float()
    ]

    best_score = float('-inf')
    best_X_adv = None
    best_lpips = float('inf')

    for init_idx, init_method in enumerate(init_methods):
        X_adv = init_method()
        X_adv.requires_grad_(True)
        
        # 2. 记忆增强机制
        history_best_X_adv = X_adv.clone()
        stagnation_counter = 0
        best_iter_score = float('-inf')

        # 3. 周期性退火策略
        cycle_size = iters // 3

        pbar = tqdm(range(iters))
        for i in pbar:
            relative_pos = (i % cycle_size) / cycle_size
            current_lr = step_size * (0.5 + 0.5 * math.cos(math.pi * relative_pos))

            # 前向计算（确保float32）
            latent = vae.encode(X_adv).latent_dist.mean
            image = vae.decode(latent).sample.clip(clamp_min, clamp_max).float()  # 强制转为float32

            # 4. 损失计算（确保输入类型正确）
            try:
                loss_cvl = compute_score(
                    image.float(), X.float(), aligner=aligner, fr_model=fr_model
                )
            except Exception:
                loss_cvl = 0.0

            loss_encoder = F.mse_loss(latent, clean_latent)
            loss_lpips = lpips_fn(image, X)

            # SSIM损失计算
            loss_ssim = 1 - ssim(image, X, data_range=2.0).mean()

            # 动态权重调整
            progress = i / iters
            w_cvl = gamma * (1 - progress)
            w_ssim = delta_ssim * min(1.0, progress * 4)
            w_lpips = beta * (progress * 3) if progress > 0.15 else 0.0
            w_encoder = alpha * (1 - 0.5 * progress)

            # 损失平衡器
            losses = [
                loss_cvl.item(), loss_encoder.item(),
                loss_lpips.item(), loss_ssim.item()
            ]
            max_loss = max(losses)
            if max_loss > 0:
                scale_encoder = max_loss / max(1e-8, losses[1])
                scale_lpips = max_loss / max(1e-8, losses[2])
                scale_ssim = max_loss / max(1e-8, losses[3])

                w_encoder = min(alpha * 2, w_encoder * min(1.5, scale_encoder))
                w_lpips = min(beta * 2, w_lpips * min(1.5, scale_lpips))
                w_ssim = min(delta_ssim * 2, w_ssim * min(1.5, scale_ssim))

            # 综合损失
            total_loss = (
                -w_cvl * loss_cvl 
                + w_encoder * loss_encoder
                + w_lpips * loss_lpips
                + w_ssim * loss_ssim
            )

            # 梯度更新
            grad, = torch.autograd.grad(total_loss, [X_adv])

            # 关键区域注意力（确保类型正确）
            if hasattr(aligner, 'get_landmarks') and i % 5 == 0:
                try:
                    with torch.no_grad():
                        attention_map = torch.ones_like(X_adv)
                        landmarks = aligner.get_landmarks(image.float())  # 强制转为float32
                        if landmarks is not None:
                            for lm in landmarks:
                                h, w = image.shape[2:]
                                for pt in lm[:5]:
                                    y, x = int(pt[0] * h), int(pt[1] * w)
                                    if 0 <= y < h and 0 <= x < w:
                                        radius = 5
                                        for dy in range(-radius, radius + 1):
                                            for dx in range(-radius, radius + 1):
                                                if 0 <= y+dy < h and 0 <= x+dx < w:
                                                    attention_map[:, :, y+dy, x+dx] = 1.5
                        grad = grad * attention_map.to(device)
                except Exception:
                    pass

            # 更新X_adv
            X_adv = X_adv + grad.detach().sign() * current_lr
            X_adv = torch.min(
                torch.max(X_adv, X - eps),
                X + eps
            ).clamp(min=clamp_min, max=clamp_max).float()  # 确保float32

            # 记忆机制与早停
            current_score = loss_cvl.item()
            current_lpips = loss_lpips.item()

            if current_score > best_iter_score:
                best_iter_score = current_score
                history_best_X_adv = X_adv.clone()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > 10:
                X_adv = history_best_X_adv + torch.rand_like(X_adv) * eps * 0.1
                X_adv = X_adv.clamp(min=clamp_min, max=clamp_max).float()
                stagnation_counter = 0
                pbar.set_description(f"Restart (Init {init_idx+1}/{len(init_methods)})")

            if i > iters * 0.5 and current_score < 0.3 and current_lpips < 0.1:
                pbar.set_description("Early Stop: 达到保护目标")
                break

            pbar.set_postfix(
                CVL=loss_cvl.item(),
                SSIM=loss_ssim.item(),
                LPIPS=loss_lpips.item(),
                LR=current_lr,
                Stagnation=stagnation_counter
            )

        # 最终评估（强制转为float32）
        with torch.no_grad():
            final_latent = vae.encode(X_adv).latent_dist.mean
            final_image = vae.decode(final_latent).sample.clip(clamp_min, clamp_max).float()
            final_score = compute_score(final_image.float(), X.float(), aligner=aligner, fr_model=fr_model)
            final_lpips = lpips_fn(final_image, X).item()

            combined_score = final_score.item() - 0.5 * final_lpips  # 转换为标量
            if combined_score > best_score or (abs(combined_score - best_score) < 0.05 and final_lpips < best_lpips):
                best_score = combined_score
                best_X_adv = X_adv.clone()
                best_lpips = final_lpips
                print(f"Improved: CVL={final_score.item():.4f}, LPIPS={final_lpips:.4f}, Init {init_idx+1}")

    return best_X_adv if best_X_adv is not None else X_adv
