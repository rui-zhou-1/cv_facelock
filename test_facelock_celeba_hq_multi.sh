#!/bin/bash

# 脚本：test_facelock_dataset_main.sh
# 用途：使用 data 文件夹中的 CelebA-HQ 数据集测试 FaceLock，包括批量编辑、防御和评估
# 前提：data 文件夹位于当前目录，包含多张图片

# 设置环境变量
export HF_HOME=/root/autodl-tmp/huggingface
export HF_ENDPOINT=https://hf-mirror.com

# 用户可配置参数
DATA_DIR="/root/cv/FaceLock/data_multi"  # 数据集路径
EDIT_DIR="/root/cv/FaceLock/celeba_hq_multi/edited"  # 批量编辑输出目录
PROTECTED_DIR="/root/cv/FaceLock/celeba_hq_multi/protected"  # 批量防御输出目录
PROTECTED_EDIT_DIR="/root/cv/FaceLock/celeba_hq_multi/protected_edited"  # 受保护图像编辑输出目录
DEFEND_METHOD="facelock"
SEED=42
NUM_INFERENCE_STEPS=100
IMAGE_GUIDANCE_SCALE=1.5
GUIDANCE_SCALE=7.5
ATTACK_BUDGET=0.03
STEP_SIZE=0.01
NUM_ITERS=50

# 创建输出目录（确保目录存在）
mkdir -p "$EDIT_DIR/seed$SEED/prompt0"
mkdir -p "$PROTECTED_DIR"
mkdir -p "$PROTECTED_EDIT_DIR/eps0/seed$SEED/prompt0"

# 检查 data 文件夹是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "错误：数据集文件夹 $DATA_DIR 不存在"
    exit 1
fi

# 检查图片列表是否为空
if [ -z "$(find "$DATA_DIR" -type f -name "*.jpg" -o -name "*.png")" ]; then
    echo "错误：$DATA_DIR 中没有找到任何图片（支持 .jpg 和 .png 格式）"
    exit 1
fi

# 步骤 1：批量编辑原始图像
echo "步骤 1：对 $DATA_DIR 中的图片进行批量编辑..."
python /root/cv/FaceLock/main_edit.py \
    --src_dir="$DATA_DIR" \
    --edit_dir="$EDIT_DIR" \
    --num_inference_steps=$NUM_INFERENCE_STEPS \
    --image_guidance_scale=$IMAGE_GUIDANCE_SCALE \
    --guidance_scale=$GUIDANCE_SCALE \
    --seed=$SEED
if [ $? -ne 0 ]; then
    echo "错误：批量编辑失败"
    exit 1
fi

# 步骤 2：批量保护原始图像
echo "步骤 2：对 $DATA_DIR 中的图片应用批量防御..."
python /root/cv/FaceLock/main_defend.py \
    --image_dir="$DATA_DIR" \
    --output_dir="$PROTECTED_DIR" \
    --defend_method="$DEFEND_METHOD" \
    --attack_budget=$ATTACK_BUDGET \
    --step_size=$STEP_SIZE \
    --num_iters=$NUM_ITERS
if [ $? -ne 0 ]; then
    echo "错误：批量防御失败"
    exit 1
fi

# 步骤 3：批量编辑受保护的图像
echo "步骤 3：对 $PROTECTED_DIR 中的受保护图片进行批量编辑..."
python /root/cv/FaceLock/main_edit.py \
    --src_dir="$PROTECTED_DIR/budget_$ATTACK_BUDGET" \
    --edit_dir="$PROTECTED_EDIT_DIR" \
    --num_inference_steps=$NUM_INFERENCE_STEPS \
    --image_guidance_scale=$IMAGE_GUIDANCE_SCALE \
    --guidance_scale=$GUIDANCE_SCALE \
    --seed=$SEED
if [ $? -ne 0 ]; then
    echo "错误：受保护图片批量编辑失败"
    exit 1
fi

# 步骤 4：评估
echo "步骤 4：评估防御效果..."
cd /root/cv/FaceLock/evaluation_multi || { echo "错误：无法进入 evaluation_multi 目录"; exit 1; }

# 评估 PSNR
echo "评估 PSNR..."
python eval_psnr.py \
    --clean_edit_dir="$EDIT_DIR" \
    --defend_edit_dirs="$PROTECTED_EDIT_DIR" \
    --seed=$SEED
if [ $? -ne 0 ]; then
    echo "错误：PSNR 评估失败"
    exit 1
fi

# 评估 SSIM
echo "评估 SSIM..."
python eval_ssim.py \
    --clean_edit_dir="$EDIT_DIR" \
    --defend_edit_dirs="$PROTECTED_EDIT_DIR" \
    --seed=$SEED
if [ $? -ne 0 ]; then
    echo "错误：SSIM 评估失败"
    exit 1
fi

# 评估 LPIPS
echo "评估 LPIPS..."
python eval_lpips.py \
    --clean_edit_dir="$EDIT_DIR" \
    --defend_edit_dirs="$PROTECTED_EDIT_DIR" \
    --seed=$SEED
if [ $? -ne 0 ]; then
    echo "错误：LPIPS 评估失败"
    exit 1
fi

# 评估 CLIP-S
echo "评估 CLIP-S..."
python eval_clip_s.py \
    --src_dir="$DATA_DIR" \
    --defend_edit_dirs="$PROTECTED_EDIT_DIR" \
    --clean_edit_dir="$EDIT_DIR" \
    --seed=$SEED
if [ $? -ne 0 ]; then
    echo "错误：CLIP-S 评估失败"
    exit 1
fi

# 评估 CLIP-I
echo "评估 CLIP-I..."
python eval_clip_i.py \
    --src_dir="$DATA_DIR" \
    --defend_edit_dirs="$PROTECTED_EDIT_DIR" \
    --clean_edit_dir="$EDIT_DIR" \
    --seed=$SEED
if [ $? -ne 0 ]; then
    echo "错误：CLIP-I 评估失败"
    exit 1
fi

# 评估 FR
echo "评估 FR..."
python eval_facial.py \
    --src_dir="$DATA_DIR" \
    --defend_edit_dirs="$PROTECTED_EDIT_DIR" \
    --clean_edit_dir="$EDIT_DIR" \
    --seed=$SEED
if [ $? -ne 0 ]; then
    echo "错误：FR 评估失败"
    exit 1
fi

cd /root/cv/FaceLock

echo "测试完成！结果保存在 $EDIT_DIR, $PROTECTED_DIR, $PROTECTED_EDIT_DIR"
echo "评估结果已输出，请检查控制台或相关日志"