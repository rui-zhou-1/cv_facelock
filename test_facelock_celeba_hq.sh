#!/bin/bash

# 脚本：test_cv_facelock_dataset.sh
# 用途：使用 data 文件夹中的 CelebA-HQ 数据集测试 cv_facelock，包括编辑、防御和评估
# 前提：data 文件夹位于当前目录，包含多张图片

BASE_PATH="/root/cv/cv_facelock" # to修改: 当前路径

# 设置环境变量
export HF_HOME=/root/autodl-tmp/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="0"                # to修改: 指定GPU

# 用户可配置参数
DATA_DIR="${BASE_PATH}/input"  # 数据集路径
EDIT_DIR="${BASE_PATH}/output/edited"
PROTECTED_DIR="${BASE_PATH}/output/protected"
PROTECTED_EDIT_DIR="${BASE_PATH}/output/protected_edited"
PROMPT="Turn the person's hair pink"  # 使用 edit_prompts[0]
DEFEND_METHOD="facelock"
SEED=42
NUM_INFERENCE_STEPS=100
IMAGE_GUIDANCE_SCALE=1.5
GUIDANCE_SCALE=7.5
ATTACK_BUDGET=0.03
STEP_SIZE=0.01
NUM_ITERS=100

# 检查 data 文件夹是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "错误：数据集文件夹 $DATA_DIR 不存在"
    exit 1
fi

# 创建输出目录（确保目录存在）
mkdir -p "$EDIT_DIR/seed$SEED/prompt0"
mkdir -p "$PROTECTED_DIR"
mkdir -p "$PROTECTED_EDIT_DIR/eps0/seed$SEED/prompt0"

# 获取图片列表
IMAGE_LIST=$(find "$DATA_DIR" -type f -name "*.jpg" -o -name "*.png")

# 检查图片列表是否为空
if [ -z "$IMAGE_LIST" ]; then
    echo "错误：$DATA_DIR 中没有找到任何图片（支持 .jpg 和 .png 格式）"
    exit 1
fi

# 遍历每张图片
for INPUT_IMAGE in $IMAGE_LIST; do
    # 提取文件名（不含路径和扩展名）
    FILENAME=$(basename "$INPUT_IMAGE")
    IMAGE_NAME="${FILENAME%.*}"

    echo "处理图片：$FILENAME"

    # 检查输入图片是否存在
    if [ ! -f "$INPUT_IMAGE" ]; then
        echo "错误：输入图片 $INPUT_IMAGE 不存在，跳过"
        continue
    fi

    # # 复制原始图片到 SRC_DIR
    # cp "$INPUT_IMAGE" "$SRC_DIR/$FILENAME"

    # 步骤 1：编辑单张原始图像
    echo "步骤 1：对图片 $FILENAME 进行编辑..."
    python edit.py \
        --input_path="$INPUT_IMAGE" \
        --output_path="$EDIT_DIR/${IMAGE_NAME}_edited.jpg" \
        --prompt="$PROMPT" \
        --num_inference_steps=$NUM_INFERENCE_STEPS \
        --image_guidance_scale=$IMAGE_GUIDANCE_SCALE \
        --guidance_scale=$GUIDANCE_SCALE \
        --seed=$SEED
    if [ $? -ne 0 ]; then
        echo "错误：图片 $FILENAME 编辑失败，跳过"
        continue
    fi

    # 步骤 2：保护单张原始图像
    echo "步骤 2：对图片 $FILENAME 应用防御..."
    python defend.py \
        --input_path="$INPUT_IMAGE" \
        --output_path="$PROTECTED_DIR/${IMAGE_NAME}_protected.jpg" \
        --defend_method="$DEFEND_METHOD" \
        --attack_budget=$ATTACK_BUDGET \
        --step_size=$STEP_SIZE \
        --num_iters=$NUM_ITERS
    if [ $? -ne 0 ]; then
        echo "错误：图片 $FILENAME 防御失败，跳过"
        continue
    fi

    # 步骤 3：编辑受保护的图像
    echo "步骤 3：对受保护的图片 $FILENAME 进行编辑..."
    python edit.py \
        --input_path="$PROTECTED_DIR/${IMAGE_NAME}_protected.jpg" \
        --output_path="$PROTECTED_EDIT_DIR/${IMAGE_NAME}_protected_edited.jpg" \
        --prompt="$PROMPT" \
        --num_inference_steps=$NUM_INFERENCE_STEPS \
        --image_guidance_scale=$IMAGE_GUIDANCE_SCALE \
        --guidance_scale=$GUIDANCE_SCALE \
        --seed=$SEED
    if [ $? -ne 0 ]; then
        echo "错误：受保护图片 $FILENAME 编辑失败，跳过"
        continue
    fi

    # 调整目录结构以兼容评估脚本
    echo "调整图片 $FILENAME 的目录结构..."
    mv "$EDIT_DIR/${IMAGE_NAME}_edited.jpg" "$EDIT_DIR/seed$SEED/prompt0/${IMAGE_NAME}_edited.jpg"
    mv "$PROTECTED_EDIT_DIR/${IMAGE_NAME}_protected_edited.jpg" "$PROTECTED_EDIT_DIR/eps0/seed$SEED/prompt0/${IMAGE_NAME}_protected_edited.jpg"
done

# 步骤 4：评估
echo "步骤 4：评估防御效果..."
cd evaluation || { echo "错误：无法进入 evaluation 目录"; exit 1; }

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

cd ..

echo "测试完成！结果保存在 $EDIT_DIR, $PROTECTED_DIR, $PROTECTED_EDIT_DIR"
echo "评估结果已输出，请检查控制台或相关日志"