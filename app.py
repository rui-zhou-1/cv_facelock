import streamlit as st
import os
import subprocess
from PIL import Image
import tempfile

def main():
    st.title("CV FaceLock图像编辑应用")

    # 设置Hugging Face环境变量
    #os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
    #os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 使用第二个GPU（如果有多个GPU）
    
    # 创建临时目录存储图片
    temp_dir = tempfile.mkdtemp()
    
    # 上传图片
    uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])
    
    # 输入提示词
    prompt = st.text_input("输入编辑提示词", "Turn the person's hair pink")
    
    if uploaded_file is not None:
        # 显示原始图片
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="原始图片", width=300)
        
        # 保存上传的图片到临时文件
        input_path = os.path.join(temp_dir, "input.jpg")
        original_image = original_image.convert("RGB")
        original_image.save(input_path)
        
        # 设置输出路径
        protected_path = os.path.join(temp_dir, "protected.jpg")
        edited_original_path = os.path.join(temp_dir, "edited_original.jpg")
        edited_protected_path = os.path.join(temp_dir, "edited_protected.jpg")
        
        # 处理按钮
        if st.button("开始处理"):
            with st.spinner("正在进行图像防御处理..."):
                # 运行defend.py
                defend_cmd = [
                    "python", "/Data/zr/cv/defend.py",
                    "--input_path", input_path,
                    "--output_path", protected_path,
                    "--defend_method", "facelock",
                    "--attack_budget", "0.03",
                    "--step_size", "0.01",
                    "--num_iters", "100"
                ]
                subprocess.run(defend_cmd, check=True)
                
                # 显示防御后的图片
                if os.path.exists(protected_path):
                    protected_image = Image.open(protected_path)
                    st.image(protected_image, caption="防御后图片", width=300)
            
            with st.spinner("正在编辑原始图片..."):
                # 编辑原始图片
                edit_original_cmd = [
                    "python", "/Data/zr/cv/edit.py",
                    "--input_path", input_path,
                    "--output_path", edited_original_path,
                    "--prompt", prompt,
                    "--num_inference_steps", "100",
                    "--image_guidance_scale", "1.5",
                    "--guidance_scale", "7.5"
                ]
                subprocess.run(edit_original_cmd, check=True)
                
                # 显示编辑后的原始图片
                if os.path.exists(edited_original_path):
                    edited_original = Image.open(edited_original_path)
                    st.image(edited_original, caption="原始图片编辑结果", width=300)
            
            with st.spinner("正在编辑防御后图片..."):
                # 编辑防御后图片
                edit_protected_cmd = [
                    "python", "/Data/zr/cv/edit.py",
                    "--input_path", protected_path,
                    "--output_path", edited_protected_path,
                    "--prompt", prompt,
                    "--num_inference_steps", "100",
                    "--image_guidance_scale", "1.5",
                    "--guidance_scale", "7.5"
                ]
                subprocess.run(edit_protected_cmd, check=True)
                
                # 显示编辑后的防御图片
                if os.path.exists(edited_protected_path):
                    edited_protected = Image.open(edited_protected_path)
                    st.image(edited_protected, caption="防御图片编辑结果", width=300)
            
            st.success("处理完成！")

if __name__ == "__main__":
    main()