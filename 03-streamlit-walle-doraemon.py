import streamlit as st
import sys
import pathlib
from fastai.vision.all import *

# Python 版本检查
if sys.version_info >= (3, 13):
    st.error("⚠️ 当前 Python 版本为 3.13+，可能与 fastai 不兼容。建议使用 Python 3.11。")
    st.stop()

def load_model():
    """加载模型并添加错误处理"""
    # 修正：添加了st前缀
    st.info("正在加载模型...")
    
    try:
        # 调试：显示当前工作目录和文件路径
        st.write(f"当前工作目录: {pathlib.Path.cwd()}")
        
        # 尝试从当前目录加载模型
        model_path = pathlib.Path("doraemon_walle_model.pkl")
        
        # 如果模型不存在，尝试使用__file__路径
        if not model_path.exists():
            st.warning(f"模型文件 '{model_path}' 不存在，尝试使用替代路径...")
            # 注意：在某些环境中__file__可能未定义，需要额外处理
            try:
                model_path = pathlib.Path(__file__).parent / "doraemon_walle_model.pkl"
            except NameError:
                st.warning("无法使用__file__路径，使用当前工作目录")
                model_path = pathlib.Path("doraemon_walle_model.pkl")
        
        st.write(f"尝试加载模型: {model_path}")
        
        # 检查文件是否存在
        if not model_path.exists():
            st.error(f"模型文件不存在: {model_path}")
            return None
        
        # Windows 路径兼容性处理
        if sys.platform == "win32":
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
        
        try:
            model = load_learner(model_path)
            st.success("模型加载成功!")
            return model
        finally:
            # 恢复原始设置
            if sys.platform == "win32":
                pathlib.PosixPath = temp
                
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        # 打印详细错误信息到控制台
        import traceback
        st.write(traceback.format_exc())
        return None

# 主应用
st.title("图像分类应用")
st.write("上传一张图片，应用将预测对应的标签。")

# 使用session_state缓存模型，避免重复加载
if "model" not in st.session_state:
    st.session_state.model = load_model()

model = st.session_state.model

if model is None:
    st.warning("模型加载失败，请检查模型文件是否存在并可访问。")
else:
    uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = PILImage.create(uploaded_file)
        st.image(image, caption="上传的图片", use_container_width=True)
        
        try:
            pred, pred_idx, probs = model.predict(image)
            st.write(f"预测结果: {pred}; 概率: {probs[pred_idx]:.04f}")
        except Exception as e:
            st.error(f"预测失败: {str(e)}")
