import os
import warnings
from SignRecognition import Sign_Recognition
import atexit

def ensure_model_dirs():
    """确保模型目录存在"""
    # 创建模型存储目录
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # 检查模型文件是否存在
    model_path = os.path.join(model_dir, 'cnn_model.h5')
    if os.path.exists(model_path):
        pass
    
    return True

def cleanup():
    # 程序退出时执行的清理函数
    try:
        # 释放全局应用资源
        if hasattr(cleanup, 'recognition'):
            cleanup.recognition.cleanup()
    except Exception:
        # 静默处理清理过程中的错误
        pass

if __name__ == '__main__':
    # 忽略警告
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings(action='ignore')
    
    # 确保模型目录存在
    ensure_model_dirs()
    
    # 注册退出清理函数
    atexit.register(cleanup)
    
    # 创建识别对象并启动
    recognition = Sign_Recognition()
    cleanup.recognition = recognition  # 保存引用以便清理
    recognition.start_camera()
