import os
import torch

def mk_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def model_load(model, trained_model_dir, model_file_name):
    model_path = os.path.join(trained_model_dir, model_file_name)
    # trained_model_dir + model_file_name    # '/modelParas.pkl'
    model.load_state_dict(torch.load(model_path))
    return model
# def model_load(model, model_path):
#     checkpoint = torch.load(model_path, map_location='cpu')  # 加载到CPU避免设备不匹配
#     if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['model_state_dict'])  # 从检查点提取模型权重
#     else:
#         model.load_state_dict(checkpoint)  # 直接加载纯权重文件
#     return model