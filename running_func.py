import os
import random
import numpy as np
import torch
import h5py
import time
import glob
import re
import torch.nn as nn
from torch.nn import init
import torchvision as tv
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio
# from skimage.metrics import structural_similarity as ssim
import cv2
from torchvision import utils as tv
from skimage.metrics import structural_similarity as ssim_skimage  # 保留skimage的SSIM
from torchmetrics.image import PeakSignalNoiseRatio  # 新增：更精确的PSNR计算
from kornia.metrics import ssim as ssim_kornia  # 新增：GPU加速的SSIM
def mk_trained_dir_if_not(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def model_restore(model, trained_model_dir):
    """改进的模型恢复函数
    
    Args:
        model: 要恢复参数的模型实例
        trained_model_dir: 检查点文件存储目录
        
    Returns:
        model: 恢复参数后的模型
        start_step: 恢复的训练步数（若无有效检查点则返回0）
    """
    # 获取目录下所有.pt和.pkl文件（兼容两种格式）
    model_list = glob.glob(os.path.join(trained_model_dir, "*.pt")) + \
                 glob.glob(os.path.join(trained_model_dir, "*.pkl"))
    
    if not model_list:
        print(f"No checkpoint found in {trained_model_dir}, start from scratch.")
        return model, 0

    # 解析epoch数值（支持多种文件名格式）
    valid_checkpoints = []
    for model_path in model_list:
        try:
            filename = os.path.basename(model_path)
            
            # 支持格式1: trained_model_100.pkl
            # 支持格式2: model_epoch_100.pt
            match = re.search(r'(_|\.)(\d+)\.(pt|pkl)$', filename)
            if match:
                epoch = int(match.group(2))
                valid_checkpoints.append((epoch, model_path))
            else:
                print(f"Skipping invalid checkpoint (bad naming): {filename}")
        except Exception as e:
            print(f"Skipping invalid checkpoint {filename}: {str(e)}")

    if not valid_checkpoints:
        print("No valid checkpoints found, start from scratch.")
        return model, 0

    # 找到epoch最大的checkpoint
    valid_checkpoints.sort(key=lambda x: x[0])
    max_epoch, latest_path = valid_checkpoints[-1]
    
    try:
        checkpoint = torch.load(latest_path, map_location='cpu')
        
        # 兼容不同保存格式
        if isinstance(checkpoint, dict):
            # 格式1: 完整checkpoint字典 {'model': ..., 'step': ...}
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
                start_step = checkpoint.get('step', max_epoch)
            # 格式2: 直接保存的state_dict
            else:
                model.load_state_dict(checkpoint, strict=False)
                start_step = max_epoch
        else:
            # 格式3: 直接保存的模型参数
            model.load_state_dict(checkpoint, strict=False)
            start_step = max_epoch
            
        print(f"Successfully loaded checkpoint from {latest_path}")
        print(f"Resuming training from epoch {max_epoch}, step {start_step}")
        return model, start_step
        
    except Exception as e:
        print(f"Error loading checkpoint {latest_path}: {str(e)}")
        print("Attempting partial load with strict=False...")
        try:
            checkpoint = torch.load(latest_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            print("Partial load successful (some layers may be randomly initialized)")
            return model, max_epoch
        except:
            print("Failed to load checkpoint, starting from scratch.")
            return model, 0
def formulate_hdr(x):
    assert len(x.shape) == 4
    _hdr = torch.clamp(x, 0, 1)
    _hdr = torch.round(_hdr[0] * 255)
    return _hdr

class data_loader(data.Dataset):
    def __init__(self, list_dir):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt)



    def __getitem__(self, index):

        sample_path = self.list_txt[index][:-1]

        if os.path.exists(sample_path):

            f = h5py.File(sample_path, 'r')
            data = f['IN'][:]
            label = f['GT'][:]
            f.close()
            crop_size = 256
            data, label = self.imageCrop(data, label, crop_size)
            data, label = self.image_Geometry_Aug(data, label)


        # print(sample_path)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return self.length

    def random_number(self, num):
        return random.randint(1, num)

    def imageCrop(self, data, label, crop_size):
        c, w, h = data.shape
        w_boder = w - crop_size  # sample point y
        h_boder = h - crop_size  # sample point x ...

        start_w = self.random_number(w_boder - 1)
        start_h = self.random_number(h_boder - 1)

        crop_data = data[:, start_w:start_w + crop_size, start_h:start_h + crop_size]
        crop_label = label[:, start_w:start_w + crop_size, start_h:start_h + crop_size]
        return crop_data, crop_label

    def image_Geometry_Aug(self, data, label):
        c, w, h = data.shape
        num = self.random_number(4)

        if num == 1:
            in_data = data
            in_label = label

        if num == 2:  # flip_left_right
            index = np.arange(w, 0, -1) - 1
            in_data = data[:, index, :]
            in_label = label[:, index, :]

        if num == 3:  # flip_up_down
            index = np.arange(h, 0, -1) - 1
            in_data = data[:, :, index]
            in_label = label[:, :, index]

        if num == 4:  # rotate 180
            index = np.arange(w, 0, -1) - 1
            in_data = data[:, index, :]
            in_label = label[:, index, :]
            index = np.arange(h, 0, -1) - 1
            in_data = in_data[:, :, index]
            in_label = in_label[:, :, index]

        return in_data, in_label

def get_lr(epoch, lr, max_epochs):
    if epoch <= max_epochs * 0.8:
        lr = lr
    else:
        lr = 0.1 * lr
    return lr


def train(epoch, model, train_loaders, optimizer, args):
    # 初始化设备
    device = next(model.parameters()).device
    model.train()

    # 预计算HDR参数（固定值移到GPU）
    scale = torch.tensor([1 + 5000], dtype=torch.float32, device=device)
    log_scale = torch.log(scale)

    # 训练统计
    total_loss = 0.0
    batch_count = 0
    start_time = time.time()

    # 强制打印头
    print("\n=== 训练开始 ===")
    print(f"设备: {device}")
    print(f"数据加载器长度: {len(train_loaders)}")

    def hdr_transform(x):
        """
        将输入数据转换为HDR格式
        """
        return torch.log(1 + 5000 * x) / log_scale

    for batch_idx, (data, target) in enumerate(train_loaders):
        # 数据验证
        if data is None or target is None:
            print(f"警告: Batch {batch_idx} 数据为空！")
            continue

        # 数据转移
        try:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
        except Exception as e:
            print(f"数据转移到设备时出错: {str(e)}")
            continue

        # 数据分块
        try:
            data1 = torch.cat((data[:, 0:3], data[:, 9:12]), dim=1)
            data2 = torch.cat((data[:, 3:6], data[:, 12:15]), dim=1)
            data3 = torch.cat((data[:, 6:9], data[:, 15:18]), dim=1)
        except IndexError as e:
            print(f"数据分块错误: {str(e)}")
            continue

        # 前向传播
        optimizer.zero_grad()
        output = model(data1, data2, data3)

        # Loss计算
        output_hdr = hdr_transform(output)
        target_hdr = hdr_transform(target)
        loss = F.l1_loss(output_hdr, target_hdr)

        # 反向传播
        loss.backward()

        # 检查梯度
        none_grad_params = [name for name, p in model.named_parameters() if p.grad is None]
        if none_grad_params:
            print(f"警告: 以下参数的梯度为None: {none_grad_params}")
        else:
            optimizer.step()

        # 累计统计
        total_loss += loss.item()
        batch_count += 1

    # Epoch总结
    if batch_count > 0:
        avg_loss = total_loss / batch_count
        end_time = time.time()
        train_duration = end_time - start_time
        print(f'\n=== Epoch {epoch} 平均Loss: {avg_loss:.6f}，训练时长: {train_duration:.2f} 秒 ===')
        with open(os.path.join(args.trained_model_dir, 'loss_log.txt'), 'a') as f:
            f.write(f'Epoch {epoch}, 平均Loss: {avg_loss:.6f}, 训练时长: {train_duration:.2f} 秒\n')
    else:
        print('\n=== 本Epoch无有效数据 ===')
def testing_fun(model, test_loaders, args):
    model.eval()
    test_loss = 0.0
    psnr_l = 0.0  # 线性空间PSNR
    psnr = 0.0    # HDR空间PSNR
    SSIM_l = 0.0  # 线性空间SSIM
    SSIM = 0.0    # HDR空间SSIM
    num_samples = 0

    with torch.no_grad():
        os.makedirs(args.result_dir, exist_ok=True)
        device = next(model.parameters()).device
        psnr_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        scale = torch.tensor([1 + 5000], dtype=torch.float32, device=device)

        for batch_idx, (data, target) in enumerate(test_loaders):
            # 数据加载
            Test_Data_name = os.path.splitext(
                os.path.basename(test_loaders.dataset.list_txt[batch_idx])
            )[0]
            data, target = data.to(device), target.to(device)

            # 数据拼接
            data_pairs = [(0,3,9,12), (3,6,12,15), (6,9,15,18)]
            data_inputs = [
                torch.cat((data[:, s1:e1, :], data[:, s2:e2, :]), dim=1)
                for (s1,e1,s2,e2) in data_pairs
            ]

            # 前向传播
            output = model(*data_inputs)

            # 维度处理函数
            def prepare_for_metrics(tensor):
                if tensor.dim() == 5:  # 如果有多余的batch维度
                    tensor = tensor.squeeze(0)
                if tensor.dim() == 3:  # 确保有batch维度
                    tensor = tensor.unsqueeze(0)
                return tensor

            # --- 线性空间计算 ---
            output_prepared = prepare_for_metrics(output)
            target_prepared = prepare_for_metrics(target)
            
            # PSNR计算
            psnr_l += psnr_calculator(output_prepared, target_prepared)
            
            # SSIM计算
            SSIM_l += ssim_kornia(
                output_prepared,
                target_prepared,
                window_size=11,
                max_val=1.0
            ).mean().item()

            # --- HDR空间计算 ---
            hdr = torch.log(1 + 5000 * output) / torch.log(scale)
            target_scaled = torch.log(1 + 5000 * target) / torch.log(scale)
            
            hdr_prepared = prepare_for_metrics(hdr)
            target_scaled_prepared = prepare_for_metrics(target_scaled)
            
            # HDR PSNR
            psnr += psnr_calculator(hdr_prepared, target_scaled_prepared)
            
            # HDR SSIM
            SSIM += ssim_kornia(
                hdr_prepared,
                target_scaled_prepared,
                window_size=11,
                max_val=1.0
            ).mean().item()

            # 损失计算
            test_loss += F.mse_loss(hdr, target_scaled).item()

            # 保存结果
            output_path = os.path.join(args.result_dir, f"{Test_Data_name}_hdr.h5")
            with h5py.File(output_path, 'w') as hdrfile:
                img_grid = tv.make_grid(output[0].detach().cpu()).numpy()
                hdrfile.create_dataset('data', data=img_grid)

            # 图像后处理
            c = formulate_hdr(hdr)
            c = c.data.cpu().numpy().astype(np.uint8)
            c = c.transpose(1, 2, 0)
            c = cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE)
            c = cv2.flip(c, 1)
            c = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join('./result', f'{Test_Data_name}.png'), c)

            num_samples += 1

    # 计算平均值
    test_loss /= num_samples
    psnr_l /= num_samples
    psnr /= num_samples
    SSIM_l /= num_samples
    SSIM /= num_samples

    print('\nTest set:')
    print(f'Average Loss: {test_loss:.4f}')
    print(f'Linear Space - PSNR: {psnr_l:.4f} dB, SSIM: {SSIM_l:.4f}')
    print(f'HDR Space (μ) - PSNR: {psnr:.4f} dB, SSIM: {SSIM:.4f}')

    return psnr
class testimage_dataloader(data.Dataset):
    def __init__(self, list_file):
        with open(list_file, 'r') as f:
            self.list_txt = [line.strip() for line in f]  
        self.length = len(self.list_txt)
    
    def __getitem__(self, index):
        sample_path = self.list_txt[index] 
        try:
            with h5py.File(sample_path, 'r') as f:
                data = f['IN'][:].astype(np.float32)
                label = f['GT'][:].astype(np.float32)
        except (FileNotFoundError, KeyError) as e:
            print(f"Error loading {sample_path}: {str(e)}")
            # 返回空数据并跳过
            data = np.zeros((18, 256, 256), dtype=np.float32)
            label = np.zeros((3, 256, 256), dtype=np.float32)
        
        return torch.from_numpy(data), torch.from_numpy(label)
    
    def __len__(self):
        return self.length
    
    def random_sample(self):
        return random.choice(self.list_txt)  