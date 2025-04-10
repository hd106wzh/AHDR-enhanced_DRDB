import torch
import numpy as np
import time
import argparse
import torch.optim as optim
import torch.utils.data
import scipy.io as scio
from torch.nn import init
from dataset import DatasetFromHdf5
from running_func import *
from utils import *
from model6 import *
import os

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Attention-guided HDR')

# 定义命令行参数
parser.add_argument('--train-data', default='train.txt')
parser.add_argument('--test_whole_Image', default='./test.txt')
parser.add_argument('--trained_model_dir', default='./tr6_model/')#用于存放模型参数文件
parser.add_argument('--trained_model_filename', default='ahdr_model.pt')
parser.add_argument('--result_dir', default='./result/')
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--restore', default=True)
parser.add_argument('--load_model', default=True)
parser.add_argument('--lr', default=0.00002)#初始学习率为0.0001
parser.add_argument('--seed', default=1)
parser.add_argument('--batchsize', default=8)
parser.add_argument('--epochs', default=20000)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--save_model_interval', default=5)

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')

# 解析命令行参数
args = parser.parse_args()

# 设置随机种子以保证结果可复现
torch.manual_seed(args.seed)
# 检查是否有可用的GPU，若有则设置为计算设备
device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")
if args.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# 加载训练数据
train_loaders = torch.utils.data.DataLoader(
    data_loader(args.train_data),
    batch_size=args.batchsize, shuffle=True, num_workers=4
)

# 加载测试数据
testimage_dataset = torch.utils.data.DataLoader(
    testimage_dataloader(args.test_whole_Image),
    batch_size=1
)

# 创建结果目录
mk_dir(args.result_dir)

# 定义Kaiming初始化函数，用于初始化卷积层的权重
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)

# 创建模型并将其移动到指定设备（GPU或CPU）
model = AHDR_Enhanced(args).to(device)
# 对模型应用Kaiming初始化
model.apply(weights_init_kaiming)

# 定义Adam优化器
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

# 初始化起始步骤
start_step = 0
# 检查是否需要恢复训练以及模型目录中是否有模型文件
if args.restore and len(os.listdir(args.trained_model_dir)):
    model, start_step = model_restore(model, args.trained_model_dir)
    print('restart from {} step'.format(start_step))

# 创建训练模型保存目录
mk_dir(args.trained_model_dir)

# 初始化最佳PSNR、当前PSNR和标志变量
best_psnr = 0.0
psnr = 0.0
flag = 0.0

# 训练循环
for epoch in range(start_step + 1, args.epochs + 1):
    start = time.time()
    # 进行一轮训练
    train(epoch, model, train_loaders, optimizer, args)
    end = time.time()
    print('epoch:{}, cost {} seconds'.format(epoch, end - start))

    # 在训练n个epoch后，开启测试和学习率调整，根据自己的需求调整学习率以及开始测试的epoch。由于前期网络还未收敛，可以将学习率调高点
    if epoch > 16500:
        # 学习率调整逻辑
        # if flag >= 15:
        #     flag = 0
        #     lr1 = optimizer.param_groups[0]['lr']
        #     if lr1 >= 1e-6: 
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = lr1 * 0.5
        #     else:
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = 5e-5         
        
        # 进行测试并计算PSNR
        psnr = testing_fun(model, testimage_dataset, args)
        if psnr > best_psnr:
            best_psnr = psnr
            model_name = os.path.join(args.trained_model_dir, 'p19_trained_bestmodel_AHDR.pt')
            torch.save(model.state_dict(), model_name)
        #     flag = 0
        # else:
        #     flag = flag + 1
        print('\n Test set: Average pnsr: {:.4f}, best pnsr: {:.4f},flag:{:.4f}'.format(psnr, best_psnr, flag))

    # 每 save_model_interval 个epoch保存一次模型
    if epoch % args.save_model_interval == 0:
        model_name = os.path.join(args.trained_model_dir, f'trained_model_{epoch}.pkl')
        torch.save(model.state_dict(), model_name)