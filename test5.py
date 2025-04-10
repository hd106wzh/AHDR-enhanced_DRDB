import argparse
import torch
import torch.utils.data
from running_func import *
from model6 import *  # 假设你有对应的模型文件

parser = argparse.ArgumentParser(description='Attention-guided HDR Testing')

# 参数设置

parser.add_argument('--test_whole_Image', default='./test.txt')
parser.add_argument('--trained_model_dir', default='./')
parser.add_argument('--trained_model_filename', default='trained_model_20000.pt')#trained_model_20000.pt是训练出来最优的模型（不是训练了2w轮，是我把后缀改成了2w，实际大概练了1w7轮）
parser.add_argument('--result_dir', default='./result/')
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--load_model', default=True)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--seed', default=1)

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')

args = parser.parse_args()

# 设置随机种子
torch.manual_seed(args.seed)
if args.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 数据加载
testimage_dataset = testimage_dataloader(args.test_whole_Image)
testimage_dataloader = torch.utils.data.DataLoader(
    testimage_dataset,
    batch_size=1,
    pin_memory=True if args.use_cuda else False
)

# 创建结果目录
mk_trained_dir_if_not(args.result_dir)

# 模型初始化
model = AHDR_Enhanced(args).to(device)  # 假设模型类为AHDR_Enhanced

# 模型加载
if args.load_model:
    model, _ = model_restore(model, args.trained_model_dir)

# 测试过程
test_loss = testing_fun(model, testimage_dataloader, args)

print('Testing finished.')