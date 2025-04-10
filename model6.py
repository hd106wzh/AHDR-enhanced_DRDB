import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d  # 官方可变形卷积

class CALayer(nn.Module):
    
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
class EnhancedDRDB(nn.Module):
    def __init__(self, nChannels=64, nDenselayer=6, growthRate=32):
        super().__init__()
        self.nChannels = nChannels
        self.nDenselayer = nDenselayer
        self.growthRate = growthRate
        
        #
        self.dilated_convs = nn.ModuleList()
        current_channels = nChannels  # 跟踪当前总通道数
        
        for i in range(nDenselayer):
            conv = nn.Sequential(
                nn.Conv2d(current_channels, growthRate, 3, 
                         padding=2, dilation=2),
                nn.LeakyReLU(0.01, True)
            )
            self.dilated_convs.append(conv)
            current_channels += growthRate  # 每层增加growthRate通道
        
        # 特征压缩层（输入通道应为初始64 + 6*32 = 256）
        self.compression = nn.Sequential(
            nn.Conv2d(nChannels + nDenselayer * growthRate, nChannels, 1),
            CALayer(nChannels)
        )
        
        # 可变形卷积
        self.offset_conv = nn.Conv2d(nChannels, 2*3*3, 3, padding=1)
        self.deform_conv = DeformConv2d(
            nChannels, nChannels, 
            kernel_size=3, 
            padding=1
        )

    def forward(self, x):
        identity = x
        all_features = [x]  # 包含初始特征
        current = x  # 当前累积的特征
        
        for conv in self.dilated_convs:
            # 当前层处理
            out = conv(current)
            all_features.append(out)
            # 更新当前累积特征（初始x + 所有中间输出）
            current = torch.cat(all_features, dim=1)
        
        # 最终拼接（应包含初始x和所有中间输出）
        concat_features = torch.cat(all_features, dim=1)
        
        # # 调试验证
        # print(f"理论总通道数: {self.nChannels + self.nDenselayer * self.growthRate}")
        # print(f"实际拼接通道数: {concat_features.size(1)}")
        
        # 特征压缩
        compressed = self.compression(concat_features)
        
        # 可变形卷积
        offset = self.offset_conv(compressed)
        out = self.deform_conv(compressed, offset)
        
        return out + identity

class AHDR_Enhanced(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        nChannel, nFeat = args.nChannel, args.nFeat
        nDenselayer, growthRate = args.nDenselayer, args.growthRate

        # 特征提取
        self.conv_init = nn.Sequential(
            nn.Conv2d(nChannel, nFeat, 3, padding=1),
            nn.LeakyReLU(0.01, True)
        )
        
        # 注意力模块
        self.attention1 = nn.Sequential(
            nn.Conv2d(nFeat*2, nFeat*2, 3, padding=1),
            nn.LeakyReLU(0.01, True),
            nn.Conv2d(nFeat*2, nFeat, 3, padding=1),
            nn.Sigmoid()
        )
        self.attention3 = nn.Sequential(
            nn.Conv2d(nFeat*2, nFeat*2, 3, padding=1),
            nn.LeakyReLU(0.01, True),
            nn.Conv2d(nFeat*2, nFeat, 3, padding=1),
            nn.Sigmoid()
        )

        # 替换为增强版DRDB块
        self.rdbs = nn.Sequential(
            EnhancedDRDB(nFeat, nDenselayer, growthRate),
            EnhancedDRDB(nFeat, nDenselayer, growthRate),
            EnhancedDRDB(nFeat, nDenselayer, growthRate)
        )

        # 特征融合与重建（保持原始结构）
        self.conv_fusion = nn.Conv2d(nFeat*3, nFeat, 3, padding=1)
        self.conv_reconstruct = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, 3, padding=1),
            nn.Conv2d(nFeat, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3):
        # 特征提取
        f1 = self.conv_init(x1)
        f2 = self.conv_init(x2)
        f3 = self.conv_init(x3)

        # 注意力加权
        f1 = f1 * self.attention1(torch.cat([f1, f2], dim=1))
        f3 = f3 * self.attention3(torch.cat([f3, f2], dim=1))

        # 特征融合
        fused = self.conv_fusion(torch.cat([f1, f2, f3], dim=1))

        # 增强的DRDB处理
        features = self.rdbs(fused)

        # 重建
        return self.conv_reconstruct(features + f2)  # 残差连接