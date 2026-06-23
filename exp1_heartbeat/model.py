"""
实验一：心跳信号分类预测
模型定义

提供两个模型，便于对比：
  1. ECGNet     —— 参考报告同款的简洁 1D-CNN（Conv + LeakyReLU + BN + MaxPool + Dropout），
                    结构清晰、训练快，可稳定复现高准确率。
  2. HeartbeatResNet —— 带残差连接的 1D ResNet（改进版），
                    通过残差块缓解梯度消失，对少数类别的特征提取更鲁棒。

损失函数（两个模型均使用）：
  多分类交叉熵 L = -sum_k [ y_k * log(softmax(z_k)) ]
  PyTorch 的 nn.CrossEntropyLoss 内部包含 softmax + log，网络只需输出 logits。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 模型一：ECGNet（参考报告同款简洁 CNN）
# ============================================================

class ECGNet(nn.Module):
    """
    一维卷积心跳分类网络（输入固定长度 205）。

    结构（核心思想：多层 Conv + LeakyReLU + BN + MaxPool + Dropout）：
      输入 (B, 1, 205)
      BN(1) -> Conv1d(1->32,k=11,p=5) -> LeakyReLU
      BN(32) -> Conv1d(32->64,k=11,p=5) -> LeakyReLU -> MaxPool(4)  => (B,64,51)
      Conv1d(64->128,k=3,p=1) -> LeakyReLU
      Conv1d(128->256,k=3,p=1) -> LeakyReLU -> MaxPool(4)           => (B,256,12)
      Dropout(0.1)
      Flatten -> FC(256*12->1024) -> LeakyReLU -> FC(1024->128) -> LeakyReLU -> FC(128->4)

    说明：心跳信号为 205 维时序，较大卷积核(11)用于捕捉宽幅波形特征，
          较小卷积核(3)用于提取局部细节。
    """

    def __init__(self, num_classes: int = 4, seq_len: int = 205, dropout: float = 0.1):
        super().__init__()
        self.conv_unit = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.LeakyReLU(),
            nn.MaxPool1d(4),                       # 205 -> 51
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(4),                       # 51 -> 12
            nn.Dropout(dropout),
        )
        # 通过一次前向推断展平后的维度，避免长度变化时手算出错
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_len)
            flat_dim = self.conv_unit(dummy).flatten(1).shape[1]

        self.fc_unit = nn.Sequential(
            nn.Linear(flat_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, L)；若传入 (B, L) 则自动补 channel 维
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_unit(x)
        x = x.flatten(1)
        return self.fc_unit(x)


# ============================================================
# 模型二：HeartbeatResNet（改进版，带残差连接）
# ============================================================

class ResBlock1D(nn.Module):
    """
    一维残差块（Pre-activation 风格）：
      x -> BN -> ReLU -> Conv -> Dropout -> BN -> ReLU -> Conv -> + shortcut(x)
    输入输出通道或 stride 不一致时，shortcut 用 1x1 卷积做维度匹配。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.dropout(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        return out + self.shortcut(x)


class HeartbeatResNet(nn.Module):
    """
    1D ResNet 用于心跳信号 4 分类。

      stem  : Conv1d(1->32,k=7) + BN + ReLU + MaxPool
      layer1: ResBlock(32->64,  stride=2)
      layer2: ResBlock(64->128, stride=2)
      layer3: ResBlock(128->256,stride=2)
      GAP -> FC(256->128) -> ReLU -> Dropout -> FC(128->4)
    全局平均池化使模型对序列长度鲁棒，参数量更小。
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = ResBlock1D(32, 64, stride=2, dropout=dropout)
        self.layer2 = ResBlock1D(64, 128, stride=2, dropout=dropout)
        self.layer3 = ResBlock1D(128, 256, stride=2, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        return self.classifier(x)


def build_model(name: str = 'ecgnet', num_classes: int = 4, seq_len: int = 205,
                dropout: float = None):
    """根据名称构建模型。name in {'ecgnet', 'resnet'}。"""
    name = name.lower()
    if name == 'ecgnet':
        return ECGNet(num_classes=num_classes, seq_len=seq_len,
                      dropout=0.1 if dropout is None else dropout)
    elif name == 'resnet':
        return HeartbeatResNet(num_classes=num_classes,
                               dropout=0.3 if dropout is None else dropout)
    raise ValueError(f"未知模型：{name}（可选 ecgnet / resnet）")


if __name__ == '__main__':
    for name in ['ecgnet', 'resnet']:
        model = build_model(name)
        dummy = torch.randn(8, 1, 205)
        out = model(dummy)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{name}] 输出形状 {tuple(out.shape)}，可训练参数 {params:,}")
