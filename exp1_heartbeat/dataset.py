"""
实验一：心跳信号分类预测
数据集加载与预处理模块

数据来源：天池竞赛 531883（心跳信号分类预测）
格式：
  train.csv  -> 列：id, heartbeat_signals, label
  testA.csv  -> 列：id, heartbeat_signals
  heartbeat_signals：**以逗号分隔**的浮点数序列，每条长度固定为 205
  label：0/1/2/3（4 分类）

重要说明（依据竞赛真实数据）：
  1) 信号字段是「逗号」分隔，不是空格。解析时按逗号切分。
  2) 训练集类别极度不平衡（约 0:64327 / 1:3562 / 2:14199 / 3:17912），
     因此提供 class_weights 与 WeightedRandomSampler 两种缓解手段。
  3) 竞赛评分采用「预测概率与真实 one-hot 的绝对差之和」，
     因此提交文件需要输出 4 列概率（见 train.py）。
"""

import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser
import os
from pathlib import Path
from torch.utils.data import Dataset

TARGET_LEN_DEFAULT = 205   # 竞赛信号固定长度
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_data_file(filename: str) -> Path:
    """Find data in GitHub-style layout first, while keeping the local course layout usable."""
    candidates = []
    env_dir = os.environ.get('ML_DATA_DIR')
    if env_dir:
        candidates.append(Path(env_dir) / filename)
    candidates.extend([
        SCRIPT_DIR.parent / 'data' / filename,
        SCRIPT_DIR.parent / '数据集' / filename,
        SCRIPT_DIR.parents[1] / 'data' / filename,
        SCRIPT_DIR.parents[1] / '数据集' / filename,
    ])
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else SCRIPT_DIR.parent / 'data' / filename

# ============================================================
# 工具函数
# ============================================================

def parse_signals(signal_str: str, target_len: int = TARGET_LEN_DEFAULT) -> np.ndarray:
    """
    将字符串信号解析为定长 float32 数组。

    兼容两种分隔符：优先按逗号切分（竞赛真实格式），
    若没有逗号则退化为按空白切分，增强健壮性。
    """
    s = signal_str.strip()
    if ',' in s:
        parts = s.split(',')
    else:
        parts = s.split()
    arr = np.array([float(x) for x in parts if x != ''], dtype=np.float32)

    if len(arr) >= target_len:
        arr = arr[:target_len]
    else:
        arr = np.pad(arr, (0, target_len - len(arr)),
                     mode='constant', constant_values=0.0)
    return arr


def normalize_signals(arr: np.ndarray) -> np.ndarray:
    """Min-Max 归一化到 [0, 1]，加速模型收敛。"""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return arr
    return (arr - mn) / (mx - mn)


# ============================================================
# Dataset
# ============================================================

class HeartbeatDataset(Dataset):
    """心跳信号分类数据集。"""

    def __init__(self, csv_path: str, target_len: int = TARGET_LEN_DEFAULT,
                 is_test: bool = False, normalize: bool = True,
                 max_samples: int = None):
        """
        Args:
            csv_path:   CSV 文件路径
            target_len: 信号统一长度（填充/截断）
            is_test:    True 时不读取 label 列
            normalize:  是否对每条信号做 Min-Max 归一化
            max_samples: 只读取前 N 条，便于快速调试
        """
        df = pd.read_csv(csv_path, nrows=max_samples)
        self.is_test = is_test
        self.ids = df['id'].values

        signals = []
        for s in df['heartbeat_signals']:
            arr = parse_signals(s, target_len)
            if normalize:
                arr = normalize_signals(arr)
            signals.append(arr)
        self.signals = np.stack(signals, axis=0)  # (N, target_len)

        if not is_test:
            self.labels = df['label'].astype(int).values
        else:
            self.labels = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 增加 channel 维度 -> (1, target_len)，适配 Conv1d
        x = torch.tensor(self.signals[idx], dtype=torch.float32).unsqueeze(0)
        if not self.is_test:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return (x,)

    # -------- 类别不平衡相关 --------
    def class_counts(self, num_classes: int = 4) -> np.ndarray:
        """返回各类别样本数，用于计算损失权重。"""
        counts = np.zeros(num_classes, dtype=np.int64)
        for c in self.labels:
            counts[c] += 1
        return counts

    def class_weights(self, num_classes: int = 4) -> torch.Tensor:
        """
        计算交叉熵损失的类别权重：
        权重 = (总样本数) / (类别数 * 该类样本数)，并归一化。
        小类别获得更大权重，缓解类别不平衡。
        """
        counts = self.class_counts(num_classes).astype(np.float64)
        counts = np.clip(counts, 1, None)
        w = counts.sum() / (num_classes * counts)
        w = w / w.sum() * num_classes
        return torch.tensor(w, dtype=torch.float32)

    def sample_weights(self) -> np.ndarray:
        """
        计算每个样本的采样权重（供 WeightedRandomSampler 使用）：
        样本权重 = 1 / 该类别总样本数，小类别样本更易被采到。
        """
        counts = self.class_counts().astype(np.float64)
        counts = np.clip(counts, 1, None)
        return np.array([1.0 / counts[c] for c in self.labels], dtype=np.float64)


# ============================================================
# 数据分析（EDA）辅助函数
# ============================================================

def eda(csv_path: str, save_path: str = 'eda_heartbeat.png', show: bool = False):
    """打印数据基本统计与类别分布，并绘图保存。"""
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    print(f"样本总数：{len(df)}")
    print(f"列名：{df.columns.tolist()}")
    print(f"缺失值：\n{df.isna().sum()}")

    if 'label' in df.columns:
        print("\n类别分布：")
        vc = df['label'].value_counts().sort_index()
        for k, v in vc.items():
            print(f"  类别 {k}: {v}  ({v / len(df) * 100:.2f}%)")

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        bars = vc.plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title('训练集各类别样本数', fontsize=13)
        axes[0].set_xlabel('类别')
        axes[0].set_ylabel('样本数')
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)
        for p in axes[0].patches:
            axes[0].annotate(f'{int(p.get_height())}',
                             (p.get_x() + p.get_width() / 2, p.get_height()),
                             ha='center', va='bottom', fontsize=9)

        # 每类绘制一条样例心跳曲线
        for label in sorted(df['label'].unique()):
            sample = df[df['label'] == label].iloc[0]['heartbeat_signals']
            arr = parse_signals(sample)
            axes[1].plot(arr, label=f'类别 {label}', alpha=0.8)
        axes[1].set_title('各类别样例心跳信号', fontsize=13)
        axes[1].set_xlabel('时间步')
        axes[1].set_ylabel('幅值')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nEDA 图已保存至 {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # 信号长度统计
    def _len(s):
        s = s.strip()
        return len(s.split(',')) if ',' in s else len(s.split())
    lengths = df['heartbeat_signals'].apply(_len)
    print(f"\n信号长度统计：min={lengths.min()}, max={lengths.max()}, "
          f"mean={lengths.mean():.1f}, std={lengths.std():.1f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv', default=str(resolve_data_file('train.csv')),
                        help='训练集 CSV 路径；默认自动查找 ML_DATA_DIR、../data、../../数据集')
    parser.add_argument('--save', default='eda_heartbeat.png',
                        help='EDA 图片保存路径')
    parser.add_argument('--show', action='store_true',
                        help='生成图片后弹出显示')
    args = parser.parse_args()
    eda(args.csv, args.save, show=args.show)
