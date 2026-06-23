"""
实验一：心跳信号分类预测 —— 训练脚本

使用方式：
  python train.py                  # 默认 ECGNet
  python train.py --model resnet   # 改进版 ResNet
  python train.py --sampler        # 启用加权采样缓解类别不平衡

默认自动查找 ML_DATA_DIR、仓库 data/ 目录或本机课程 数据集/ 目录，并按实验指导书随机 8:2 划分 TrainSet/TestSet。

运行完成后生成：
  best_model.pth        最优模型权重
  split_indices.csv     TrainSet/TestSet 划分记录
  training_curve.png    训练曲线
  final_metrics.csv     TestSet 最终评测指标
  testset_submission_format.csv  内部 TestSet 的网站提交格式概率文件

补充：如果额外提供 testA.csv，可通过 --predict_csv 生成 submission.csv。
竞赛评分 = 预测概率与真实 one-hot 的「绝对差之和」的平均：
      score = mean_n( sum_k |onehot(y_n)_k - p_n_k| )   （越小越好）
"""

import random
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler

from dataset import HeartbeatDataset
from model import build_model

# ============================================================
# 超参数（可按实验结果调整）
# ============================================================
SEED         = 42
SCRIPT_DIR   = Path(__file__).resolve().parent


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


TRAIN_CSV    = resolve_data_file('train.csv')
TEST_CSV     = resolve_data_file('testA.csv')

TARGET_LEN   = 205
NUM_CLASSES  = 4
BATCH_SIZE   = 64
NUM_EPOCHS   = 20
LR           = 1e-3
WEIGHT_DECAY = 1e-4
TEST_RATIO   = 0.2
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 竞赛评分指标：abs-sum score
# ============================================================
def abs_sum_score(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 4) -> float:
    """sum_k |onehot - prob| 的样本平均，越小越好。"""
    onehot = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum(np.abs(onehot - y_prob), axis=1)))


def class_weights_from_labels(labels, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    counts = np.clip(counts.astype(np.float64), 1, None)
    w = counts.sum() / (num_classes * counts)
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


# ============================================================
# 训练 / 评测 / 预测
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    probs = np.concatenate(all_probs, 0)
    labels = np.concatenate(all_labels, 0)
    score = abs_sum_score(labels, probs, NUM_CLASSES)
    return total_loss / total, correct / total, score


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs = []
    for (x,) in loader:               # 测试集 __getitem__ 返回 (x,)
        x = x.to(device)
        probs = torch.softmax(model(x), dim=1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, 0)


@torch.no_grad()
def predict_with_labels(model, loader, device):
    """Return probabilities and labels for an internal labeled TestSet."""
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        probs = torch.softmax(model(x), dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_probs, 0), np.concatenate(all_labels, 0)


def save_probability_submission(ids, probs, save_path: str):
    """Save probabilities in the website submission format."""
    sub = pd.DataFrame(probs, columns=[f'label_{k}' for k in range(NUM_CLASSES)])
    sub.insert(0, 'id', ids)
    sub.to_csv(save_path, index=False, float_format='%.8f')
    print(f"概率结果已保存至 {save_path}，格式：id,label_0,label_1,label_2,label_3")


def save_split_file(dataset: HeartbeatDataset, train_idx: np.ndarray,
                    test_idx: np.ndarray, save_path: str = 'split_indices.csv'):
    split = np.full(len(dataset), 'TrainSet', dtype=object)
    split[test_idx] = 'TestSet'
    pd.DataFrame({
        'index': np.arange(len(dataset)),
        'id': dataset.ids,
        'label': dataset.labels,
        'split': split,
    }).to_csv(save_path, index=False)
    print(f"数据划分已保存至 {save_path}（TrainSet/TestSet=8:2）")


def plot_curves(tr_loss, te_loss, tr_acc, te_acc, show: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(tr_loss, label='Train Loss', color='royalblue')
    axes[0].plot(te_loss, label='Test Loss', color='tomato')
    axes[0].set_title('Loss 曲线'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(tr_acc, label='Train Acc', color='royalblue')
    axes[1].plot(te_acc, label='Test Acc', color='tomato')
    axes[1].set_title('Accuracy 曲线'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
    print("训练曲线已保存至 training_curve.png")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ecgnet',
                        choices=['ecgnet', 'resnet'], help='选择模型')
    parser.add_argument('--sampler', action='store_true',
                        help='启用 WeightedRandomSampler 缓解类别不平衡')
    parser.add_argument('--class_weight', action='store_true',
                        help='交叉熵使用类别权重缓解不平衡')
    parser.add_argument('--train_csv', type=str, default=str(TRAIN_CSV),
                        help='训练集 CSV 路径；默认自动查找 ML_DATA_DIR、../data、../../数据集')
    parser.add_argument('--predict_csv', '--test_csv', dest='predict_csv',
                        type=str, default=str(TEST_CSV),
                        help='可选竞赛测试集 CSV 路径；不存在时只做 TrainSet/TestSet 评测')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='批大小')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader 进程数；Windows/内存不足时建议 0')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='只读取前 N 条训练样本，便于快速调试；正式训练不要设置')
    parser.add_argument('--show_plots', action='store_true',
                        help='训练结束后弹出显示曲线图')
    args = parser.parse_args()

    set_seed(SEED)
    print(f"使用设备：{DEVICE}，模型：{args.model.upper()}")

    # ---------- 数据 ----------
    train_csv = Path(args.train_csv)
    predict_csv = Path(args.predict_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"找不到训练集：{train_csv}")
    has_predict = predict_csv.exists()
    if not has_predict:
        print(f"提示：未找到竞赛测试集 {predict_csv}，本次按指导书只做 8:2 TestSet 评测，不生成 submission.csv。")

    full = HeartbeatDataset(str(train_csv), target_len=TARGET_LEN, is_test=False,
                            max_samples=args.max_samples)
    if len(full) < 2:
        raise ValueError("训练样本数至少需要 2 条。")
    n_test = max(1, int(len(full) * TEST_RATIO))
    perm = np.random.RandomState(SEED).permutation(len(full))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    train_ds, test_split_ds = Subset(full, train_idx), Subset(full, test_idx)
    save_split_file(full, train_idx, test_idx)

    if args.sampler:
        sw = full.sample_weights()[train_idx]
        sampler = WeightedRandomSampler(torch.tensor(sw, dtype=torch.double),
                                        num_samples=len(sw), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=(DEVICE == 'cuda'))
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=(DEVICE == 'cuda'))
    test_split_loader = DataLoader(test_split_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=(DEVICE == 'cuda'))

    predict_ds, predict_loader = None, None
    if has_predict:
        predict_ds = HeartbeatDataset(str(predict_csv), target_len=TARGET_LEN, is_test=True)
        predict_loader = DataLoader(predict_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=(DEVICE == 'cuda'))
    print(f"TrainSet：{len(train_ds)}，TestSet：{len(test_split_ds)}，竞赛测试集：{len(predict_ds) if predict_ds is not None else 0}")
    print(f"全量标签分布：{full.class_counts().tolist()}")
    print(f"TrainSet 标签分布：{np.bincount(full.labels[train_idx], minlength=NUM_CLASSES).tolist()}")
    print(f"TestSet 标签分布：{np.bincount(full.labels[test_idx], minlength=NUM_CLASSES).tolist()}")

    # ---------- 模型 / 损失 / 优化器 ----------
    model = build_model(args.model, num_classes=NUM_CLASSES, seq_len=TARGET_LEN).to(DEVICE)
    if args.class_weight:
        cw = class_weights_from_labels(full.labels[train_idx], NUM_CLASSES).to(DEVICE)
        print(f"类别权重：{cw.cpu().numpy().round(3).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # 验证集损失不下降时自动降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5)

    # ---------- 训练循环 ----------
    best_score = float('inf')
    tr_losses, te_losses, tr_accs, te_accs = [], [], [], []
    print("\n开始训练...")
    for epoch in range(1, args.epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        te_loss, te_acc, te_score = evaluate(model, test_split_loader, criterion, DEVICE)
        scheduler.step(te_loss)

        tr_losses.append(tl); te_losses.append(te_loss)
        tr_accs.append(ta); te_accs.append(te_acc)

        flag = ''
        if te_score < best_score:        # abs-sum 越小越好
            best_score = te_score
            torch.save(model.state_dict(), 'best_model.pth')
            flag = '  已保存'
        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"Train Loss {tl:.4f} Acc {ta:.4f} | "
              f"Test Loss {te_loss:.4f} Acc {te_acc:.4f} AbsSum {te_score:.4f} | "
              f"LR {optimizer.param_groups[0]['lr']:.2e}{flag}")

    print(f"\n训练完成，最佳 TestSet abs-sum：{best_score:.4f}")

    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    final_loss, final_acc, final_score = evaluate(model, test_split_loader, criterion, DEVICE)
    test_probs, _ = predict_with_labels(model, test_split_loader, DEVICE)
    save_probability_submission(full.ids[test_idx], test_probs, 'testset_submission_format.csv')
    pd.DataFrame([{
        'model': args.model,
        'train_size': len(train_ds),
        'test_size': len(test_split_ds),
        'test_loss': final_loss,
        'test_accuracy': final_acc,
        'test_abs_sum': final_score,
    }]).to_csv('final_metrics.csv', index=False)
    print(f"最终 TestSet 指标：Loss={final_loss:.4f}, Acc={final_acc:.4f}, AbsSum={final_score:.4f}")
    print("最终指标已保存至 final_metrics.csv")

    plot_curves(tr_losses, te_losses, tr_accs, te_accs, show=args.show_plots)

    # ---------- 预测并生成「概率」提交文件 ----------
    if not has_predict:
        return
    probs = predict(model, predict_loader, DEVICE)        # (N, 4)
    save_probability_submission(predict_ds.ids, probs, 'submission.csv')
    print(f"提交文件已保存至 submission.csv，共 {len(predict_ds)} 条记录（4 列概率）")


if __name__ == '__main__':
    main()
