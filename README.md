# 机器学习课程实验

本仓库整理了两个机器学习课程实验的完整代码：

| 实验 | 任务 | 主要模型 | 数据文件 | 代码目录 |
|---|---|---|---|---|
| 实验一 | 心跳信号分类预测 | ECGNet 一维卷积神经网络 | `数据集/train.csv` | `exp1_heartbeat/` |
| 实验二 | 新闻文本分类 | `bert-base-chinese` 微调 | `数据集/train_set.csv` | `exp2_news/` |

建议最终上传 GitHub 时保持下面这种结构：

```text
.
├─ README.md
├─ requirements.txt
├─ requirements-bert.txt
├─ 数据集/
│  ├─ README.md
│  ├─ train.csv
│  └─ train_set.csv
├─ exp1_heartbeat/
│  ├─ dataset.py
│  ├─ model.py
│  ├─ train.py
│  └─ README.md
└─ exp2_news/
   ├─ dataset.py
   ├─ model.py
   ├─ train.py
   ├─ train_bert.py
   └─ README.md
```

注意：`train.csv` 和 `train_set.csv` 文件很大，普通 GitHub 仓库不能直接上传超过 100MB 的单个文件。如果必须把数据也放到 GitHub，请使用 Git LFS；如果只是提交代码仓库，建议只保留 `数据集/README.md`，真实数据集放在本地。

## 环境安装

进入项目根目录后安装依赖：

```powershell
pip install -r requirements.txt
pip install -r requirements-bert.txt
```

其中：

| 文件 | 作用 |
|---|---|
| `requirements.txt` | 两个实验的基础依赖，如 `numpy`、`pandas`、`torch`、`scikit-learn`、`matplotlib` |
| `requirements-bert.txt` | 实验二 BERT 路线额外依赖，如 `transformers` |

如果使用 GPU，建议先确认 CUDA 是否可用：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 数据集放置

本项目代码会自动查找数据文件，推荐放在仓库根目录的 `数据集/` 下：

```text
数据集/
├─ train.csv       # 实验一：心跳信号分类数据
└─ train_set.csv   # 实验二：新闻文本分类数据
```

也可以通过环境变量指定数据目录：

```powershell
$env:ML_DATA_DIR="D:\课程\研一下\机器学习\shiyan\数据集"
```

两个实验都会按照实验指导书要求，在给定训练集内部随机划分 `TrainSet` 和 `TestSet`，比例为 8:2。

## 实验一：心跳信号分类

进入实验一目录：

```powershell
cd exp1_heartbeat
```

先运行数据分析脚本：

```powershell
python dataset.py
```

该脚本会读取 `train.csv`，统计样本数量、类别分布、信号长度，并生成：

```text
eda_heartbeat.png
```

快速测试代码是否能跑通：

```powershell
python train.py --model ecgnet --epochs 1 --max_samples 256
```

正式训练：

```powershell
python train.py --model ecgnet --epochs 20
```

训练完成后会生成：

| 文件 | 说明 |
|---|---|
| `best_model.pth` | 最优 ECGNet 模型权重 |
| `split_indices.csv` | 8:2 划分后的 TrainSet/TestSet 索引 |
| `final_metrics.csv` | 最终 TestSet 指标 |
| `training_curve.png` | 训练过程曲线 |
| `testset_submission_format.csv` | 内部 TestSet 的概率输出格式 |

当前已跑出的参考结果：

| 指标 | 结果 |
|---|---:|
| TestSet Loss | 0.0481 |
| TestSet Accuracy | 0.9948 |
| TestSet abs-sum | 0.0109 |

其中 `testset_submission_format.csv` 的列格式为：

```csv
id,label_0,label_1,label_2,label_3
```

该文件用于说明心跳信号任务的概率输出格式，不是线上提交文件。

## 实验二：新闻文本分类

进入实验二目录：

```powershell
cd exp2_news
```

先运行数据分析脚本：

```powershell
python dataset.py
```

该脚本会读取 `train_set.csv`，统计 14 个新闻类别的样本分布和文本长度，并生成：

```text
eda_news.png
```

快速测试 BERT 流程是否能跑通：

```powershell
python train_bert.py --epochs 1 --max_samples 512 --batch_size 8 --max_len 64
```

正式训练推荐命令：

```powershell
python train_bert.py --bert_path bert-base-chinese --epochs 30 --batch_size 16 --max_len 128 --balance_strategy sampler
```

参数说明：

| 参数 | 含义 |
|---|---|
| `--bert_path bert-base-chinese` | 使用中文预训练 BERT |
| `--epochs 30` | 最多训练 30 轮，代码中带早停 |
| `--batch_size 16` | 适合 8GB 显存的批大小 |
| `--max_len 128` | BERT 输入最大长度 |
| `--balance_strategy sampler` | 使用加权采样处理类别不均衡 |

如果显存不足，可以把 `batch_size` 改成 8：

```powershell
python train_bert.py --bert_path bert-base-chinese --epochs 30 --batch_size 8 --max_len 128 --balance_strategy sampler
```

训练完成后会生成：

| 文件 | 说明 |
|---|---|
| `best_bert_model.pth` | 最优 BERT 模型权重 |
| `split_indices_bert.csv` | 8:2 划分后的 TrainSet/TestSet 索引 |
| `final_metrics_bert.csv` | 最终 TestSet 指标 |
| `final_metrics_bert_history.csv` | 多次训练的历史指标记录 |
| `classification_report_bert.txt` | 14 类 precision、recall、F1-score |
| `test_predictions_bert.csv` | TestSet 预测结果 |
| `training_loss.png` | BERT 训练损失曲线 |

当前已跑出的正式 BERT 结果：

| 指标 | 结果 |
|---|---:|
| TrainSet | 160000 |
| TestSet | 40000 |
| Batch Size | 16 |
| Max Length | 128 |
| Actual Epochs | 16 |
| Best Epoch | 11 |
| TestSet Loss | 0.2630 |
| TestSet Accuracy | 0.9332 |
| TestSet Macro-F1 | 0.9222 |

训练开始时如果看到类似 `UNEXPECTED` 的提示，一般是因为加载预训练 BERT 时忽略了预训练任务头，属于正常现象，不影响文本分类微调。

## GitHub 上传建议

建议上传：

```text
README.md
requirements.txt
requirements-bert.txt
数据集/README.md
exp1_heartbeat/
exp2_news/
```

不建议上传：

```text
数据集/train.csv
数据集/train_set.csv
*.pth
*.pt
split_indices*.csv
final_metrics*.csv
classification_report*.txt
test_predictions*.csv
training_curve.png
training_loss.png
eda_*.png
__pycache__/
```

如果老师明确要求 GitHub 里包含真实数据集，需要使用 Git LFS：

```powershell
git lfs install
git lfs track "数据集/*.csv"
git add .gitattributes 数据集/train.csv 数据集/train_set.csv
```

否则普通 `git push` 会因为单个 CSV 文件超过 100MB 而失败。

## 最短复现实验流程

实验一：

```powershell
cd exp1_heartbeat
python dataset.py
python train.py --model ecgnet --epochs 20
```

实验二：

```powershell
cd exp2_news
python dataset.py
python train_bert.py --bert_path bert-base-chinese --epochs 30 --batch_size 16 --max_len 128 --balance_strategy sampler
```

跑完后，实验一主要查看 `final_metrics.csv` 和 `training_curve.png`；实验二主要查看 `final_metrics_bert.csv`、`classification_report_bert.txt` 和 `training_loss.png`。
