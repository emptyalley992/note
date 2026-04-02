


> Quant

[TOC]

## IV
这份论文介绍了一个名为 **DeepVol** 的深度学习模型，旨在利用**高频（High-Frequency, HF）盘中数据**来预测**次日的波动率**。

以下是对该论文任务要求的详细解读：

---

### 任务一：论文总结与创新点

**1. 总结：**
DeepVol 放弃了传统金融计量模型中“先将高频数据聚合为单个标量（如已实现波动率 RV），再进行建模”的步骤。它直接以原始的盘中收益率序列作为输入，利用**膨胀因果卷积（Dilated Causal Convolutions, DCC）**网络自动提取特征，从而更有效地捕捉高频数据中的预测信息，并能更好地处理微观结构噪声。

**2. 创新点与工作重心：**
*   **端到端建模：** 直接从原始高频数据预测日波动率，避免了预计算“已实现指标（Realised Measures）”造成的信息损失。
*   **引入 DCC 架构：** 首次将用于音频生成的膨胀因果卷积引入波动率预测，解决了长序列处理中参数爆炸的问题。
*   **注意力机制集成：** 结合了注意力机制，让模型能动态识别盘中哪些时间点对次日波动率影响更大。
*   **强大的泛化能力：** 证明了模型具有“迁移学习”能力，即在部分股票上训练后，能直接应用于从未见过的股票，且在金融危机（如 COVID-19）期间表现稳健。

---

### 任务二：技术背景与原理

**1. 波动率的不可观测性（Latent Variable）：**
波动率在金融中是“隐变量”，我们无法直接看到，只能通过代理变量（如日收益率的平方）来估算。

**2. 传统方法的局限：**
*   **GARCH 系列：** 主要使用每日的收盘价，忽略了盘中剧烈的价格变动。
*   **HEAVY/HAR 模型：** 虽然使用了高频数据，但通常是将其聚合为日度标量（如 5 分钟 RV）。这种做法假设了某种人工设计的计算公式是完美的，但实际上会损失盘中动态信息，且容易受到“微观结构噪声”（如买卖价差导致的频繁跳动）的影响。

**3. 膨胀因果卷积（DCC）原理：**
*   **因果性（Causal）：** 保证预测 $t$ 时刻的数据只使用 $t$ 之前的信息，防止“未来预测过去”。
*   **膨胀（Dilated）：** 普通卷积核覆盖范围小，要看长序列需要非常深的网络。DCC 通过跳过中间点的方式（跳格卷积），让感受野随深度呈指数级增长，从而用很少的层数就能覆盖一整天甚至多天的高频观测点。

---

### 任务三：DeepVol 方法及原理（核心细节）

DeepVol 的目标是学习一个函数 $f_\theta$，输入过去 $T$ 天的盘中高频收益率，输出第 $T+1$ 天的已实现波动率 $\hat{\sigma}^2_{T+1}$。

#### 1. 模型输入
*   **输入内容：** 原始盘中收益率序列 $r_{i,t}$。
*   **维度变化：** 假设使用 5 分钟采样频率，每个交易日（6.5 小时）有 $J=78$ 个观测点。如果模型查看过去 $T$ 天（Receptive Field），输入就是一个长度为 $T \times J$ 的一维向量。
*   **预处理：** 仅进行对数收益率计算：$r_{i,t} = \log(p_{i,t} / p_{i-1,t})$，保持数据的原始动态。

#### 2. 重要模块处理
*   **膨胀因果卷积层（DCC Layers）：**
    这是最核心的模块。其数学表达式为：
    $$F^{(l)}(t) = \sum_{\tau=0}^{s-1} k_\tau^{(l)} \cdot xt-d\tau$$
    *   **$x$**：输入序列。
    *   **$k_\tau^{(l)}$**：第 $l$ 层的卷积核（Filter）。
    *   **$s$**：卷积核大小（论文中设为 3）。
    *   **$d$**：膨胀因子（Dilation factor）。在第一层 $d=1$（普通卷积），在后续层 $d$ 通常按 2 的指数级增加（1, 2, 4, 8...）。
    *   **数据流变化：** 通过增大 $d$，每一层输出的特征图（Feature Map）虽然维度在减小，但每个点包含的时间跨度（感受野）在迅速扩大。
*   **残差连接（Residual Connections）：**
    每层卷积后都会将原始输入加回输出中。这解决了深层网络的梯度消失问题，允许模型堆叠更多层（论文使用了 6 层）来捕捉更复杂的非线性关系。
*   **注意力机制（Attention Mechanism）：**
    模型会对不同层、不同时间步的特征进行加权。
    *   **作用：** 自动学习出比如“开盘”和“收盘”前的波动是否对明天更有指导意义。
*   **输出层（Output Layer）：**
    经过多层处理后的特征流向一个全连接层，并配合 **ReLU 激活函数**。
    *   **公式：** $\sigma^2_{T+1} = \alpha_0 + \sum_{l=1}^L \alpha_l \sigma_{ReLu}(F^{(l)})$
    *   **维度变化：** 将复杂的特征向量最终映射为一个正数标量，即次日的波动率预测值。

#### 3. 损失函数（QLIKE）
模型优化使用的是 **QLIKE** 函数，公式为：
$$\ell_{QLIKE} = \log(\hat{\sigma}^2_t) + \frac{\sigma^2_t}{\hat{\sigma}^2_t}$$
*   **原理：** 在波动率预测中，QLIKE 比常见的 MSE（均方误差）更稳健。它对低估波动的惩罚非常重，且不会因为观测代理（Proxy）的选择偏误而导致预测失效。

---

### 任务四：实验部分介绍

**1. 数据集：**
*   **来源：** NASDAQ-100 指数的成分股。
*   **跨度：** 2019 年 9 月 30 日至 2021 年 9 月 30 日（共 **2 年**）。
*   **划分：** 前 12 个月训练，中间 3 个月验证，最后 9 个月测试。
*   **采样频率：** 实验对比了 1, 5, 15, 30, 60 分钟不同的采样粒度。最终发现 **5 分钟** 效果最好。

**2. 对比基线（Baselines）：**
论文对比了 16 种模型，分为三大类：
*   **传统日度模型：** GARCH, TARCH, EGARCH, IGARCH 等（只用日数据）。
*   **已实现测度模型：** HEAVY, HAR, Realized-GARCH（用预计算的 RV 标量）。
*   **经典深度学习：** MLP, LSTM（直接用高频数据，但非 DCC 架构）。
*   **基准模型：** Martingale（假设明天波动率等于今天）。

**3. 评价指标：**
*   **MAE / RMSE：** 衡量预测值与实际值的绝对/平方偏差。
*   **QLIKE：** 核心损失指标，衡量预测的稳健性。
*   **SMAPE：** 对称平均绝对百分比误差。
*   **MedAE：** 中值绝对误差（衡量对离群值的鲁棒性）。
*   **MCS（Model Confidence Set）：** 统计学检验，用来判断 DeepVol 是否在统计意义上显著优于其他所有模型。

**4. 实验结论：**
*   DeepVol 在几乎所有指标（MAE, RMSE, QLIKE）上都显著优于所有传统模型和 LSTM。
*   在“跨股票泛化”实验中，DeepVol 表现出极强的适应力，证明了它学到了金融市场通用的波动特征，而不仅仅是某只股票的特定规律。



## Option

### OPHR: Mastering Volatility Trading with Multi-Agent Deep Reinforcement Learning

#### 波动率交易
由于期权的复杂性，波动性交易策略有很多种。其中之一是多头/空头 Gamma 策略，旨在从 IV（市场定价的期权波动率）和 RV（标的资产实际价格波动）之间的差异中获利。

该论文用跨式期权组合，确保Δ为0，然后持有标的资产（对应的永续期货）进行对冲；








### 任务一：论文总结、创新点与工作重心

**总结：**
这篇论文研究的是金融领域中极其复杂的**期权波动率交易（Volatility Trading）**。波动率交易的核心逻辑在于利用“隐含波动率（IV，市场预测的）”与“实现波动率（RV，现实发生的）”之间的差异获利。论文提出了一个名为 **OPHR** 的强化学习框架，通过两个专门的智能体协作：一个负责“择时”（买卖什么），一个负责“对冲”（如何控制风险），从而在数字货币（BTC/ETH）市场实现了远超传统模型和单体机器学习算法的效果。

**创新点：**
1. **首个专门针对波动率交易的 RL 框架：** 之前的研究多集中在期权定价或单纯的风险对冲，而 OPHR 首次完整解决了从开仓择时到动态对冲的全流程。
2. **多智能体分工架构（OP + HR）：** 将复杂任务拆解为期权头寸管理（OP-Agent）和对冲路由管理（HR-Agent），降低了单个智能体的学习难度。
3. **两阶段训练法：** 引入了“先知策略（Oracle Policy）”蒸馏技术，利用未来的真实信息在离线阶段指导模型，解决强化学习早期探索效率低的问题。

**工作重心：**
如何通过**精准的择时**（在波动爆发前买入 Gamma，在市场平稳时卖出 Gamma）以及**高效的对冲**（在减少对冲成本的同时，最大限度锁住波动率收益）来最大化风险调整后的收益。

---

### 任务二：技术背景与基本原理

期权波动率交易依赖于经典的 **Black-Scholes-Merton (BSM) 模型**。

**1. 基本理论公式（BSM 偏微分方程）：**
$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0$$
*   **含义：** 该方程描述了期权价格 $V$ 随时间 $t$、标的价格 $S$ 和波动率 $\sigma$ 的变化关系。

**2. 核心技术指标：希腊字母（Greeks）**
在 OPHR 的模型输入中，Greeks 是最重要的特征：
*   **Delta ($\Delta = \frac{\partial V}{\partial S}$):** 期权价格对标的价格变化的敏感度。对冲的目标通常是让 $\Delta \approx 0$。
*   **Gamma ($\Gamma = \frac{\partial^2 V}{\partial S^2}$):** Delta 对标的价格变化的敏感度，衡量“凸性”。**波动率交易本质上是在交易 Gamma。**
*   **Theta ($\Theta = \frac{\partial V}{\partial t}$):** 时间流逝带来的价值损失。买入 Gamma 的代价就是付出 Theta。
*   **Vega ($\nu = \frac{\partial V}{\partial \sigma}$):** 期权价格对波动率变化的敏感度。

**3. 波动率交易获利的理论基础：**
当进行 Delta 对冲后，期权的盈亏（PnL）近似为：
$$PnL \approx \int_{0}^{T} \frac{S_t^2}{2} \Gamma (\sigma_t^2 - \sigma^2) dt$$
*   **公式意义：** 利润取决于**真实发生的波动率 $\sigma_t$** 与**期权定价里的隐含波动率 $\sigma$** 的平方差。如果现实比预测更剧烈，做多波动率（Long Gamma）就赚钱。

---

### 任务三：提出的方法（OPHR）及原理

OPHR 框架由两个核心智能体和一个两阶段训练流程组成。

#### 1. 模块化架构

*   **OP-Agent（期权头寸智能体）：**
    *   **职责：** 决定“何时出手”。
    *   **输入：** 市场特征 $F_t$（包括价格动量、历史波动率、隐含波动率曲面等）。
    *   **动作：** $\{+1（做多波动率）, -1（做空波动率）, 0（观望）\}$。通常操作的是“跨式组合（Straddle）”，因为这种组合初始 Delta 为 0，纯粹赌波动。
    *   **输出：** 目标期权仓位。

*   **HR-Agent（对冲路由智能体）：**
    *   **职责：** 决定“怎么防守”。
    *   **输入：** 状态 $s_t^{hr} = (F_t, P_t, G_t)$。其中 $P_t$ 是当前持仓信息，$G_t$ 是实时计算的 Greeks。
    *   **动作：** 从一个“对冲策略池”中选择最适合当前环境的策略（如基于阈值的对冲、基于价格变化的对冲或深度强化学习对冲器）。
    *   **输出：** 具体的底层资产买卖数量，以抵消 Delta 风险。
    * 每N步做出一次决策，选出的对冲策略每步执行对冲
  
*   **Hedgers（对冲策略池）：**
	* 一组具有不同**风险厌恶水平**的对冲(套期保值)策略 $\{\pi^{hedger}_{i}\}^K_{i=1}$



#### 2. 特征与数据流变化
1.  **特征提取层：** 将小时级的标的价格、IV 数据输入。特征维度从原始序列转化为高维特征向量。
2.  **决策层（OP）：** 输入特征向量 $\to$ 多层感知机（MLP） $\to$ 输出三个动作的概率（维度：3）。
3.  **协同层（HR）：** OP 确定的仓位会改变账户的实时 Greeks（如 $\Delta$ 暴露）。HR 获取更新后的 Greeks $\to$ 决策对冲策略。
4.  **反馈环：** 环境返回仓位盈亏和对冲成本。**Reward（奖励函数）** 定义为：$r = \Delta NetValue - \text{对冲成本}$。

#### 3. 核心训练原理（n-step TD Learning）
为了解决期权收益延迟（Theta 每天扣除，Gamma 收益需等待价格波动）的问题，论文采用了 **n-步时间差分学习**（公式 5）：
$$L(\theta) = \mathbb{E} \left[ \left( \sum_{k=0}^{n-1} \gamma^k r_j^{(k)} + \gamma^n Q(s^{(n)}, \arg\max Q) - Q(s, a) \right)^2 \right]$$
*   **原理：** 它不再只看下一秒的奖励，而是累积未来 $n$ 步的奖励再进行更新。这对于“持有期权等待波动爆发”的交易逻辑至关重要。

#### 4. 两阶段训练过程
*   **Phase I (Oracle Distillation)：** 引入一个“先知策略”，它能看到未来的真实波动率。先让 OP-Agent 在离线状态下模仿这个“先知”，快速学会基本的择时逻辑。
*   **Phase II (Iterative Training)：** 固定 OP 训练 HR，再固定 HR 训练 OP。这种博弈式的交替迭代，让两个智能体达到最佳配合。

---

### 任务四：实验部分简述

**1. 数据集：**
*   **来源：** Deribit（全球最大的加密货币期权交易所）。
*   **对象：** **BTC** 和 **ETH** 的期权及永续合约数据。
*   **时间跨度：** **2019年4月1日 至 2024年7月1日**（约5年多）。
    *   **训练集：** 2019/04 - 2022/12（涵盖牛熊转换）。
    *   **验证集：** 2023/01 - 2023/06。
    *   **测试集：** 2023/07 - 2024/07。
*   **粒度：** 逐小时（Hourly）数据。

**2. 评价指标（8大指标）：**
*   **盈利能力：** 总收益率 (TR)。
*   **风险调整收益：** 夏普比率 (ASR)、卡玛比率 (ACR)、索提诺比率 (ASoR)。
*   **风险控制：** 年化波动率 (AVOL)、最大回撤 (MDD)。
*   **交易胜率：** 胜率 (WR)、盈亏比 (PLR)。

**3. 对比基线（Baselines）：**
*   **传统策略：** 纯做多 (Long)、纯做空 (Short)、均值回归 (MR)、动量策略 (MOM)。
*   **传统模型：** GARCH（经典的波动率预测模型）。
*   **机器学习/深度学习：** GBDT、MLP、LSTM、DeepVol。
*   **端到端 RL 基线：** DLOT（之前的期权组合管理 SOTA 方法）。

**实验结论：**
OPHR 在所有关键指标上均显著超过上述基线。特别是在测试期内，其 **Total Return (TR)** 在 BTC 上达到 33.10%，ETH 上达到 44.89%，而传统的 GARCH 和简单的 LSTM 模型在复杂的期权环境下大多表现为亏损。





# tmp

这是一个非常典型的量化金融机器学习任务（量价因子挖掘与截面收益预测）。针对您的需求，我将整个方案分为三个部分：**模型迭代架构深度思考**、**核心代码框架**（包含数据处理、解耦的模型定义、带IC和早停的训练流程）。

### 一、 深度思考：截面收益预测模型迭代路线

在量化中，预测截面收益率不仅要看单资产的时间序列规律，还要看同一时刻多资产之间的横向关系。针对此任务（最多8只债券ETF的日频数据），我为您构想了以下**由简到繁的四代迭代路线**：

1.  **V1 - 纯时序特征提取（当前基线）：GRU / LSTM**
    *   **思路**：将每只ETF视为独立的时间序列，使用历史 $N$ 天的特征预测未来的收益。
    *   **优点**：实现简单，能捕捉单只资产的动量、反转等时序规律。
    *   **缺点**：忽略了同一天不同ETF之间的联动关系和市场整体的宏观状态。
2.  **V2 - 时序特征 + 注意力机制：GRU + Temporal Attention / ALSTM**
    *   **思路**：在GRU的基础上，增加对历史不同时间步的Attention权重（有些历史日期的信息更关键）。
    *   **改进**：模型不仅知道历史趋势，还能“聚焦”重要事件日。
3.  **V3 - 时空联合建模（加入截面交互）：GRU + Cross-sectional Transformer / Self-Attention**
    *   **思路**：第一步用GRU分别提取当天 $N$ 只ETF的时序向量；第二步，将这 $N$ 个向量输入到一个Transformer Encoder中进行横向（截面）信息交互。
    *   **改进**：由于数据中有时ETF不足8只，Transformer的Attention机制天生能处理变长序列，能完美解决同一天资产数量不固定的问题，实现真正的“截面预测”。
4.  **V4 - 引入图谱交互：时序图神经网络 (T-GNN / GAT)**
    *   **思路**：基于ETF的底层资产相似度、久期或交易量相关性构建动态图，使用Graph Attention Network (GAT) 聚合信息。
    *   **改进**：最高阶解法，显式地利用金融逻辑来约束深度学习模型的截面信息流动。

---

### 二、 完整 Python 代码流程

为了方便迭代，代码高度解耦。只需修改 `model = XXX` 即可无缝切换模型。

#### 1. 导入依赖与设置
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子保证复现
torch.manual_seed(42)
np.random.seed(42)
```

#### 2. 数据预处理与特征工程
```python
def preprocess_data(csv_path):
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 时间格式处理与排序 (量化任务必须严格按时间排序)
    df['trade_dt'] = pd.to_datetime(df['trade_dt'].astype(str))
    df = df.sort_values(by=['instrument', 'trade_dt']).reset_index(drop=True)
    
    # 选取所需特征列 (这里选取一部分作为示范)
    feature_cols = ['s_dq_pctchange', 's_dq_volume', 's_dq_amount', 
                    's_dq_adjopen', 's_dq_adjhigh', 's_dq_adjlow', 's_dq_adjclose', 
                    'trades_count', 'discount_rate']
    
    # 缺失值填充 (简单的向前填充)
    df[feature_cols] = df.groupby('instrument')[feature_cols].ffill().fillna(0)
    
    # ---------------- 标签构建 (重点) ----------------
    # 1. 计算真实的次日收益率：(明天复权收盘价 / 今天复权收盘价) - 1
    df['next_return'] = df.groupby('instrument')['s_dq_adjclose'].shift(-1) / df['s_dq_adjclose'] - 1
    
    # 2. 截面收益率标签化 (按日期做 Z-score)
    # 因为存在不足8只的情况，当某天只有1只时无法做截面处理，直接赋予0或剔除
    def calc_cs_return(group):
        if len(group) > 1:
            return (group['next_return'] - group['next_return'].mean()) / (group['next_return'].std() + 1e-8)
        else:
            return pd.Series(0.0, index=group.index)
            
    df['target'] = df.groupby('trade_dt').apply(calc_cs_return).reset_index(level=0, drop=True)
    
    # 剔除因为shift产生的最后一天NaN
    df = df.dropna(subset=['target'])
    
    # 特征标准化 (时序标准化，避免使用未来数据，此处简单使用整体标准化，实盘需用滚动标准化)
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
    return df, feature_cols

```

#### 3. 构建 PyTorch Dataset (滑动窗口)
```python
class ETFTimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len=10):
        self.seq_len = seq_len
        self.X = []
        self.y = []
        self.dates = []  # 用于后续计算IC
        
        # 按股票分组构建滑动窗口
        for inst, group in df.groupby('instrument'):
            group = group.sort_values('trade_dt').reset_index(drop=True)
            values = group[feature_cols].values
            targets = group['target'].values
            dates = group['trade_dt'].values
            
            for i in range(len(group) - seq_len):
                self.X.append(values[i : i + seq_len])
                self.y.append(targets[i + seq_len - 1]) # 使用窗口最后一天预测明天
                self.dates.append(dates[i + seq_len - 1])
                
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(-1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.dates[idx]
```

#### 4. 模型定义 (高度解耦)
```python
# 基线模型：GRU
class GRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(GRUPredictor, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, 
                          hidden_size=hidden_dim, 
                          num_layers=num_layers, 
                          batch_first=True, 
                          dropout=dropout)
        
        # 全连接层映射到单一收益率预测值
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        # 取时间序列的最后一步作为表征
        last_out = out[:, -1, :] 
        # 预测截面收益率
        pred = self.fc(last_out)
        return pred

# 如果未来要尝试其他模型，直接在这里定义，例如：
# class TransformerPredictor(nn.Module): ...
# class ALSTMPredictor(nn.Module): ...
```

#### 5. 训练与验证逻辑 (带 Rank IC 和 早停)
```python
def calculate_ic(preds, targets, dates):
    """计算按日期分组的截面 Rank IC"""
    df_eval = pd.DataFrame({'date': dates, 'pred': preds.flatten(), 'target': targets.flatten()})
    ic_list = []
    
    for date, group in df_eval.groupby('date'):
        if len(group) > 1: # 至少有两个资产才能计算Rank IC
            ic, _ = spearmanr(group['pred'], group['target'])
            if not np.isnan(ic):
                ic_list.append(ic)
                
    return np.mean(ic_list) if ic_list else 0.0

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=5, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_ic = -np.inf
    patience_counter = 0
    best_model_state = None
    
    model.to(device)
    
    for epoch in range(epochs):
        # ---------------- Training ----------------
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # ---------------- Validation ----------------
        model.eval()
        val_preds, val_targets, val_dates = [], [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch, dates in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
                val_dates.extend(dates) # dates 是 tuple
                
        val_loss /= len(val_loader)
        val_ic = calculate_ic(np.array(val_preds), np.array(val_targets), np.array(val_dates))
        
        print(f"Epoch {epoch+1}/{epochs} - Train MSE: {train_loss:.4f} - Val MSE: {val_loss:.4f} - Val IC: {val_ic:.4f}")
        
        # ---------------- Early Stopping (根据 IC 监控) ----------------
        if val_ic > best_val_ic:
            best_val_ic = val_ic
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break
            
    # 加载最优模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model
```

#### 6. 主流程组合 (划分数据集与执行)
```python
def main():
    # 1. 准备数据
    csv_path = 'sample.csv' # 替换为您的实际路径
    df, feature_cols = preprocess_data(csv_path)
    
    # 2. 时序划分数据集
    # 根据题意数据到2026年，这里假设按时间切割：训练(前70%) / 验证(10%) / 测试(20%)
    dates_sorted = np.sort(df['trade_dt'].unique())
    n_dates = len(dates_sorted)
    
    train_end = dates_sorted[int(n_dates * 0.7)]
    val_end = dates_sorted[int(n_dates * 0.8)]
    
    train_df = df[df['trade_dt'] < train_end]
    val_df = df[(df['trade_dt'] >= train_end) & (df['trade_dt'] < val_end)]
    test_df = df[df['trade_dt'] >= val_end]
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # 3. 建立 Dataset & DataLoader
    seq_len = 10 # 滑动窗口天数
    train_dataset = ETFTimeSeriesDataset(train_df, feature_cols, seq_len=seq_len)
    val_dataset = ETFTimeSeriesDataset(val_df, feature_cols, seq_len=seq_len)
    test_dataset = ETFTimeSeriesDataset(test_df, feature_cols, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 4. 初始化模型 (解耦处，后续只需改此处即可)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = len(feature_cols)
    
    model = GRUPredictor(input_dim=input_dim, hidden_dim=64, num_layers=2)
    # model = TransformerPredictor(...) # 未来替换点
    
    # 5. 训练模型
    print("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=5, device=device)
    
    # 6. 在测试集上最终评估
    trained_model.eval()
    test_preds, test_targets, test_dates = [], [], []
    with torch.no_grad():
        for X_batch, y_batch, dates in test_loader:
            preds = trained_model(X_batch.to(device))
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(y_batch.numpy())
            test_dates.extend(dates)
            
    test_ic = calculate_ic(np.array(test_preds), np.array(test_targets), np.array(test_dates))
    print(f"\nFinal Test IC: {test_ic:.4f}")

if __name__ == '__main__':
    # 运行前请确保环境中已安装必要库，并且 sample.csv 与代码同级
    main()
```

### 关键细节说明：
1. **数据泄露防范**：数据集划分必须是**基于时间严格切割**（`train_end`, `val_end`），在时序和量价任务中绝不能用随机打乱法（`train_test_split`），否则未来数据泄露会导致模型在测试集虚高，实盘崩溃。
2. **为何用 Z-score 作为标签**：截面预测的本质是排序。用横截面 Z-score 作为Target，可以让模型的MSE Loss自动倾向于“拉大胜者和败者的预测差距”，与 IC（秩相关系数）指标天然契合。
3. **样本不均衡（不足8只）的处理**：因为在Dataset中，我是基于`ETFTimeSeriesDataset`进行逐只ETF的时间窗滚动提取的。预测时是对单只预测。验证时，我是用`df.groupby('date')`动态统计同一天存在的资产进行Rank IC计算。这种解法优雅地避开了因为截面维度对不齐导致的Tensor报错问题。

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0NzIzMDAwMjEsLTg1MjEyMjM4LC0xMj
c2NDUwMjk2LC02MTMzNjc3MDksMTc3ODEyOTQ2LC04NDI3MDE2
NTJdfQ==
-->