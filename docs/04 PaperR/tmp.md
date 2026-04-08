


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

你是一个光电方向的专家，同时也是一个ai专家，研究方向为二者的交叉，你当前在进行一个实验，对番茄喷洒农药，然后对齐进行光谱分析，根据光谱结果来预测所喷洒的农药的浓度。从ai的角度理解，我现在有一堆样本数据，其存储在excel表格中，其中每行是一个样本（首行是列名），最后一列是预测的label值，例如，0表示未喷洒农药，0.5表示喷洒农药其浓度为0.5；1表示喷洒农药浓度为1。前面438列表示不同的频段。注意这是一个回归任务，在指标里需要你计算R2.
我已经实现了一个简易版本的代码，我需要你审计我的代码，思考改进的方案，改进的思路包括数据预处理（可能包括对光谱数据的特殊处理，如多元散射校正MSC等）、预测模型等。
我可以为你提供的信息是多元散射校正 (MSC）是一个相对好一点的预处理方法
需要你深度思考，给出思考改进方法的思维过程。

# 输出python代码要求：
在一个python文件中，包括文件读取，数据预处理，数据集划分，模型训练，模型测试，测试结果的保存，测试结果的性能（准确率等）的输出
深度学习代码要使用pytorch编写

# 已实现的版本







# code
这是一个非常典型的量化金融与深度学习交叉任务（时序动量/截面反转与截面回归预测）。

为了满足你的需求，我将回答分为两个部分：**第一部分**是对模型架构迭代路线的深度思考（由简入深）；**第二部分**是满足你所有要求的、高度解耦且可运行的完整Python流程代码。

---

### 第一部分：模型架构改进迭代路线 (深度思考)

在量化截面预测任务中，核心难点是：**既要捕捉单只标的（ETF）的时间序列特征（时序依赖），又要捕捉同时期不同标的之间的相对强弱关系（截面依赖）。** 

为了保持代码的高解耦性，所有的模型都会遵循相同的输入输出格式：
- **输入维度**: `(Batch_size, T, Features)`，本任务中 T=60。
- **输出维度**: `(Batch_size, 1)`，即预测的截面收益率（Rank化处理后的数值）。

**🚀 建议的迭代路线（从V1到V4）：**

*   **V1 (当前基线 - 时序特征提取为主):** `GRU` / `ALSTM` 
    *   *逻辑*：将每只ETF独立看待，通过RNN类模型提取过去60天的价量演变特征。ALSTM通过引入Attention机制，能更好识别60天中对未来一天影响最大的关键节点。
    *   *缺点*：只看了自己，没看别人，缺乏真实的“截面相互作用”感知。
*   **V2 (时序+静态截面特征融合):** `LSTM-CNN` 或 `TS-Mixer`
    *   *逻辑*：输入端不仅包含时间序列，还将所有ETF在时间 $t$ 的状态作为一个矩阵进行一维卷积操作（或者使用多层感知机进行混频），迫使模型在最后几层融合不同资产的隐含特征。
*   **V3 (引入自注意力机制感知截面):** `Cross-Sectional Transformer` (横向Attention)
    *   *逻辑*：先用GRU处理每个ETF的60天序列得到时序Embedding，然后在截面维度（每天的 N 个 ETF Embeddings）上做一层 Self-Attention。这样某只ETF的预测打分会自动参考同日其他ETF的状态。
*   **V4 (图神经网络+时空注意力):** `GAT-LSTM` 或 `AlphaNet` 架构
    *   *逻辑*：构建一张图，节点是ETF，边可以是ETF之间的历史相关系数矩阵或协方差矩阵。使用图注意力网络(GAT)在截面上进行信息传递，再结合LSTM。这是目前顶级学术论文和头部量化私募常用的高级架构。

---

### 第二部分：完整 Python 流程代码

代码已完全解耦，修改 `Model` 类即可无缝迭代。包含了：**数据防重叠划分、无未来函数标准化、截面排序标签、IC监控、早停机制以及Top-K多指标回测**。

#### 依赖库安装提示
你需要安装：`pip install pandas numpy torch scikit-learn scipy`

#### 完整代码：

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置参数 (Config)
# ==========================================
class Config:
    file_path = 'sample.csv'  # 你的数据路径
    T = 60                    # 回看窗口大小
    batch_size = 256
    epochs = 100
    lr = 1e-3
    patience = 10             # 早停耐心值
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    top_k_list = [1, 2, 3, 4] # 回测Top-K种类
    
    # 划分比例
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2

# ==========================================
# 2. 数据预处理模块 (Data Processing)
# ==========================================
def preprocess_data(cfg):
    df = pd.read_csv(cfg.file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', 'instrument'])
    
    # 特征选择: 剔除ID类、字符串类，只保留量价特征
    features = ['s_dq_preclose', 's_dq_open', 's_dq_high', 's_dq_low', 's_dq_close', 
                's_dq_change', 's_dq_pctchange', 's_dq_volume', 's_dq_amount',
                's_dq_adjpreclose', 's_dq_adjopen', 's_dq_adjhigh', 's_dq_adjlow', 
                's_dq_adjclose', 's_dq_adjfactor', 'trades_count', 'discount_rate']
    
    # 计算明日真实收益率 (策略回测用)
    # 假设：今天收盘时进行预测和建仓，持有到明天收盘。实际收益率 = (明收 - 昨收(即今收)) / 昨收
    df['actual_return'] = df.groupby('instrument')['s_dq_close'].shift(-1) / df['s_dq_close'] - 1
    
    # 计算截面收益率Rank作为标签 (分布在 [-0.5, 0.5])
    # 注意：这里的标签也是基于明日的实际收益率排序，因为我们要预测明日
    def calc_cross_sectional_rank(group):
        count = len(group)
        if count == 1:
            group['target'] = 0.0
        else:
            ranks = group['actual_return'].rank(method='first')
            group['target'] = ranks / count - 0.5
        return group

    df = df.dropna(subset=['actual_return']) # 丢弃最后一天（没有明日数据）
    df = df.groupby('date').apply(calc_cross_sectional_rank).reset_index(drop=True)
    df = df.dropna(subset=['target'] + features)

    # 划分数据集 (时间序列划分，防泄露)
    unique_dates = df['date'].sort_values().unique()
    train_split = int(len(unique_dates) * cfg.train_ratio)
    val_split = int(len(unique_dates) * (cfg.train_ratio + cfg.val_ratio))
    
    train_dates = unique_dates[:train_split]
    val_dates = unique_dates[train_split:val_split]
    test_dates = unique_dates[val_split:]
    
    df_train = df[df['date'].isin(train_dates)].copy()
    df_val = df[df['date'].isin(val_dates)].copy()
    df_test = df[df['date'].isin(test_dates)].copy()
    
    # 特征标准化 (避免未来信息，仅用Train拟合)
    scaler = StandardScaler()
    df_train[features] = scaler.fit_transform(df_train[features])
    df_val[features] = scaler.transform(df_val[features])
    df_test[features] = scaler.transform(df_test[features])
    
    return df_train, df_val, df_test, features

class ETFDataset(Dataset):
    def __init__(self, df, features, T):
        self.data = []
        df_grouped = df.groupby('instrument')
        
        for _, group in df_grouped:
            group = group.sort_values('date').reset_index(drop=True)
            if len(group) <= T:
                continue
                
            feat_values = group[features].values
            targets = group['target'].values
            returns = group['actual_return'].values
            dates = group['date'].values
            instruments = group['instrument'].values
            
            # 滚动窗口 T
            for i in range(T, len(group)):
                x = feat_values[i-T : i]     # 过去 T 天的特征
                y = targets[i]               # t时刻预测的截面标签(实际上是t+1的截面表现)
                ret = returns[i]             # 真实收益率(回测用)
                date = dates[i]
                ins = instruments[i]
                self.data.append((x, y, ret, date, ins))
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        x, y, ret, date, ins = self.data[idx]
        return (torch.tensor(x, dtype=torch.float32), 
                torch.tensor(y, dtype=torch.float32), 
                ret, 
                str(date), 
                ins)

# ==========================================
# 3. 模型架构解耦区 (Models)
# ==========================================
# 基础GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :]) # 取最后一个时间步
        return out

# Attention LSTM 模型 (ALSTM)
class ALSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(ALSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        # Attention 参数
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (batch, T, input_dim)
        lstm_out, _ = self.lstm(x) # lstm_out: (batch, T, hidden_dim)
        
        # Attention 机制
        # u = tanh(W * H)
        u = torch.tanh(torch.matmul(lstm_out, self.w_omega)) # (batch, T, hidden_dim)
        # alpha = softmax(u * u_w)
        att = torch.matmul(u, self.u_omega) # (batch, T, 1)
        att_score = torch.softmax(att, dim=1) 
        
        # v = sum(alpha * H)
        scored_x = lstm_out * att_score # (batch, T, hidden_dim)
        context = torch.sum(scored_x, dim=1) # (batch, hidden_dim)
        
        out = self.fc(context)
        return out

# ==========================================
# 4. 训练与验证模块 (Training & IC Evaluation)
# ==========================================
def evaluate_ic(model, dataloader, device):
    """
    计算验证集上的平均 Rank IC
    """
    model.eval()
    preds_list = []
    trues_list = []
    dates_list = []
    
    with torch.no_grad():
        for x, y, _, dates, _ in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).squeeze().cpu().numpy()
            trues = y.cpu().numpy()
            
            # 单样本补齐为数组
            if preds.ndim == 0: preds = np.array([preds])
            if trues.ndim == 0: trues = np.array([trues])
            
            preds_list.extend(preds)
            trues_list.extend(trues)
            dates_list.extend(dates)
            
    df_eval = pd.DataFrame({'date': dates_list, 'pred': preds_list, 'true': trues_list})
    ic_list = []
    for _, group in df_eval.groupby('date'):
        if len(group) > 1:
            # Rank IC使用Spearman相关系数
            ic, _ = spearmanr(group['pred'], group['true'])
            if not np.isnan(ic):
                ic_list.append(ic)
                
    return np.mean(ic_list) if ic_list else 0.0

def train_model(model, train_loader, val_loader, cfg):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    best_ic = -np.inf
    patience_counter = 0
    
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for x, y, _, _, _ in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            preds = model(x).squeeze()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            
        train_loss /= len(train_loader.dataset)
        val_ic = evaluate_ic(model, val_loader, cfg.device)
        
        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | Train MSE: {train_loss:.4f} | Val RankIC: {val_ic:.4f}")
        
        # 早停机制 (以 Val IC 为监控指标)
        if val_ic > best_ic:
            best_ic = val_ic
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch+1}. Best Val IC: {best_ic:.4f}")
                break

# ==========================================
# 5. 回测与评价模块 (Backtesting)
# ==========================================
def calculate_metrics(strat_returns, true_ranks, pred_ranks, k):
    """ 计算单一K值下的所有回测指标 """
    # 策略收益率曲线
    strat_returns = np.array(strat_returns)
    cum_returns = np.cumprod(1 + strat_returns)
    
    # 年化收益率 (假设每年252个交易日)
    ann_return = cum_returns[-1] ** (252 / len(strat_returns)) - 1 if len(strat_returns) > 0 else 0
    
    # 夏普比率 (无风险利率设为0)
    volatility = np.std(strat_returns)
    sharpe = (np.mean(strat_returns) / volatility * np.sqrt(252)) if volatility > 1e-6 else 0
    
    # 最大回撤
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
    
    # 卡玛比率
    calmar = (ann_return / max_drawdown) if max_drawdown > 1e-6 else 0
    
    # Precision@K
    precisions = []
    ndcgs = []
    for tr, pr in zip(true_ranks, pred_ranks):
        if len(tr) <= k: continue  # 标的数量不足k个，跳过
        
        true_top_k = np.argsort(tr)[-k:]
        pred_top_k = np.argsort(pr)[-k:]
        hits = len(set(true_top_k) & set(pred_top_k))
        precisions.append(hits / k)
        
        # NDCG@K
        # scikit-learn的ndcg输入维度需要是二维的(n_samples, n_labels)
        try:
            ndcg = ndcg_score(np.array([tr]), np.array([pr]), k=k)
            ndcgs.append(ndcg)
        except:
            pass

    return {
        'K': k,
        'Ann_Return': ann_return,
        'Sharpe': sharpe,
        'Max_Drawdown': max_drawdown,
        'Calmar': calmar,
        'Precision@K': np.mean(precisions) if precisions else 0,
        'NDCG@K': np.mean(ndcgs) if ndcgs else 0
    }

def run_backtest(model, test_loader, cfg):
    model.eval()
    
    # 收集测试集预测结果
    res = []
    with torch.no_grad():
        for x, y, ret, dates, ins in test_loader:
            x = x.to(cfg.device)
            preds = model(x).squeeze().cpu().numpy()
            
            # 单样本补齐
            if preds.ndim == 0: preds = np.array([preds])
            rets = ret.numpy()
            ys = y.cpu().numpy()
            
            for i in range(len(preds)):
                res.append({
                    'date': dates[i], 'instrument': ins[i], 
                    'pred': preds[i], 'true_rank': ys[i], 'actual_return': rets[i]
                })
                
    df_res = pd.DataFrame(res)
    
    # 逐日回测
    results = {k: {'strat_returns': [], 'true_ranks': [], 'pred_ranks': []} for k in cfg.top_k_list}
    
    for date, group in df_res.groupby('date'):
        group = group.sort_values('pred', ascending=False)
        true_r = group['true_rank'].values
        pred_r = group['pred'].values
        
        for k in cfg.top_k_list:
            if len(group) >= k:
                # Top-K 均匀建仓 (取预测排名前K的实际收益率均值)
                daily_ret = group.head(k)['actual_return'].mean()
                
                results[k]['strat_returns'].append(daily_ret)
                results[k]['true_ranks'].append(true_r)
                results[k]['pred_ranks'].append(pred_r)
                
    # 输出所有Top-K结果
    print("\n" + "="*50)
    print("BACKTEST RESULTS ON TEST SET")
    print("="*50)
    
    for k in cfg.top_k_list:
        metrics = calculate_metrics(
            results[k]['strat_returns'], 
            results[k]['true_ranks'], 
            results[k]['pred_ranks'], 
            k
        )
        print(f"--- Top-{k} Strategy ---")
        print(f"Annualized Return : {metrics['Ann_Return']*100:.2f}%")
        print(f"Sharpe Ratio      : {metrics['Sharpe']:.4f}")
        print(f"Max Drawdown      : {metrics['Max_Drawdown']*100:.2f}%")
        print(f"Calmar Ratio      : {metrics['Calmar']:.4f}")
        print(f"Precision@{k}       : {metrics['Precision@K']*100:.2f}%")
        print(f"NDCG@{k}            : {metrics['NDCG@K']:.4f}\n")


# ==========================================
# 6. 主程序执行区 (Main)
# ==========================================
if __name__ == "__main__":
    cfg = Config()
    
    print("1. Data Preprocessing...")
    df_train, df_val, df_test, feature_cols = preprocess_data(cfg)
    input_dim = len(feature_cols)
    
    # 创建Dataset (不打乱时间顺序，确保逻辑正确)
    train_dataset = ETFDataset(df_train, feature_cols, cfg.T)
    val_dataset = ETFDataset(df_val, feature_cols, cfg.T)
    test_dataset = ETFDataset(df_test, feature_cols, cfg.T)
    
    # Train DataLoader 可以打乱，Val/Test不打乱
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    # ----------------------------------------------------
    # 🌟 架构解耦：这里只要切换模型即可 
    # model = GRUModel(input_dim).to(cfg.device)
    # ----------------------------------------------------
    print("\n2. Initializing Model (ALSTM)...")
    model = ALSTMModel(input_dim=input_dim).to(cfg.device)
    
    print("\n3. Starting Training...")
    train_model(model, train_loader, val_loader, cfg)
    
    print("\n4. Running Backtest Module...")
    # 加载最佳模型进行回测
    model.load_state_dict(torch.load('best_model.pth'))
    run_backtest(model, test_loader, cfg)
```

### 代码精妙点说明：
1. **防止数据泄露设计**：标准化 `StandardScaler` 只在训练集拟合（`fit`），对于验证集和测试集只调用 `transform`。这是极为重要的量化准则。
2. **多生命周期处理**：在构建 `ETFDataset` 时，使用 `group.groupby('instrument')` 以及动态检查 `len(group) <= T` 自动适配了随着时间推移新上市或退市的ETF。
3. **真实收益对齐**：截面回归预测的是“明日的相对排名”，但在回测时必须赚取“明日的实际收益率”，代码中通过 `target` （排名用）和 `actual_return`（计算策略PnL用）严格区分并对齐，避免了假性高收益。
4. **一键切换Top-K**：`calculate_metrics` 封装了所有的IR指标（包含NDCG）和Finance指标。你可以直接在 `Config` 的 `top_k_list` 修改列表，程序自动循环并打印。




<!--stackedit_data:
eyJoaXN0b3J5IjpbMTgwOTg5MTMwNSwtMTgxNjk1MTkzNCwxMT
czMzc1OTI2LC0xNTgwODUwMTE3LDIwMjM4Njk5MjgsLTE0NzIz
MDAwMjEsLTg1MjEyMjM4LC0xMjc2NDUwMjk2LC02MTMzNjc3MD
ksMTc3ODEyOTQ2LC04NDI3MDE2NTJdfQ==
-->