# Cliff Walking: Q-learning vs SARSA 對比實驗

**English Version Below** ⬇️

---

## 📋 中文說明

### 項目概述
本項目實作並比較兩種經典強化學習演算法：**Q-learning（離策略）** 與 **SARSA（同策略）**，通過相同的環境與參數設定，分析其學習行為、收斂特性及最終策略差異。

### 核心問題：Cliff Walking 環境
- **環境**：4 × 12 的矩形網格
- **起點**：左下角 (3, 0) 
- **終點**：右下角 (3, 11)
- **懸崖**：起點與終點間的底部區域
- **懲罰**：進入懸崖得 -100 獎勵並回到起點

### 🔑 核心發現

#### Q-learning（冒險家）❌
```
→ ← ↓ ↓ ↓ ↓ ↓ ↓ ← ↓ → ←
↓ ↓ ↓ → ↓ ↓ ↓ ↓ ← ↓ ↓ ←
↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ←
S C C C C C C C C C C G
```
- **路徑特性**：試圖走懸崖邊的最短路徑（13步）
- **平均獎勵（最後50回合）**：-58.70
- **波動程度**：標準差 154.24（極高）
- **原因**：Off-policy 忽視 ε-greedy 的探索風險

#### SARSA（保守派）✅
```
↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ←
↑ ↓ ↓ ↑ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ←
↑ ↓ ↑ ↓ ↓ ↑ ↑ ↓ ↓ → ↓ ←
S C C C C C C C C C C G
```
- **路徑特性**：遠離懸崖，選擇安全的上方路線（15-17步）
- **平均獎勵（最後50回合）**：-21.12
- **波動程度**：標準差 129.09（相對穩定）
- **原因**：On-policy 預估有 10% 概率隨機走，選擇風險較低的路徑

### 性能對比表

| 指標 | Q-learning | SARSA | 勝者 |
|------|-----------|-------|------|
| 平均獎勵 | -58.70 | -21.12 | **SARSA** ✅ |
| 最高獎勵 | -13.00 | -15.00 | Q-learning |
| 標準差 | 154.24 | 129.09 | **SARSA** ✅ |
| 穩定性 | 低（波動劇烈） | 高（相對平穩） | **SARSA** ✅ |
| 收斂速度 | 較快 | 較慢 | Q-learning |

### 為什麼 SARSA 獎勵更高？

$$E[\text{reward}]_{\text{SARSA}} = -15 + (-1) \times 2 + 0.1 \times (-100) = -23 \text{ (相對安全)}$$

$$E[\text{reward}]_{\text{Q-learning}} = -13 + 0.1 \times (-100) = -23 \text{ (但訓練過程中掉崖次數多)}$$

在 ε=0.1 的探索環境下，Q-learning 的理論最短路徑實際上**極其危險**——每 10 步中有 1 步會隨機執行，導致頻繁掉入懸崖。

### 應用建議

#### 使用 SARSA 的場景 🛡️
- 自主駕駛、機器人控制
- 醫療決策、金融交易
- 任何實際成本很高的應用
- **需要在訓練過程中保持安全**

#### 使用 Q-learning 的場景 🎮
- 遊戲 AI、模擬環境
- 可以容忍失敗的應用
- 離線強化學習
- **最終策略比訓練過程更重要**

---

## 📊 文件說明

### 程式碼文件
- **`cliff_walking.py`** - 主要實作
  - Q-learning 類別
  - SARSA 類別
  - 訓練、評估、可視化函數
  - 完整的對比分析

- **`visualize_qlearning.py`** - Q-learning 深度分析
  - 6 個子圖的詳細分析
  - 策略網格視覺化
  - 價值函數熱力圖
  - 英文標籤（可正常顯示）

### 圖表文件
- **`cliff_walking_comparison.png`** 
  - SARSA vs Q-learning 對比（4個子圖）
  
- **`qlearning_detailed_analysis.png`**
  - Q-learning 完整分析（6個子圖）
  
- **`qlearning_policy_grid.png`**
  - Q-learning 策略網格詳細圖
  
- **`qlearning_value_heatmap.png`**
  - Q-learning 狀態價值函數熱力圖

---

## 🚀 快速開始

### 環境要求
```bash
Python 3.10+
gymnasium >= 0.26
numpy >= 1.20
matplotlib >= 3.5
```

### 安裝依賴
```bash
cd H2
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 或 .venv\Scripts\activate  (Windows)

pip install gymnasium numpy matplotlib
```

### 執行程式

**1. 執行 SARSA vs Q-learning 對比實驗**
```bash
python cliff_walking.py
```

**2. 執行 Q-learning 詳細分析**
```bash
python visualize_qlearning.py
```

### 預期輸出
```
============================================================
Cliff Walking Experiment: SARSA vs Q-learning
============================================================

Starting SARSA Agent Training (On-policy)...
SARSA - Episode 100 | Cumulative Reward: -49
SARSA - Episode 200 | Cumulative Reward: -23
...

Starting Q-learning Agent Training (Off-policy)...
Q-learning - Episode 100 | Cumulative Reward: -64
Q-learning - Episode 200 | Cumulative Reward: -32
...

Performance Comparison Analysis
============================================================
SARSA:
  - Mean Reward (Last 50 episodes): -21.12
  - Max Reward: -15.00
  - Min Reward: -2557.00
  - Std Dev: 129.09

Q-learning:
  - Mean Reward (Last 50 episodes): -58.70
  - Max Reward: -13.00
  - Min Reward: -2883.00
  - Std Dev: 154.24

✅ Chart saved: cliff_walking_comparison.png
```

---

## 📐 演算法細節

### 環境參數
| 參數 | 值 |
|------|-----|
| State Space | 4×12 = 48 states |
| Action Space | {0:↑, 1:↓, 2:←, 3:→} |
| Learning Rate (α) | 0.1 |
| Discount Factor (γ) | 0.9 |
| Exploration Rate (ε) | 0.1 |
| Episodes | 500 |
| Reward per step | -1 |
| Cliff penalty | -100 |

### Q-learning 更新公式（Off-policy）
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

**特點**：使用 $\max_{a} Q(S_{t+1}, a)$ —— 假設未來採取最優動作

### SARSA 更新公式（On-policy）
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

**特點**：使用 $Q(S_{t+1}, A_{t+1})$ —— 實際採取的下一個動作

---

## 🔬 理論解釋

### 為什麼會有差異？

#### 1️⃣ **價值函數的定義**

**Q-learning**：學習 $Q^*$ （最優價值函數）
- 無視探索策略
- 期望最終得到理論最優解
- 訓練過程中可能很糟糕

**SARSA**：學習在實際策略下的 $Q^\pi$ 
- 考慮探索的風險
- 期望在實際執行中表現良好
- 訓練與測試性能一致

#### 2️⃣ **Cliff Walking 的陷阱**

在這個環境中：
- 最短路徑 = 13 步（沿著懸崖邊）
- 安全路徑 = 15-17 步（上方繞過）

當 ε=0.1 時：
- **Q-learning**：學到「走最短路」，但 10% 的隨機動作會掉崖
- **SARSA**：學到「考慮 10% 風險」，所以選擇安全的繞路

#### 3️⃣ **期望獎勵計算**

最短路徑（13步）+ 每步 -1 獎勵：
$$E = -13 + 0.1 \times (-100) = -23$$

上方路徑（16步）+ 每步 -1 獎勵：
$$E = -16 + 0.1 \times (-100) = -26$$

差異只有 3 分，但在訓練過程中，Q-learning 掉崖的頻率遠高於 SARSA！

---

## 📚 引用與參考

### 經典論文
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Rummery, G. A., & Niranjan, M. (1994). "On-Line Q-Learning Using Connectionist Systems"

### 相關資源
- [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/)
- [Sutton & Barto - RL Book](http://incompleteideas.net/book/the-book-2nd.html)

---

## 📄 作業要求檢核

| 要求項目 | 狀態 | 說明 |
|---------|------|------|
| 演算法實作 | ✅ | Q-learning 與 SARSA 皆已實作 |
| 環境設定 | ✅ | CliffWalking-v1, 4×12 網格 |
| 參數配置 | ✅ | α=0.1, γ=0.9, ε=0.1, 500 episodes |
| 訓練過程 | ✅ | 相同環境與參數進行對比 |
| 結果分析 | ✅ | 獎勵曲線、策略視覺化、穩定性分析 |
| 理論討論 | ✅ | Off-policy vs On-policy 完整說明 |
| 結論 | ✅ | 應用場景與選擇建議 |

---

---

# English Version

## 📋 Project Overview

This project implements and compares two classic reinforcement learning algorithms: **Q-learning (Off-policy)** and **SARSA (On-policy)** using the Cliff Walking environment.

### The Cliff Walking Problem
- **Environment**: 4 × 12 rectangular grid
- **Start**: Bottom-left corner (3, 0)
- **Goal**: Bottom-right corner (3, 11)
- **Cliff**: Dangerous area between start and goal
- **Penalty**: -100 reward for entering cliff, agent resets to start

### 🔑 Key Findings

#### Q-learning (Risk-taker) ❌
- **Path**: Attempts shortest path along cliff edge (13 steps)
- **Mean Reward**: -58.70
- **Volatility**: Std Dev 154.24 (Very High)
- **Reason**: Off-policy ignores exploration risk

#### SARSA (Conservative) ✅
- **Path**: Safe route avoiding cliff (15-17 steps)
- **Mean Reward**: -21.12
- **Volatility**: Std Dev 129.09 (Relatively Stable)
- **Reason**: On-policy accounts for 10% random exploration

### Performance Comparison

| Metric | Q-learning | SARSA | Winner |
|--------|-----------|-------|--------|
| Mean Reward | -58.70 | -21.12 | **SARSA** ✅ |
| Stability | Low | High | **SARSA** ✅ |
| Convergence Speed | Fast | Slow | Q-learning |
| Final Optimality | Better | Suboptimal | Q-learning |

### When to Use Each Algorithm

#### SARSA: Safe Learning 🛡️
- Autonomous driving, robotics
- Medical/financial decisions
- High-cost failures
- **Safety during training is critical**

#### Q-learning: Optimal Offline Learning 🎮
- Game AI, simulations
- Failure-tolerant applications
- Offline RL scenarios
- **Final policy > training process**

---

## 📊 File Description

### Code Files
- **`cliff_walking.py`** - Main implementation
  - Q-learning class
  - SARSA class
  - Training and comparison analysis

- **`visualize_qlearning.py`** - Q-learning detailed analysis
  - 6-subplot comprehensive analysis
  - Policy grid visualization
  - Value function heatmap

### Chart Files
- `cliff_walking_comparison.png` - SARSA vs Q-learning comparison
- `qlearning_detailed_analysis.png` - Q-learning deep dive
- `qlearning_policy_grid.png` - Policy grid detail
- `qlearning_value_heatmap.png` - State value function

---

## 🚀 Quick Start

### Requirements
```bash
Python 3.10+
gymnasium >= 0.26
numpy >= 1.20
matplotlib >= 3.5
```

### Installation
```bash
cd H2
python -m venv .venv
source .venv/bin/activate

pip install gymnasium numpy matplotlib
```

### Run Experiments

**1. SARSA vs Q-learning Comparison**
```bash
python cliff_walking.py
```

**2. Q-learning Detailed Analysis**
```bash
python visualize_qlearning.py
```

---

## 📐 Algorithm Details

### Update Formulas

**Q-learning (Off-policy)**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

**SARSA (On-policy)**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

### Environment Parameters
| Parameter | Value |
|-----------|-------|
| State Space | 48 states (4×12) |
| Action Space | 4 (UP, DOWN, LEFT, RIGHT) |
| Learning Rate | 0.1 |
| Discount Factor | 0.9 |
| Exploration Rate | 0.1 |
| Episodes | 500 |

---

## 🔬 Theoretical Explanation

### Why the Difference?

**Q-learning** learns $Q^*$ (optimal) but ignores exploration risk
- Best asymptotic performance
- Risky during training

**SARSA** learns $Q^\pi$ (on-policy value) accounting for actual exploration
- Consistent training/testing performance
- Safer exploration strategy

In Cliff Walking with ε=0.1:
- Shortest path (13 steps) + 10% cliff falls = Expected reward -23
- Safe path (16 steps) + 10% cliff falls = Expected reward -26

SARSA learns the safe path because it accounts for the risk!

---

## 📚 References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- [OpenAI Gymnasium](https://gymnasium.farama.org/)

---

## ✅ Assignment Checklist

- ✅ Algorithm Implementation (Q-learning & SARSA)
- ✅ Environment Setup (CliffWalking 4×12)
- ✅ Parameter Configuration
- ✅ Training Process
- ✅ Results Analysis
- ✅ Theoretical Discussion
- ✅ Conclusions & Recommendations

---

## 📧 Author
Reinforcement Learning Assignment - Cliff Walking Experiment

## 📜 License
MIT License
