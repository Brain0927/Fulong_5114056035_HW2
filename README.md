# Q-Learning vs SARSA Algorithm Comparative Study
# Q-Learning 與 SARSA 演算法之比較研究

**Student ID**: 5114056035 (Fulong)  
**Date**: 2026年5月4日 (May 4, 2026)  
**Environment**: Gymnasium CliffWalking-v1

---

## 📋 Table of Contents | 目錄

- [Project Overview | 項目概述](#project-overview)
- [Quick Start | 快速開始](#quick-start)
- [Environment Details | 環境詳情](#environment-details)
- [Algorithms | 算法詳解](#algorithms)
- [Results | 實驗結果](#results)
- [File Structure | 文件結構](#file-structure)
- [Usage | 使用方式](#usage)
- [Key Findings | 主要發現](#key-findings)
- [Conclusions | 結論](#conclusions)

---

## 🎯 Project Overview | 項目概述

This project implements and compares two fundamental **Temporal Difference (TD)** reinforcement learning algorithms:
- **Q-Learning** (Off-policy)
- **SARSA** (On-policy)

本項目實現並比較了兩種基本的**時間差分 (TD)** 強化學習算法：
- **Q-Learning**（離策略）
- **SARSA**（同策略）

### Key Features | 核心特性

✅ **50 Independent Experiments** | 50 次獨立實驗  
✅ **Comprehensive Statistical Analysis** | 完整的統計分析  
✅ **4 High-Quality Visualizations** | 4 張高質量可視化圖表  
✅ **Detailed Theoretical Analysis** | 詳細的理論分析  
✅ **Bilingual Documentation** | 中英雙語文檔  

---

## 🚀 Quick Start | 快速開始

### Prerequisites | 前置要求

```bash
Python 3.10+
pip (Python package manager)
```

### Installation | 安裝

```bash
# Clone or navigate to the project directory
cd /Users/brainshi/Desktop/強化學習/H2

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Study | 運行研究

```bash
# Run the complete comparison study (50 independent runs, ~5-10 minutes)
python integrated_study.py

# Or run the quick test version (3 runs, ~30 seconds)
python test_comparison.py

# Or run with English labels
python comparison_study_english.py
```

### Expected Output | 預期輸出

```
✅ Gymnasium environment check: OK

============================================================================
Q-Learning vs SARSA Algorithm Comparative Study
============================================================================

Run 1/50: QL avg= -49.23 | SARSA avg= -23.45
Run 2/50: QL avg= -48.56 | SARSA avg= -24.12
...
Run 50/50: QL avg= -50.12 | SARSA avg= -22.89

====== EXPERIMENTAL RESULTS ======
Q-LEARNING (Off-policy):
  Mean reward (last 50 eps): -49.17
  Std deviation:              9.04
  
SARSA (On-policy):
  Mean reward (last 50 eps): -23.56
  Std deviation:              2.86

✓ Chart saved: qlearning_vs_sarsa_comparison.png
```

---

## 🎮 Environment Details | 環境詳情

### Cliff Walking Environment | 懸崖行走環境

```
Grid Layout (4 rows × 12 columns):
  S = Start Position (3, 0)
  G = Goal Position (3, 11)  
  C = Cliff Area (3, 1-10)
  
  Row 3: S [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] G
  Row 2: [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
  Row 1: [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
  Row 0: [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
```

### Environment Parameters | 環境參數

| Parameter | Value | Description |
|-----------|-------|-------------|
| States | 48 | 4 rows × 12 columns |
| Actions | 4 | UP, RIGHT, DOWN, LEFT |
| Step Reward | -1 | Reward per step |
| Cliff Reward | -100 | Penalty for hitting cliff |
| Goal Reward | 0 | Reward for reaching goal |

### Training Hyperparameters | 訓練超參數

| Hyperparameter | Symbol | Value | Description |
|---|---|---|---|
| Learning Rate | α (alpha) | 0.1 | Step size for Q-value updates |
| Discount Factor | γ (gamma) | 0.9 | Future reward discount factor |
| Exploration Rate | ε (epsilon) | 0.1 | Probability of random action |
| Episodes per Run | - | 500 | Episodes per experiment |
| Number of Runs | - | 50 | Independent experiments |

---

## 🧠 Algorithms | 算法詳解

### Q-Learning (Off-Policy)

**Update Rule**: 
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

**Characteristics**:
- ✓ Uses **max** Q-value of next state
- ✓ **Off-policy**: Learns optimal policy while exploring
- ✓ **Optimistic bias**: Overestimates values
- ✓ **Faster convergence**: Learns directly toward optimal policy
- ⚠️ **Risk**: May take cliff edge in training

**特點**:
- ✓ 使用下一狀態的 **最大** Q 值
- ✓ **離策略**: 邊探索邊學習最優策略
- ✓ **樂觀偏差**: 高估價值
- ✓ **快速收斂**: 直接向最優策略學習
- ⚠️ **風險**: 訓練中可能掉崖

---

### SARSA (On-Policy)

**Update Rule**: 
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

**Characteristics**:
- ✓ Uses **actual** next action's Q-value
- ✓ **On-policy**: Learns the policy being used
- ✓ **Conservative bias**: Underestimates values
- ✓ **Slower convergence**: But more stable
- ✓ **Safety**: Avoids cliff in training

**特點**:
- ✓ 使用**實際採取**的下一動作的 Q 值
- ✓ **同策略**: 學習正在使用的策略
- ✓ **保守偏差**: 低估價值
- ✓ **慢速收斂**: 但更穩定
- ✓ **安全**: 訓練中避免掉崖

---

## 📊 Results | 實驗結果

### Summary Statistics | 總結統計

**Last 50 Episodes Average**:

| Algorithm | Mean Reward | Std Dev | Improvement |
|-----------|------------|---------|------------|
| Q-Learning | -49.17 | 9.04 | Baseline |
| SARSA | -23.56 | 2.86 | **+109%** ✓ |

### Performance Comparison | 性能對比

```
Q-Learning:
  • Average: -49.17 ± 9.04
  • Max: -37.00
  • Min: -100.00
  
SARSA:
  • Average: -23.56 ± 2.86
  • Max: -18.00
  • Min: -100.00

Key Finding: SARSA is 109% better in average reward!
核心發現: SARSA 平均獎勵提升 109%!
```

### Learning Curves | 學習曲線

Generated visualization: **qlearning_vs_sarsa_comparison.png**

The 4-subplot chart shows:
1. **Raw Learning Curves**: Both algorithms' reward trends
2. **Moving Average (50 eps)**: Smoothed performance over time
3. **Performance Metrics**: Mean, Max, Min comparison
4. **Reward Distribution**: Box plot of last 50 episodes

---

## 📁 File Structure | 文件結構

```
H2/
├── README.md                              ⭐ Project overview
├── integrated_study.py                    ⭐ Main entry point (RECOMMENDED)
├── comparison_study.py                    Alternative version (Chinese)
├── comparison_study_english.py            Alternative version (English)
├── test_comparison.py                     Quick test (3 runs, 30 sec)
├── visualize_cliff_environment.py         Environment visualization
│
├── CODE_INTEGRATION_GUIDE.md              Architecture & code details
├── GETTING_STARTED.md                     Setup & first steps
├── PROJECT_SUMMARY.md                     Complete project overview
├── PROJECT_STATUS.txt                     Current status & checklist
├── QUICK_REFERENCE.md                     30-second reference card
├── VISUALIZATION_GUIDE.md                 Chart explanations
│
├── qlearning_vs_sarsa_comparison.png      Main results (4 subplots)
├── cliff_walking_policies.png             Policy visualization
├── cliff_walking_policy_comparison.png    Side-by-side policies
├── cliff_walking_value_comparison.png     Value function heatmaps
│
├── requirements.txt                       Python dependencies
└── .venv/                                 Virtual environment (if created)
```

---

## 💻 Usage | 使用方式

### Option 1: Full Study (Recommended) | 選項 1: 完整研究（推薦）

```bash
# Run 50 independent experiments (~5-10 minutes)
python integrated_study.py
```

Output includes:
- Statistical analysis for each run
- Comparison results with detailed metrics
- Policy visualization for both algorithms
- Theoretical analysis
- 4-subplot comparison chart saved as PNG

---

### Option 2: Quick Test | 選項 2: 快速測試

```bash
# Run 3 quick experiments (~30 seconds)
python test_comparison.py
```

Perfect for:
- Testing environment setup
- Verifying installation
- Quick validation

---

### Option 3: Alternative Versions | 選項 3: 替代版本

```bash
# Chinese version with detailed comments
python comparison_study.py

# Pure English version
python comparison_study_english.py
```

---

### Option 4: Environment Visualization | 選項 4: 環境可視化

```bash
# Visualize policies and value functions
python visualize_cliff_environment.py
```

Generates:
- Policy arrow visualization
- Value function heatmaps
- Comparison between algorithms

---

## 🔍 Key Findings | 主要發現

### 1. Performance Comparison | 性能對比

**SARSA outperforms Q-Learning by 109% in average reward**

```
Q-Learning:  -49.17 ± 9.04
SARSA:       -23.56 ± 2.86
Difference:  +25.61 (109% improvement)
```

### 2. Stability Analysis | 穩定性分析

**SARSA shows 68.4% lower variance**

```
Q-Learning Std Dev: 9.04
SARSA Std Dev:      2.86
Reduction:          68.4%
```

### 3. Policy Behavior | 策略行為

**Different strategies emerge**:

| Algorithm | Strategy | Risk Level | Path Length |
|-----------|----------|-----------|------------|
| Q-Learning | Aggressive (cliff edge) | HIGH | ~13 steps |
| SARSA | Conservative (safe path) | LOW | ~15-17 steps |

### 4. Convergence Pattern | 收斂模式

**Q-Learning**:
- Fast initial learning
- High variability
- Risky behavior during training
- Optimistic bias

**SARSA**:
- Slower initial learning
- Stable performance
- Safe behavior during training
- Conservative but reliable

---

## 📈 Theoretical Analysis | 理論分析

### Bias-Variance Tradeoff | 偏差-方差權衡

**Q-Learning (Off-Policy)**:
- ✓ Lower bias (learns optimal policy)
- ✗ Higher variance (overestimates values)
- Risk of cliff collision during training

**SARSA (On-Policy)**:
- ✗ Higher bias (learns actual policy)
- ✓ Lower variance (stable estimates)
- Safe training behavior

### Convergence Properties | 收斂性質

**Both algorithms guarantee convergence to optimal Q-values** with:
- Proper learning rate schedule
- Sufficient exploration (ε-greedy)
- Sufficient episodes

However:
- Q-Learning → converges to Q* (optimal)
- SARSA → converges to Q^π (policy-dependent)

---

## ✅ Conclusions | 結論

### When to Use Each Algorithm | 何時使用各算法

#### Choose Q-Learning When: | 選擇 Q-Learning 何時:
- ✓ Offline learning (no real-time interaction)
- ✓ Simulation environment (safe to explore)
- ✓ Goal is to find theoretical optimal policy
- ✓ System can tolerate training failures

#### Choose SARSA When: | 選擇 SARSA 何時:
- ✓ Online learning (learning while acting)
- ✓ Real-world environment (safety critical)
- ✓ Need stable, reliable performance
- ✓ Risk is expensive (medical, autonomous vehicles)

### Summary | 總結

This study demonstrates that:
1. **SARSA's stability** makes it preferable in real-world scenarios
2. **Q-Learning's optimality** makes it better for offline simulation
3. **Policy-on strategy alignment** (SARSA) leads to safer learning
4. **Off-policy learning** (Q-Learning) risks damage during training

本研究表明:
1. **SARSA 的穩定性**使其在實際應用中更優
2. **Q-Learning 的最優性**使其更適合離線模擬
3. **策略一致性**（SARSA）導致更安全的學習
4. **離策略學習**（Q-Learning）訓練中存在風險

---

## 📚 Additional Resources | 更多資源

### Documentation Files | 文檔文件

- **GETTING_STARTED.md** - Setup and first steps
- **CODE_INTEGRATION_GUIDE.md** - Architecture details
- **PROJECT_SUMMARY.md** - Comprehensive overview
- **QUICK_REFERENCE.md** - 30-second reference

### Visualization Files | 可視化文件

- **qlearning_vs_sarsa_comparison.png** - Main comparison chart
- **cliff_walking_policies.png** - 6-subplot analysis
- **cliff_walking_policy_comparison.png** - Side-by-side policies
- **cliff_walking_value_comparison.png** - Value function heatmaps

---

## 🔧 Troubleshooting | 故障排除

### Issue: "No module named 'gymnasium'" | 問題：找不到 gymnasium 模塊

```bash
Solution: pip install --upgrade gymnasium
```

### Issue: Plot window doesn't appear | 問題：繪圖窗口不顯示

```bash
Solution: Add to the end of your script:
import matplotlib.pyplot as plt
plt.show()
```

### Issue: Python version too old | 問題：Python 版本過舊

```bash
Check your Python version:
python --version

Required: Python 3.10 or higher
```

---

## 📝 Assignment Checklist | 作業檢查清單

- ✅ Q-Learning algorithm implementation
- ✅ SARSA algorithm implementation  
- ✅ Cliff Walking environment setup
- ✅ 50 independent experiments
- ✅ Statistical analysis
- ✅ Visualization charts
- ✅ Theoretical analysis
- ✅ Policy visualization
- ✅ Value function analysis
- ✅ Comprehensive documentation
- ✅ Code comments and explanation
- ✅ Performance comparison
- ✅ Conclusions and insights

---

## 📞 Contact & Support | 聯絡方式

**Project Status**: ✅ Complete  
**Last Updated**: 2026年5月4日  
**GitHub Repository**: https://github.com/Brain0927/Fulong_5114056035_HW2

---

## 📄 License | 許可證

This project is created for educational purposes as part of reinforcement learning coursework.

---

## 🙏 Acknowledgments | 致謝

- Gymnasium documentation for environment support
- NumPy and Matplotlib for scientific computing
- Sutton & Barto's "Reinforcement Learning: An Introduction"

---

**Happy Learning! | 祝您學習愉快！** 🎓
