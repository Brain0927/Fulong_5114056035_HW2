# 🎓 Q-Learning 與 SARSA 演算法之比較研究 - 代碼整合指南

## 📋 專案概述

本專案是對強化學習中兩種重要時間差分（Temporal Difference, TD）演算法的全面比較研究：
- **Q-Learning**：離策略（Off-policy）演算法
- **SARSA**：在策略（On-policy）演算法

在 Gymnasium 懸崖行走環境（CliffWalking-v1）中進行 50 次獨立實驗。

---

## 📁 檔案結構說明

```
📦 H2/
├── 🎯 主程式
│   ├── integrated_study.py          ⭐ 【完整整合版本】
│   ├── comparison_study.py          【英文標籤版本】
│   └── comparison_study_english.py  【純英文版本】
│
├── 🔬 專門化分析
│   ├── visualize_cliff_environment.py   【環境策略可視化】
│   └── visualize_qlearning.py          【Q-Learning 深度分析】
│
├── 📊 生成的可視化
│   ├── qlearning_vs_sarsa_comparison.png          【4 子圖對比】
│   ├── cliff_walking_policies.png                【6 子圖政策】
│   ├── cliff_walking_policy_comparison.png       【策略對比】
│   └── cliff_walking_value_comparison.png        【價值函數】
│
└── 📚 文檔
    ├── VISUALIZATION_GUIDE.md       【可視化說明】
    ├── README.md                    【快速開始】
    └── DETAILED_ANALYSIS.md         【深度分析】
```

---

## 🚀 快速開始

### 執行整合版本（推薦）
```bash
python integrated_study.py
```

**輸出：**
- 50 次實驗運行結果
- 統計分析對比
- 生成 PNG 圖表

### 執行環境可視化
```bash
python visualize_cliff_environment.py
```

**輸出：**
- 策略箭頭可視化
- 價值函數熱力圖
- 3 個不同角度的 PNG 圖表

---

## 📚 程式碼架構

### integrated_study.py 主要類別

#### 1️⃣ EnvironmentConfig
```python
配置類 - 管理環境參數和超參數
├── 環境參數
│   ├── grid_size: (4, 12)
│   ├── num_states: 48
│   ├── num_actions: 4
│   └── cliff_states: {37-46}
├── 獎勵設置
│   ├── step_reward: -1
│   ├── cliff_reward: -100
│   └── goal_reward: 0
└── 超參數
    ├── learning_rate (α): 0.1
    ├── discount_factor (γ): 0.9
    ├── epsilon: 0.1
    └── episodes: 500
```

#### 2️⃣ QLearningAgent
```python
Q-Learning 代理實現
├── 特徵：離策略演算法
├── 更新規則：max Q(s',a')
├── 方法
│   ├── select_action()      - ε-貪心選擇
│   ├── update()             - Q值更新
│   ├── train()              - 訓練循環
│   └── get_learned_policy() - 獲取策略
└── 屬性
    ├── Q: 行動價值表
    ├── rewards_history: 獎勵歷史
    └── episode_paths: 路徑長度
```

#### 3️⃣ SARSAAgent
```python
SARSA 代理實現
├── 特徵：在策略演算法
├── 更新規則：Q(s',a') 實際執行的
├── 方法
│   ├── select_action()      - ε-貪心選擇
│   ├── update()             - Q值更新
│   ├── train()              - 訓練循環
│   └── get_learned_policy() - 獲取策略
└── 屬性
    ├── Q: 行動價值表
    ├── rewards_history: 獎勵歷史
    └── episode_paths: 路徑長度
```

#### 4️⃣ AlgorithmComparison
```python
實驗比較管理類
├── 方法
│   ├── run_experiments()              - 執行 50 次實驗
│   ├── analyze_results()              - 統計分析
│   └── print_comparison_results()     - 打印報告
├── 屬性
│   ├── ql_results: Q-Learning 結果
│   └── sarsa_results: SARSA 結果
└── 存儲
    ├── ql_agents: Q-Learning 代理列表
    └── sarsa_agents: SARSA 代理列表
```

---

## 🎯 核心演算法

### Q-Learning 更新公式
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

**特徵：**
- 使用 $\max_{a'} Q(s',a')$：樂觀估計
- 收斂到最優值函數 $Q^*$
- 快速收斂但訓練不穩定

**代碼實現：**
```python
max_next_q = 0.0 if terminated else np.max(self.Q[next_state])
td_error = reward + gamma * max_next_q - self.Q[state][action]
self.Q[state][action] += alpha * td_error
```

### SARSA 更新公式
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]$$

**特徵：**
- 使用 $Q(s',a')$：實際執行的動作
- 收斂到當前策略值函數 $Q^\pi$
- 收斂慢但訓練穩定

**代碼實現：**
```python
next_q = 0.0 if terminated else self.Q[next_state][next_action]
td_error = reward + gamma * next_q - self.Q[state][action]
self.Q[state][action] += alpha * td_error
```

---

## 📊 實驗結果

### 50 次實驗後 (最後 50 回合平均)

| 指標 | Q-Learning | SARSA | 差異 |
|------|-----------|-------|------|
| **平均獎勵** | -49.17 | -23.56 | +25.61 ✓ |
| **標準差** | 9.04 | 2.86 | -6.19 ✓ |
| **最大值** | -29.02 | -18.44 | +10.58 ✓ |
| **最小值** | -67.68 | -31.32 | +36.36 ✓ |

### 關鍵發現

1. **性能：** SARSA 平均獎勵高 25.61 分
2. **穩定性：** SARSA 標準差低 68.4%
3. **原因：** Q-Learning 樂觀估計導致頻繁掉崖
4. **策略：** SARSA 採用保守路線避免危險區域

---

## 🎨 可視化輸出

### qlearning_vs_sarsa_comparison.png
```
┌─────────────────────────────────────┐
│ 1. Learning Curves  │ 2. Moving Avg │
├─────────────────────┼───────────────┤
│ 3. Performance      │ 4. Distribution│
└─────────────────────────────────────┘

紅線 = Q-Learning  藍線 = SARSA
```

### cliff_walking_policies.png
```
┌───────────────────────┬───────────────────────┐
│  Q-Learning Policy    │  SARSA Policy         │
├───────────────────────┼───────────────────────┤
│  沿懸崖邊 (13 步)     │  上方安全路線 (15 步) │
└───────────────────────┴───────────────────────┘

↓ = DOWN  → = RIGHT  ↑ = UP  ← = LEFT
C = Cliff  S = Start  G = Goal
```

---

## 🔧 使用指南

### 修改實驗參數

```python
# 改變運行次數
config = EnvironmentConfig(num_runs=100)  # 默認 50

# 修改超參數
config.learning_rate = 0.05      # 更小的學習率
config.discount_factor = 0.95    # 更大的未來權重
config.epsilon = 0.05            # 更少的探索
config.episodes = 1000           # 更多回合
```

### 獲取特定代理的策略

```python
# 從比較器獲取第一個 Q-Learning 代理
first_ql_agent = comparator.ql_agents[0]
policy = first_ql_agent.get_learned_policy()

# 查看特定狀態的最佳動作
best_action = policy[state]  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
```

---

## 📈 進階分析

### 觀察 Q 值分布

```python
# 訓練後查看 Q 值
agent = QLearningAgent(config)
agent.train(env)

# 獲取特定狀態的 Q 值
state_q_values = agent.Q[state]  # 4 個動作的 Q 值
best_action = np.argmax(state_q_values)
```

### 追蹤訓練動態

```python
# 查看獎勵歷史
print(f"第 1-50 回合平均: {np.mean(agent.rewards_history[:50])}")
print(f"第 450-500 回合平均: {np.mean(agent.rewards_history[450:])}")

# 查看路徑長度
print(f"初期平均路徑長度: {np.mean(agent.episode_paths[:50])}")
print(f"最後平均路徑長度: {np.mean(agent.episode_paths[450:])}")
```

---

## 💡 教學要點

### 1. 離策略 vs 在策略
```
Q-Learning (Off-policy):
  ├─ 探索策略：ε-貪心
  └─ 學習策略：greedy (max)

SARSA (On-policy):
  ├─ 探索策略：ε-貪心
  └─ 學習策略：ε-貪心 (same)
```

### 2. 收斂性質
```
Q-Learning:
  收斂到 Q*(s,a) - 最優值函數
  ∑α_t → ∞, ∑α_t² < ∞

SARSA:
  收斂到 Q^π(s,a) - 當前策略值
  π = ε-貪心
```

### 3. 環境中的表現
```
懸崖環境特性：
  • 高風險區域（懸崖）
  • 有明確的最優路線
  • 探索時的失敗代價高

Q-Learning: 樂觀 → 冒險 → 頻繁失敗
SARSA: 謹慎 → 保守 → 穩定學習
```

---

## 🎓 相關資源

### 理論基礎
- Sutton & Barto (2018): *Reinforcement Learning: An Introduction*
- Q-Learning: Watkins & Dayan (1992)
- SARSA: Rummery & Niranjan (1994)

### 環境
- Gymnasium CliffWalking-v1
- 官方文檔：https://gymnasium.farama.org/

### 延伸學習
- Deep Q-Learning (DQN)
- Expected SARSA
- Double Q-Learning
- Dueling Architecture

---

## 📝 貢獻日誌

| 日期 | 更新內容 |
|------|--------|
| 2026-05-04 | 完成整合版本 integrated_study.py |
| 2026-05-04 | 添加可視化模組 visualize_cliff_environment.py |
| 2026-05-04 | 英文標籤版本 comparison_study_english.py |
| 2026-05-04 | 完成 50 次實驗運行和統計分析 |

---

## ✅ 檢查清單

- [x] Q-Learning 演算法實現
- [x] SARSA 演算法實現
- [x] 環境配置與參數化
- [x] 50 次獨立實驗運行
- [x] 統計分析與對比
- [x] 四向量可視化圖表
- [x] 策略和價值函數可視化
- [x] 詳細英文/中文標籤
- [x] 完整文檔說明
- [x] 代碼整合與優化

---

**最後更新：2026-05-04**  
**學生：Fulong (5114056035)**  
**課程：強化學習**
