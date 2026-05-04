# 🎯 Q-Learning vs SARSA - 快速參考卡片

## ⚡ 30 秒快速對比

| 特性 | Q-Learning | SARSA |
|------|-----------|-------|
| **類型** | 離策略 | 在策略 |
| **更新** | $\max Q(s',a')$ | $Q(s',A')$ |
| **性質** | 樂觀 | 保守 |
| **風險** | 高 | 低 |
| **收斂** | 快 | 慢 |
| **最終性能** | 差 (懸崖) | 優 (懸崖) |

---

## 📐 公式速記

### Q-Learning
```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
         ^^^^^^   ^  ^^^^^^^^^^^^^^^^^^^
         當前值   學習 貝爾曼目標值 (樂觀)
```

### SARSA
```
Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
         ^^^^^^   ^  ^^^^^^^^^^^^^^^
         當前值   學習 貝爾曼目標值 (實際)
```

---

## 🔑 核心差異

### Q-Learning (Off-policy)
```
1️⃣ 當前狀態 s 執行動作 a (ε-貪心)
2️⃣ 觀察獎勵 r 和下一狀態 s'
3️⃣ 計算最佳動作的 Q 值（即使不執行）
4️⃣ 更新基於最優估計
```

### SARSA (On-policy)
```
1️⃣ 當前狀態 s 執行動作 a (ε-貪心)
2️⃣ 觀察獎勵 r 和下一狀態 s'
3️⃣ 在 s' 執行ε-貪心選擇的動作 a'
4️⃣ 更新基於實際執行的動作
```

---

## 📊 實驗結果速覽

### 50 次運行 (最後 50 回合)

```
┌─────────────┬──────────┬────────┐
│   指標      │ Q-Learn  │ SARSA  │
├─────────────┼──────────┼────────┤
│ 平均獎勵    │  -49.17  │ -23.56 │ ⭐ SARSA 優勝
│ 標準差      │   9.04   │  2.86  │ ⭐ SARSA 穩定
│ 最大獎勵    │  -29.02  │ -18.44 │ ⭐ SARSA 最高
│ 最小獎勵    │  -67.68  │ -31.32 │ ⭐ SARSA 風險低
└─────────────┴──────────┴────────┘

📈 SARSA 性能優勢：+25.61 分 (109%)
📉 SARSA 穩定優勢：降低 68% 方差
```

---

## 🎮 懸崖環境解讀

```
Grid:  0    1    2    3    4    5    6    7    8    9   10   11
     ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
   0 │    │    │    │    │    │    │    │    │    │    │    │    │
     ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
   1 │    │    │    │    │    │    │    │    │    │    │    │    │
     ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
   2 │    │    │    │    │    │    │    │    │    │    │    │    │
     ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
   3 │ S  │ C  │ C  │ C  │ C  │ C  │ C  │ C  │ C  │ C  │ G  │    │
     └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

策略選擇：
├─ Q-Learning: → → → → → → → → → ↓ (13 步, 沿邊緣)
└─ SARSA:      → ↑ → ↑ ↑ → ↑ → ↑ ↓ (15-17 步, 上方)

獎勵：
├─ 步驟：-1
├─ 懸崖：-100
└─ 目標：0
```

---

## 💻 代碼模板

### 最小化實現
```python
import gymnasium as gym
import numpy as np
from collections import defaultdict

# ============ Q-Learning ============
env = gym.make("CliffWalking-v1")
Q = defaultdict(lambda: np.zeros(4))
alpha, gamma, epsilon = 0.1, 0.9, 0.1

for episode in range(500):
    state, _ = env.reset()
    while True:
        # ε-貪心
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        
        # ⭐ Q-Learning: max
        if done:
            Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
        else:
            max_q = np.max(Q[next_state])
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * max_q - Q[state][action])
        
        state = next_state
        if done:
            break

# ============ SARSA ============
# 相同結構，但改為：
next_action = np.argmax(Q[next_state]) if np.random.random() > epsilon else np.random.randint(4)
next_q = Q[next_state][next_action]  # ⭐ SARSA: 實際 Q
# ... 使用 next_q
```

---

## 🎯 何時使用哪個？

### 使用 Q-Learning 當
- ✅ 離線學習（訓練後再部署）
- ✅ 模擬環境（可接受失敗）
- ✅ 追求最優性能
- ✅ 計算資源充足

### 使用 SARSA 當
- ✅ 在線學習（邊學邊做）
- ✅ 現實環境（失敗代價高）
- ✅ 強調安全性
- ✅ 實時決策

---

## 📈 學習曲線特徵

```
獎勵
   0 │
     │    Q-Learning        SARSA
  -20│  ╱              ╱────────
     │ ╱              ╱
  -40│╱   ╱╲╱╲       ╱
     │    ╱  ╲╱╲    ╱
  -60│   ╱       ╲╱╲  ╱
     │  ╱           ╲╱
     └──────────────────────────> 回合
       0          250        500

特徵：
├─ Q-Learning: 波動大，有峰值
└─ SARSA: 平滑，單調遞增
```

---

## 🔬 實驗設定詳情

```
環境：       Gymnasium CliffWalking-v1
網格：       4×12 (48 狀態)
動作：       4 (UP, RIGHT, DOWN, LEFT)
起點：       (3,0) 狀態 36
終點：       (3,11) 狀態 47
懸崖：       (3,1-10) 狀態 37-46
回合：       500 回合/次
運行：       50 次獨立運行

超參數：
├─ α (學習率)：0.1
├─ γ (折扣):   0.9
├─ ε (探索):   0.1
└─ 更新：      贏者通吃 (TD)
```

---

## 🧮 統計分析方法

```python
# 50 次運行的結果聚合
results = np.array([run1, run2, ..., run50])  # shape: (50, 500)

# 計算平均曲線
avg_curve = np.mean(results, axis=0)          # shape: (500,)

# 最後 50 回合統計
final_50 = avg_curve[-50:]
mean = np.mean(final_50)
std = np.std(final_50)
max_val = np.max(final_50)
min_val = np.min(final_50)

# 信心區間
ci_95 = 1.96 * std / np.sqrt(50)
```

---

## 🔍 檢查清單

執行 `integrated_study.py` 確認有以下輸出：

- [ ] 50 次實驗完整運行日誌
- [ ] 最終統計結果表格
- [ ] 結論分析（SARSA 優勢）
- [ ] `qlearning_vs_sarsa_comparison.png` 生成
- [ ] 圖表包含 4 個子圖
- [ ] 無 Python 錯誤信息

---

## 📚 深入學習推薦

### 基礎
1. 馬可夫決策過程 (MDP)
2. 貝爾曼方程
3. 時間差分 (TD) 學習

### 進階
1. Deep Q-Learning (DQN)
2. 策略梯度方法 (Policy Gradient)
3. Actor-Critic 方法

### 實踐
1. 在更複雜環境測試
2. 超參數敏感性分析
3. 多進程並行實驗

---

**速記記住：**
> Q-Learning 樂觀但冒險  
> SARSA 謹慎但穩定  
> 在懸崖環境，謹慎勝過冒險！

🎓 **學生：Fulong (5114056035)**  
📅 **日期：2026-05-04**  
🏫 **課程：強化學習**
