# 📚 Q-Learning 與 SARSA 演算法之比較研究 - 完整總結

**學生：** Fulong (ID: 5114056035)  
**日期：** 2026年5月4日  
**課程：** 強化學習  
**環境：** Gymnasium CliffWalking-v1  

---

## 🎯 研究目標

本研究實現並詳細比較了強化學習中兩種核心的時間差分（Temporal Difference）演算法：

1. **Q-Learning** - 離策略（Off-policy）演算法
2. **SARSA** - 在策略（On-policy）演算法

通過在懸崖行走環境中進行 50 次獨立實驗，收集統計數據，分析其性能差異、收斂特性和實用應用場景。

---

## 📊 研究架構

### 實驗設計
```
懸崖行走環境 (4×12 網格)
    ↓
50 次獨立運行
    ↓
├─ Q-Learning 代理 (25 次結果)
├─ SARSA 代理 (25 次結果)
    ↓
統計分析
    ↓
└─ 可視化與比較報告
```

### 環境參數
- **狀態空間：** 48 個狀態 (4×12 網格)
- **動作空間：** 4 個動作 (UP, RIGHT, DOWN, LEFT)
- **獎勵結構：** 步驟 -1，懸崖 -100，目標 0
- **起點：** (3,0)，終點：(3,11)，懸崖：(3,1-10)

### 超參數配置
| 參數 | 值 | 說明 |
|------|-----|------|
| 學習率 (α) | 0.1 | Q值更新步長 |
| 折扣因子 (γ) | 0.9 | 未來獎勵權重 |
| 探索率 (ε) | 0.1 | 隨機動作概率 |
| 回合數 | 500 | 每次運行 |
| 運行次數 | 50 | 獨立實驗 |

---

## 🧠 演算法實現

### Q-Learning (離策略)

**更新規則：**
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

**核心特徵：**
- 使用下一狀態的 **最大 Q 值**
- 學習 **最優策略** (Q*)
- **樂觀估計**：假設未來最好的行為

**代碼實現：**
```python
max_next_q = np.max(Q[next_state]) if not terminated else 0
td_error = reward + gamma * max_next_q - Q[state][action]
Q[state][action] += alpha * td_error
```

**優缺點：**
| 優點 | 缺點 |
|------|------|
| 收斂到最優值函數 Q* | 訓練過程不穩定 |
| 收斂速度快 | 容易過度樂觀 |
| 理論保證收斂 | 在線應用風險高 |

---

### SARSA (在策略)

**更新規則：**
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]$$

**核心特徵：**
- 使用實際執行的 **下一動作 Q 值**
- 學習 **當前策略** (Q^π)
- **保守估計**：考慮實際探索的風險

**代碼實現：**
```python
next_action = select_action(next_state)  # 實際執行的動作
next_q = Q[next_state][next_action] if not terminated else 0
td_error = reward + gamma * next_q - Q[state][action]
Q[state][action] += alpha * td_error
```

**優缺點：**
| 優點 | 缺點 |
|------|------|
| 訓練過程穩定 | 收斂到子最優值 |
| 在線應用安全 | 收斂速度慢 |
| 對現實友好 | 性能依賴探索 |

---

## 📈 實驗結果

### 最終統計 (50 次運行，最後 50 回合平均)

```
┌─────────────────────┬──────────────┬──────────────┬──────────┐
│      指標          │ Q-Learning   │   SARSA      │  優勝者  │
├─────────────────────┼──────────────┼──────────────┼──────────┤
│ 平均獎勵           │   -49.17     │   -23.56     │ ⭐ SARSA │
│ 標準差             │    9.04      │    2.86      │ ⭐ SARSA │
│ 最大獎勵           │   -29.02     │   -18.44     │ ⭐ SARSA │
│ 最小獎勵           │   -67.68     │   -31.32     │ ⭐ SARSA │
├─────────────────────┼──────────────┼──────────────┼──────────┤
│ 性能差異           │   -25.61 分  │   基準       │  109%    │
│ 穩定性改善         │   基準       │   -68.4%     │  穩定    │
└─────────────────────┴──────────────┴──────────────┴──────────┘
```

### 關鍵數據

1. **平均獎勵：** SARSA 優於 Q-Learning 25.61 分
   - Q-Learning: -49.17
   - SARSA: -23.56
   - 相對改善：109% ✓

2. **穩定性：** SARSA 更穩定
   - Q-Learning 標準差：9.04
   - SARSA 標準差：2.86
   - 穩定性提升：68.4% ✓

3. **極值範圍：** SARSA 風險更低
   - Q-Learning 最小值：-67.68 (可能掉崖)
   - SARSA 最小值：-31.32 (更安全)

---

## 🎯 策略對比

### 學習到的路線

**Q-Learning 策略：**
```
S → → → → → → → → → ↓ G
  └─────────────────┘
     沿著懸崖邊緣
     路徑：13 步
     風險：極高
     特點：最優但危險
```

**SARSA 策略：**
```
S → ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↓ G
    ├─────────────────┤
         上方安全路線
     路徑：15-17 步
     風險：低
     特點：次優但安全
```

### 為何不同？

1. **Q-Learning (樂觀)：**
   - 計算：`max Q(s',a')` = -13（假設最好的）
   - 實際：經常掉崖 = -100
   - 結果：訓練不穩定，但最終找到最優路

2. **SARSA (保守)：**
   - 計算：`Q(s',a')` = 實際執行的動作
   - 實際：避免懸崖，走安全路
   - 結果：訓練穩定，找到次優路

---

## 📊 可視化結果

### 生成的圖表

| 檔案名 | 大小 | 內容 |
|--------|------|------|
| qlearning_vs_sarsa_comparison.png | 349 KB | 4 子圖對比 |
| cliff_walking_policies.png | 425 KB | 6 子圖政策可視化 |
| cliff_walking_policy_comparison.png | 104 KB | 策略箭頭對比 |
| cliff_walking_value_comparison.png | 192 KB | 價值函數熱力圖 |

### 子圖 1：學習曲線
- X軸：回合數 (0-500)
- Y軸：平均獎勵
- 紅線：Q-Learning (波動大)
- 藍線：SARSA (相對平穩)

### 子圖 2：50 回合移動平均
- 平滑化獎勵曲線
- 展示長期趨勢
- SARSA 更穩定上升

### 子圖 3：性能指標柱狀圖
- 比較平均、標準差、最大值
- 直觀看出 SARSA 優勢
- SARSA 在所有指標勝出

### 子圖 4：獎勵分布箱線圖
- 顯示最後 50 回合的分布
- Q-Learning 箱線寬大
- SARSA 箱線緊湊

---

## 💡 主要發現

### 1️⃣ 算法收斂性
- **Q-Learning：** 約 100 回合後收斂，但波動大
- **SARSA：** 約 150 回合後收斂，波動小
- **結論：** Q-Learning 收斂快但不穩定

### 2️⃣ 環境適應性
- **懸崖特性：** 高風險區域，樂觀估計導致失敗
- **Q-Learning 反應：** 樂觀導致頻繁掉崖 (-100 罰分)
- **SARSA 反應：** 謹慎避免危險區域
- **結論：** 懸崖環境更適合保守算法

### 3️⃣ 性能與安全的權衡
```
        性能
        ↑
        │
    QL →│    最優路線 (13 步)
        │      ×
        │    ╱
        │  ╱  非帕累托前沿
        │╱
────────┼──→ 安全性
        │
        │
   SARSA│    次優安全路線 (15 步)
        │         ×
        │ 帕累托優
```

### 4️⃣ 現實應用啟示
- **離線學習環境：** 使用 Q-Learning (追求最優)
- **在線安全環境：** 使用 SARSA (追求穩定)
- **本懸崖環境：** SARSA 更適合

---

## 📁 項目文件清單

### Python 源代碼
```
✅ integrated_study.py (19 KB)
   - 【推薦主程序】完整整合實現
   - 包含所有類別和分析

✅ comparison_study.py (25 KB)
   - 原始版本，支持中文標籤

✅ comparison_study_english.py (16 KB)
   - 純英文版本

✅ visualize_cliff_environment.py (11 KB)
   - 環境與策略可視化

✅ test_comparison.py (7.7 KB)
   - 快速測試版本 (3 次運行)
```

### 可視化輸出
```
✅ qlearning_vs_sarsa_comparison.png (349 KB)
✅ cliff_walking_policies.png (425 KB)
✅ cliff_walking_policy_comparison.png (104 KB)
✅ cliff_walking_value_comparison.png (192 KB)
```

### 文檔說明
```
✅ CODE_INTEGRATION_GUIDE.md (9.5 KB)
   - 代碼架構詳細說明

✅ QUICK_REFERENCE.md (7.3 KB)
   - 快速參考卡片

✅ VISUALIZATION_GUIDE.md (7.4 KB)
   - 可視化說明書

✅ README.md
   - 快速開始指南

✅ DETAILED_ANALYSIS.md
   - 深度理論分析

✅ PROJECT_SUMMARY.md (本文件)
   - 完整項目總結
```

---

## 🚀 使用指南

### 快速開始
```bash
# 運行完整實驗
python integrated_study.py

# 運行環境可視化
python visualize_cliff_environment.py

# 快速測試 (3 次運行)
python test_comparison.py
```

### 修改實驗
```python
# 改變運行次數
config = EnvironmentConfig(num_runs=100)

# 改變超參數
config.learning_rate = 0.05      # 更小的步長
config.epsilon = 0.05             # 更少的探索
config.episodes = 1000            # 更多回合
```

### 獲取結果數據
```python
# 訪問原始數據
results = comparator.analyze_results()
ql_rewards = results['qlearning']['rewards']
sarsa_rewards = results['sarsa']['rewards']

# 獲取代理政策
ql_policy = comparator.ql_agents[0].get_learned_policy()
sarsa_policy = comparator.sarsa_agents[0].get_learned_policy()
```

---

## 🎓 學習要點

### 概念理解
1. **離策略 vs 在策略**
   - 離策略：學習目標策略，執行行為策略
   - 在策略：學習執行策略本身

2. **貝爾曼方程**
   - 現在值 = 立即獎勵 + 折扣未來值
   - Q-Learning 和 SARSA 的基礎

3. **ε-貪心探索**
   - 平衡探索和利用
   - 10% 隨機，90% 最優動作

### 實踐技能
1. 實現 TD 學習算法
2. 設計實驗比較框架
3. 統計分析和可視化
4. 環境與代理交互

---

## 📚 延伸資源

### 理論深化
- **Expected SARSA**：結合 Q-Learning 的期望值更新
- **Double Q-Learning**：解決高估問題
- **Dueling Architecture**：分離價值和優勢估計

### 相關算法
- Deep Q-Learning (DQN)：處理大狀態空間
- Policy Gradient：直接優化策略
- Actor-Critic：結合兩者優點

### 實驗擴展
- 在其他 Gymnasium 環境測試
- 超參數敏感性分析
- 多進程並行實驗

---

## ✅ 完成清單

- [x] Q-Learning 算法實現 (完成度 100%)
- [x] SARSA 算法實現 (完成度 100%)
- [x] 50 次獨立實驗 (完成度 100%)
- [x] 統計分析 (完成度 100%)
- [x] 4 個視覺化圖表 (完成度 100%)
- [x] 策略與值函數可視化 (完成度 100%)
- [x] 代碼整合與優化 (完成度 100%)
- [x] 完整文檔 (完成度 100%)
- [x] 快速參考指南 (完成度 100%)
- [x] GitHub 推送準備 (完成度 100%)

**總完成度：100% ✅**

---

## 🎯 結論

### 研究發現
1. **SARSA 在懸崖環境中優於 Q-Learning**
   - 平均獎勵高 25.61 分
   - 穩定性提升 68.4%

2. **算法選擇應考慮環境特性**
   - 高風險環境：選擇 SARSA (保守)
   - 安全環境：可選 Q-Learning (最優)

3. **理論與實踐的差異**
   - 理論：Q-Learning 優，SARSA 次優
   - 實踐：環境特性決定勝負

### 實際應用建議
- **自動駕駛：** 使用 SARSA（安全第一）
- **遊戲 AI：** 使用 Q-Learning（追求勝利）
- **機器人控制：** 使用 SARSA（避免破損）
- **金融交易：** 使用 SARSA（控制風險）

---

## 📞 聯絡信息

**學生資訊**
- 姓名：Fulong
- 學號：5114056035
- 課程：強化學習
- 日期：2026年5月4日

**項目信息**
- 主題：Q-Learning 與 SARSA 演算法之比較研究
- 環境：Gymnasium CliffWalking-v1
- 實驗次數：50 次獨立運行
- 代碼行數：2000+ 行

---

**最後更新：2026-05-04**  
**代碼狀態：✅ 完成並測試**  
**文檔狀態：✅ 完整且詳細**  
**項目狀態：✅ 已就緒提交**
