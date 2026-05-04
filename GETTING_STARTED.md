# 🚀 Q-Learning vs SARSA 完整代碼整合 - 快速啟動指南

## 📌 核心內容

您已經準備好一個**完整的強化學習研究項目**，包括：

### ✅ 所有必要元素

| 類別 | 檔案 | 功能 |
|------|------|------|
| **主程序** | `integrated_study.py` | 完整整合版本 (推薦!) |
| **算法實現** | Q-Learning + SARSA | 兩種 TD 算法 |
| **實驗框架** | AlgorithmComparison | 50 次獨立運行 |
| **可視化** | 4 個 PNG 圖表 | 全面的結果展示 |
| **文檔** | 4 個詳細指南 | 完整的說明 |

---

## 🎯 立即開始 (3 行命令)

```bash
# 1️⃣ 進入專案目錄
cd /Users/brainshi/Desktop/強化學習/H2

# 2️⃣ 運行完整實驗 (推薦!)
python integrated_study.py

# 3️⃣ 查看結果
open qlearning_vs_sarsa_comparison.png
```

---

## 📖 各檔案說明

### 🎯 主要程序 (選一個運行)

#### 1️⃣ `integrated_study.py` ⭐ **推薦**
- **特點：** 完全模組化、有詳細註解
- **包含：** 6 個核心類別 + 所有功能
- **輸出：** 算法比較圖表 + 詳細報告
- **時間：** ~2-3 分鐘
- **使用場景：** 作業提交、論文寫作

```python
# 完整的類別結構
EnvironmentConfig → 配置管理
QLearningAgent → Q-Learning 實現
SARSAAgent → SARSA 實現
AlgorithmComparison → 實驗框架
plot_results() → 可視化
```

#### 2️⃣ `comparison_study.py`
- **特點：** 中文標籤版本
- **使用場景：** 中文論文、教學演示

#### 3️⃣ `comparison_study_english.py`
- **特點：** 純英文版本
- **使用場景：** 英文論文、國際會議

#### 4️⃣ `visualize_cliff_environment.py`
- **特點：** 環境策略可視化
- **輸出：** 3 個額外的 PNG 圖表
- **用途：** 展示學習到的策略

#### 5️⃣ `test_comparison.py`
- **特點：** 快速測試版本 (只運行 3 次)
- **時間：** ~30 秒
- **用途：** 驗證環境正常

---

## 📊 生成的可視化

運行後會生成 **4 個 PNG 圖表**：

### 圖表 1: `qlearning_vs_sarsa_comparison.png` (349 KB)
```
4 子圖對比：
┌─────────────────────────────────────┐
│ 學習曲線  │ 移動平均  │             │
├─────────────────────────────────────┤
│ 性能指標  │ 獎勵分布  │             │
└─────────────────────────────────────┘
```

### 圖表 2: `cliff_walking_policies.png` (425 KB)
```
6 子圖完整對比：
┌────────────────┬────────────────┐
│ Q-Learning     │ SARSA          │
├────────────────┼────────────────┤
│ 環境佈局       │ 環境佈局       │
│ 學習策略       │ 學習策略       │
│ 價值函數       │ 價值函數       │
└────────────────┴────────────────┘
```

### 圖表 3 & 4: 策略和價值函數對比
- 展示算法的決策差異
- 可視化價值分布

---

## 📚 文檔說明

### 4 份完整文檔

| 文檔 | 內容 | 何時查看 |
|------|------|--------|
| **QUICK_REFERENCE.md** | 30 秒快速對比 | 需要快速理解 |
| **CODE_INTEGRATION_GUIDE.md** | 代碼架構詳解 | 想修改代碼 |
| **VISUALIZATION_GUIDE.md** | 圖表說明 | 不懂圖表 |
| **PROJECT_SUMMARY.md** | 完整研究總結 | 寫論文 |

---

## 🔧 常見操作

### 1️⃣ 改變運行次數
```python
# 從 50 次改為 100 次
config = EnvironmentConfig(num_runs=100)
```

### 2️⃣ 調整超參數
```python
config.learning_rate = 0.05      # 更小的步長
config.epsilon = 0.05            # 更少的探索
config.episodes = 1000           # 更多回合
```

### 3️⃣ 只快速測試
```bash
python test_comparison.py  # 3 次運行，30 秒完成
```

### 4️⃣ 生成環境可視化
```bash
python visualize_cliff_environment.py
```

---

## 📊 實驗結果速查

### 一句話結論
> **SARSA 在懸崖環境中性能優於 Q-Learning**

### 數字對比
```
Q-Learning  SARSA      改善
─────────────────────────────
-49.17      -23.56  → +25.61 ✓
  9.04       2.86   → -68.4% ✓
 -29.02     -18.44  → +10.58 ✓
```

### 策略差異
```
Q-Learning: 沿懸崖邊 (13 步, 高風險)
SARSA:      上方安全 (15 步, 低風險)
```

---

## 🎓 代碼結構速覽

### 完整類別架構
```
integrated_study.py
├── EnvironmentConfig
│   ├── 環境參數 (48 狀態, 4 動作)
│   └── 超參數 (α=0.1, γ=0.9, ε=0.1)
│
├── QLearningAgent
│   ├── select_action() → ε-貪心
│   ├── update() → max Q(s',a')
│   └── train() → 500 回合循環
│
├── SARSAAgent
│   ├── select_action() → ε-貪心
│   ├── update() → Q(s',a')
│   └── train() → 500 回合循環
│
├── AlgorithmComparison
│   ├── run_experiments() → 50 次
│   ├── analyze_results() → 統計
│   └── print_comparison_results() → 報告
│
└── plot_results() → 4 子圖可視化
```

---

## ⏱️ 運行時間參考

| 程序 | 運行時間 | 用途 |
|------|--------|------|
| test_comparison.py | ~30 秒 | 驗證環境 |
| comparison_study_english.py | ~2 分鐘 | 完整測試 |
| integrated_study.py | ~2-3 分鐘 | 完整實驗 |
| visualize_cliff_environment.py | ~1 分鐘 | 策略可視化 |

---

## ✨ 特色功能

### 1️⃣ 完整的 TD 算法實現
```python
# Q-Learning: 離策略
max_next_q = np.max(Q[next_state])

# SARSA: 在策略
next_q = Q[next_state][next_action]
```

### 2️⃣ 統計分析框架
- 50 次獨立運行
- 平均曲線計算
- 標準差、最大值、最小值

### 3️⃣ 專業級可視化
- 4 個信息豐富的圖表
- 清晰的對比分析
- 高分辨率 PNG (300 DPI)

### 4️⃣ 詳細文檔
- 算法原理解釋
- 代碼註解完整
- 使用案例豐富

---

## 🎯 使用場景

### 📝 作業提交
1. 執行 `integrated_study.py`
2. 準備 4 個 PNG 圖表
3. 參考 `PROJECT_SUMMARY.md`
4. 完成作業報告

### 🔬 論文寫作
1. 參考 `DETAILED_ANALYSIS.md`
2. 使用可視化圖表
3. 引用實驗數據
4. 討論結果發現

### 👨‍🏫 教學演示
1. 運行 `test_comparison.py` (快速)
2. 展示 `comparison_study_english.py` (English)
3. 解釋 `QUICK_REFERENCE.md`
4. 演示策略可視化

### 🔧 代碼修改
1. 參考 `CODE_INTEGRATION_GUIDE.md`
2. 在 `integrated_study.py` 中修改
3. 理解類別結構
4. 調整超參數

---

## 🔍 故障排除

### 問題：運行時出現 ImportError
**解決：**
```bash
# 檢查依賴
pip install gymnasium numpy matplotlib

# 或使用虛擬環境
source .venv/bin/activate
```

### 問題：中文標籤顯示不正確
**解決：** 使用英文版本
```bash
python comparison_study_english.py
# 或 integrated_study.py (已優化)
```

### 問題：圖表生成但未顯示
**解決：** 檢查 PNG 檔案
```bash
ls -lh *.png
# 應該看到 4 個 PNG 檔案
```

---

## 📞 快速幫助

**Q: 要修改實驗參數怎麼做？**
A: 編輯 `EnvironmentConfig(num_runs=50)` 中的數值

**Q: 如何獲取具體代理的策略？**
A: 使用 `agent.get_learned_policy()` 方法

**Q: 可以在其他環境運行嗎？**
A: 是的，修改 `config.env_name` 即可

**Q: 代碼有多少行？**
A: 2194 行 Python 代碼 + 文檔

**Q: 可以並行運行嗎？**
A: 可以，在迴圈中使用 multiprocessing

---

## 🚀 一鍵運行

### 快速體驗（30 秒）
```bash
cd /Users/brainshi/Desktop/強化學習/H2
python test_comparison.py
```

### 完整實驗（3 分鐘）
```bash
cd /Users/brainshi/Desktop/強化學習/H2
python integrated_study.py
open qlearning_vs_sarsa_comparison.png
```

### 全部可視化（5 分鐘）
```bash
cd /Users/brainshi/Desktop/強化學習/H2
python integrated_study.py
python visualize_cliff_environment.py
open *.png
```

---

## ✅ 最後檢查清單

在提交前確認：

- [ ] `integrated_study.py` 能成功運行
- [ ] 生成了 4 個 PNG 圖表
- [ ] 查看了 `PROJECT_SUMMARY.md`
- [ ] 理解了 Q-Learning vs SARSA 差異
- [ ] 準備好文檔和圖表

---

## 📌 記住這 3 點

1. **核心發現：** SARSA 在懸崖環境優於 Q-Learning (高 25.61 分)
2. **原因：** Q-Learning 樂觀導致頻繁掉崖，SARSA 保守更穩定
3. **應用：** 安全環境用 SARSA，離線環境用 Q-Learning

---

**祝您研究順利！🎓**

需要幫助？查看相應的文檔：
- 快速理解 → `QUICK_REFERENCE.md`
- 代碼問題 → `CODE_INTEGRATION_GUIDE.md`
- 圖表說明 → `VISUALIZATION_GUIDE.md`
- 完整信息 → `PROJECT_SUMMARY.md`

---

**更新時間：2026-05-04**  
**狀態：✅ 所有功能完成**  
**準備狀態：✅ 可提交**
