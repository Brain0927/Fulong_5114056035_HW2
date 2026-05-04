# 📚 上傳到 GitHub 指南

這是一份快速指南，幫助你將本項目上傳到 GitHub。

## 步驟 1：在 GitHub 上建立新倉庫

1. 登入 [GitHub](https://github.com)
2. 點擊右上角的 **+** → **New repository**
3. 填寫倉庫信息：
   - **Repository name**: `Cliff-Walking-RL`
   - **Description**: `Q-learning vs SARSA comparison on Cliff Walking environment`
   - **Visibility**: Public（如果想分享）或 Private
   - **Initialize repository**: ❌ 不選（我們已有本地倉庫）

4. 點擊 **Create repository**

## 步驟 2：連接遠程倉庫

複製你在 GitHub 上建立的倉庫 URL（形如：`https://github.com/YOUR_USERNAME/Cliff-Walking-RL.git`）

然後在終端執行：

```bash
cd /Users/brainshi/Desktop/強化學習/H2

# 添加遠程倉庫
git remote add origin https://github.com/YOUR_USERNAME/Cliff-Walking-RL.git

# 驗證連接
git remote -v
```

## 步驟 3：上傳到 GitHub

```bash
# 推送到 GitHub
git push -u origin main

# 如果是第一次，可能需要認證
# GitHub 會提示你輸入個人訪問令牌 (Personal Access Token)
```

### 生成 GitHub Personal Access Token（如果需要）

1. 進入 [GitHub Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens)
2. 點擊 **Generate new token**
3. 選擇作用域：
   - ✅ repo
   - ✅ read:user
   - ✅ user:email
4. 複製生成的令牌
5. 在 Git 推送時，使用此令牌作為密碼

## 步驟 4：驗證上傳成功

訪問 `https://github.com/YOUR_USERNAME/Cliff-Walking-RL` 驗證文件是否已上傳。

你應該看到：
- ✅ `README.md` - 項目文檔
- ✅ `REPORT.md` - 詳細實驗報告
- ✅ `cliff_walking.py` - 主實現
- ✅ `visualize_qlearning.py` - 可視化工具
- ✅ 圖表 PNG 文件

---

## 🔄 後續更新

如果後續需要更新代碼：

```bash
# 做出更改後
git add .
git commit -m "Update message here"
git push origin main
```

---

## 📝 建議添加的文件

### .gitignore

建立 `.gitignore` 文件來忽略不必要的文件：

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.AppleDouble
.LSOverride

# Jupyter
.ipynb_checkpoints/

# Data
*.pkl
*.pickle
*.npy
EOF
```

### requirements.txt

```bash
cat > requirements.txt << 'EOF'
gymnasium>=0.26.0
numpy>=1.20.0
matplotlib>=3.5.0
EOF
```

## 🎯 最終檢查清單

- ✅ 本地 Git 倉庫初始化
- ✅ 所有代碼和報告文件已添加
- ✅ 初始提交已完成
- ✅ 遠程倉庫已在 GitHub 建立
- ✅ 本地倉庫已連接到遠程
- ✅ 代碼已推送到 GitHub
- ✅ README.md 可在 GitHub 上正確顯示
- ✅ 所有圖表 PNG 文件已上傳

---

## 💡 提示

- GitHub 支持 Markdown 格式，所以 README.md 會自動渲染成美觀的頁面
- 如果 README.md 包含中文，GitHub 會自動使用 UTF-8 編碼
- 推薦添加 `.gitignore` 以避免上傳虛擬環境等大文件

---

## 📞 遇到問題？

常見問題：

1. **認證失敗**
   - 確認 GitHub 用戶名和密碼
   - 或使用 Personal Access Token

2. **Remote already exists**
   ```bash
   git remote remove origin
   git remote add origin [新URL]
   ```

3. **推送被拒絕**
   ```bash
   git pull origin main --allow-unrelated-histories
   git push -u origin main
   ```

---

祝上傳順利！ 🚀
