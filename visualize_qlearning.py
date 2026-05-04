import numpy as np
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# 設定中文字體支援
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        初始化 Q-learning 代理 (Off-policy)
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        
    def epsilon_greedy(self, state):
        """ε-greedy 動作選擇"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])
    
    def train(self, episodes=500):
        """訓練 Q-learning 代理"""
        rewards_history = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.epsilon_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Q-learning 更新：使用最優的下一個動作
                self.Q[state][action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
                )
                
                state = next_state
            
            rewards_history.append(episode_reward)
            
            if (episode + 1) % 100 == 0:
                print(f"Q-learning - 第 {episode + 1} 回合 | 累積獎勵: {episode_reward}")
        
        return rewards_history
    
    def get_policy(self):
        """提取貪心策略"""
        policy = {}
        for state in self.Q.keys():
            policy[state] = np.argmax(self.Q[state])
        return policy
    
    def get_state_values(self):
        """獲取狀態價值函數"""
        state_values = {}
        for state in self.Q.keys():
            state_values[state] = np.max(self.Q[state])
        return state_values


def visualize_qlearning_policy(agent):
    """視覺化 Q-learning 最終策略"""
    arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    policy = agent.get_policy()
    
    # 4x12 的 Cliff Walking 環境
    grid = np.full((4, 12), '·', dtype=object)
    
    for state in range(48):
        row, col = state // 12, state % 12
        if state in policy:
            grid[row, col] = arrows[policy[state]]
        else:
            grid[row, col] = '·'
    
    # 標記特殊位置
    grid[3, 0] = 'S'   # 起點
    grid[3, 11] = 'G'  # 終點
    grid[3, 1:11] = 'C' # 懸崖
    
    print("\n" + "="*60)
    print("Q-learning Final Policy (Off-policy, Risk-taker)")
    print("="*60)
    for i, row in enumerate(grid):
        print(' '.join(row))
    print("\nLegend: S=Start | G=Goal | C=Cliff | ↑↓←→=Recommended Action")


def visualize_qlearning_values(agent):
    """視覺化 Q-learning 狀態價值函數（熱力圖）"""
    state_values = agent.get_state_values()
    
    # 建立 4x12 的價值網格
    value_grid = np.zeros((4, 12))
    
    for state in range(48):
        row, col = state // 12, state % 12
        if state in state_values:
            value_grid[row, col] = state_values[state]
        else:
            value_grid[row, col] = 0
    
    # 懸崖位置設為特殊值
    value_grid[3, 1:11] = -100
    
    return value_grid


def create_comprehensive_visualization(agent, rewards_history):
    """建立完整的 Q-learning 視覺化圖表"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 標題
    fig.suptitle('Q-learning (Off-policy) Detailed Analysis - Cliff Walking', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # ========== 第一行 ==========
    
    # 1. 訓練曲線
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(rewards_history, alpha=0.5, color='red', linewidth=1, label='Reward per Episode')
    
    # 添加移動平均線
    window = 50
    ma = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    ax1.plot(range(window-1, len(rewards_history)), ma, color='darkred', 
             linewidth=2.5, label=f'Moving Average (window={window})')
    
    ax1.set_xlabel('Training Episode', fontsize=11)
    ax1.set_ylabel('Cumulative Reward', fontsize=11)
    ax1.set_title('Q-learning Training Curve', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=-13, color='green', linestyle='--', linewidth=2, label='Optimal Path Theory (-13)', alpha=0.7)
    
    # 2. 訓練統計
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    stats_text = f"""
Training Statistics

Mean Reward (Full)
{np.mean(rewards_history):.2f}

Mean Reward (Last 50)
{np.mean(rewards_history[-50:]):.2f}

Max Reward
{np.max(rewards_history):.2f}

Min Reward
{np.min(rewards_history):.2f}

Std Dev
{np.std(rewards_history):.2f}

Features:
• High Volatility
• Frequent Cliff Falls
• Seeking Shortest Path
    """
    ax2.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')
    
    # ========== 第二行 ==========
    
    # 3. 狀態價值熱力圖
    ax3 = fig.add_subplot(gs[1, 0])
    value_grid = visualize_qlearning_values(agent)
    
    # 創建特殊的顏色映射
    im = ax3.imshow(value_grid, cmap='RdYlGn', aspect='auto', vmin=-100, vmax=0)
    ax3.set_xlabel('Column Position', fontsize=10)
    ax3.set_ylabel('Row Position', fontsize=10)
    ax3.set_title('State Value Function V(s)', fontsize=12, fontweight='bold')
    
    # 添加網格線
    for i in range(5):
        ax3.axhline(i-0.5, color='black', linewidth=0.5)
    for j in range(13):
        ax3.axvline(j-0.5, color='black', linewidth=0.5)
    
    # 標記特殊位置
    ax3.text(0, 3, 'S', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax3.text(11, 3, 'G', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    plt.colorbar(im, ax=ax3, label='Value V(s)')
    
    # 4. 策略網格 - 箭頭視圖
    ax4 = fig.add_subplot(gs[1, 1:])
    policy = agent.get_policy()
    arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    # 繪製策略網格
    for state in range(48):
        row, col = state // 12, state % 12
        
        # 繪製單元格背景
        if state in range(1, 11) and row == 3:  # 懸崖
            color = 'red'
            alpha = 0.3
        elif state == 0:  # 起點
            color = 'green'
            alpha = 0.3
        elif state == 47:  # 終點
            color = 'blue'
            alpha = 0.3
        else:
            color = 'lightgray'
            alpha = 0.1
        
        rect = mpatches.Rectangle((col-0.4, 3-row-0.4), 0.8, 0.8, 
                                  linewidth=1, edgecolor='black', 
                                  facecolor=color, alpha=alpha)
        ax4.add_patch(rect)
        
        # 添加箭頭或標記
        if state in policy:
            ax4.text(col, 3-row, arrows[policy[state]], 
                    ha='center', va='center', fontsize=14, fontweight='bold')
        
        # 特殊標記
        if state == 0:
            ax4.text(col, 3-row, 'S', ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='white')
        elif state == 47:
            ax4.text(col, 3-row, 'G', ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='white')
        elif state in range(1, 11) and row == 3:
            ax4.text(col, 3-row, 'C', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
    
    ax4.set_xlim(-0.5, 11.5)
    ax4.set_ylim(-0.5, 3.5)
    ax4.set_aspect('equal')
    ax4.invert_yaxis()
    ax4.set_xlabel('Column Position', fontsize=10)
    ax4.set_ylabel('Row Position', fontsize=10)
    ax4.set_title('Q-learning Final Policy (Off-policy, Risk-taker)', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(12))
    ax4.set_yticks(range(4))
    
    # ========== 第三行 ==========
    
    # 5. 獎勵分佈直方圖
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(rewards_history, bins=30, color='red', alpha=0.7, edgecolor='darkred')
    ax5.axvline(np.mean(rewards_history), color='darkred', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(rewards_history):.2f}')
    ax5.axvline(np.median(rewards_history), color='blue', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(rewards_history):.2f}')
    ax5.set_xlabel('Reward Value', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 回合獎勵趨勢（分段平均）
    ax6 = fig.add_subplot(gs[2, 1])
    segment_size = 50
    segments = len(rewards_history) // segment_size
    segment_means = [np.mean(rewards_history[i*segment_size:(i+1)*segment_size]) 
                    for i in range(segments)]
    segment_x = [i*segment_size + segment_size//2 for i in range(len(segment_means))]
    
    ax6.plot(segment_x, segment_means, 'o-', color='red', linewidth=2, markersize=6)
    ax6.fill_between(segment_x, segment_means, alpha=0.3, color='red')
    ax6.set_xlabel('Training Progress', fontsize=10)
    ax6.set_ylabel('Average Reward', fontsize=10)
    ax6.set_title('Segment Mean Reward Trend', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. 策略特性分析
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    # 分析路徑特性
    policy = agent.get_policy()
    policy_list = [policy.get(i, 0) for i in range(48)]
    
    analysis_text = f"""
Q-learning Policy Analysis

Action Distribution:
UP: {policy_list.count(0)}
DOWN: {policy_list.count(1)}
LEFT: {policy_list.count(2)}
RIGHT: {policy_list.count(3)}

Strategy Style:
Risk-taker (Off-policy)
• Seek Shortest Path
• Ignore Exploration Risk
• With ε-greedy,
  Falls off Cliff Often

Theoretical Path Length:
13 Steps (Optimal)

Actual Performance:
High Volatility
Unstable
    """
    
    ax7.text(0.05, 0.5, analysis_text, fontsize=9, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3),
             family='monospace')
    
    plt.savefig('/Users/brainshi/Desktop/強化學習/H2/qlearning_detailed_analysis.png', 
               dpi=200, bbox_inches='tight')
    print("\n✅ Detailed Analysis Chart Saved: qlearning_detailed_analysis.png")


def create_policy_grid_detailed(agent):
    """建立詳細的策略網格圖"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    policy = agent.get_policy()
    state_values = agent.get_state_values()
    
    arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
    colors_map = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1', 3: '#FFA07A'}
    
    for state in range(48):
        row, col = state // 12, state % 12
        y = 3 - row  # 反轉 y 坐標
        
        # 決定背景顏色
        if state in range(1, 11) and row == 3:  # 懸崖
            bg_color = '#8B0000'
            bg_alpha = 0.8
            text_color = 'white'
        elif state == 0:  # 起點
            bg_color = '#228B22'
            bg_alpha = 0.8
            text_color = 'white'
        elif state == 47:  # 終點
            bg_color = '#4169E1'
            bg_alpha = 0.8
            text_color = 'white'
        else:
            bg_color = 'lightgray'
            bg_alpha = 0.3
            text_color = 'black'
        
        # 繪製單元格
        rect = mpatches.FancyBboxPatch((col-0.45, y-0.45), 0.9, 0.9,
                                       boxstyle="round,pad=0.05",
                                       linewidth=2, edgecolor='black',
                                       facecolor=bg_color, alpha=bg_alpha)
        ax.add_patch(rect)
        
        # 添加箭頭
        if state in policy:
            ax.text(col-0.2, y+0.15, arrows[policy[state]], 
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   color=text_color)
        
        # 添加狀態值（小字）
        if state in state_values and not (state in range(1, 11) and row == 3):
            value = state_values[state]
            value_text = f"{value:.1f}"
            ax.text(col+0.2, y-0.15, value_text, 
                   ha='center', va='center', fontsize=8, color='gray',
                   style='italic')
        
        # 特殊標記
        if state == 0:
            ax.text(col-0.2, y+0.15, 'S', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')
        elif state == 47:
            ax.text(col-0.2, y+0.15, 'G', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')
        elif state in range(1, 11) and row == 3:
            ax.text(col-0.2, y+0.15, 'C', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
    
    ax.set_xlim(-0.6, 11.6)
    ax.set_ylim(-0.6, 3.6)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    ax.set_xlabel('Column Position (Column)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Row Position (Row)', fontsize=12, fontweight='bold')
    ax.set_title('Q-learning Final Policy Grid (Off-policy - Risk-taker)\n' + 
                'S=Start | G=Goal | C=Cliff | ↑↓←→=Recommended Action', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticks(range(12))
    ax.set_yticks(range(4))
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # 添加圖例
    legend_elements = [
        mpatches.Patch(facecolor='#228B22', label='Start Point', alpha=0.8),
        mpatches.Patch(facecolor='#4169E1', label='Goal', alpha=0.8),
        mpatches.Patch(facecolor='#8B0000', label='Cliff', alpha=0.8),
        mpatches.Patch(facecolor='lightgray', label='Normal State', alpha=0.3),
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/Users/brainshi/Desktop/強化學習/H2/qlearning_policy_grid.png',
               dpi=200, bbox_inches='tight')
    print("✅ Policy Grid Chart Saved: qlearning_policy_grid.png")


def main():
    print("\n" + "="*70)
    print("Q-learning (Off-policy) Deep Analysis - Cliff Walking")
    print("="*70 + "\n")
    
    # 建立環境
    env = gym.make('CliffWalking-v1')
    
    # 訓練 Q-learning 代理
    print("Starting Q-learning Agent Training...\n")
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    rewards = agent.train(episodes=500)
    
    # 顯示最終策略（文本格式）
    visualize_qlearning_policy(agent)
    
    # 生成詳細分析圖表
    print("\nGenerating Detailed Analysis Chart...")
    create_comprehensive_visualization(agent, rewards)
    
    # 生成策略網格圖
    print("Generating Policy Grid Chart...")
    create_policy_grid_detailed(agent)
    
    # 生成價值函數熱力圖
    print("Generating Value Function Heatmap...")
    fig, ax = plt.subplots(figsize=(12, 5))
    
    value_grid = visualize_qlearning_values(agent)
    im = ax.imshow(value_grid, cmap='RdYlGn', aspect='auto', vmin=-100, vmax=0)
    
    ax.set_xlabel('Column Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Row Position', fontsize=12, fontweight='bold')
    ax.set_title('Q-learning State Value Function V(s) - Heatmap', 
                fontsize=14, fontweight='bold')
    
    # 添加網格線
    for i in range(5):
        ax.axhline(i-0.5, color='black', linewidth=1)
    for j in range(13):
        ax.axvline(j-0.5, color='black', linewidth=1)
    
    # 標記特殊位置文本
    ax.text(0, 3, 'S', ha='center', va='center', fontsize=12, 
           fontweight='bold', color='white', 
           bbox=dict(boxstyle='circle', facecolor='black', alpha=0.5))
    ax.text(11, 3, 'G', ha='center', va='center', fontsize=12, 
           fontweight='bold', color='white',
           bbox=dict(boxstyle='circle', facecolor='black', alpha=0.5))
    
    cbar = plt.colorbar(im, ax=ax, label='State Value V(s)')
    
    plt.tight_layout()
    plt.savefig('/Users/brainshi/Desktop/強化學習/H2/qlearning_value_heatmap.png',
               dpi=200, bbox_inches='tight')
    print("✅ Value Function Heatmap Saved: qlearning_value_heatmap.png")
    
    env.close()
    
    print("\n" + "="*70)
    print("✅ All Charts Generated Successfully!")
    print("="*70)
    print("\nGenerated Chart Files:")
    print("1. qlearning_detailed_analysis.png - Complete Analysis (6 subplots)")
    print("2. qlearning_policy_grid.png - Policy Grid Detail Chart")
    print("3. qlearning_value_heatmap.png - State Value Function Heatmap")


if __name__ == "__main__":
    main()
