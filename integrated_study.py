#!/usr/bin/env python
"""
Q-Learning 與 SARSA 演算法之比較研究
Comparative Study of Q-Learning and SARSA Algorithms on Cliff Walking

學生：Fulong (5114056035)
日期：2026年5月4日
環境：Gymnasium CliffWalking-v1 (4×12 網格懸崖環境)

本項目實現並比較了兩種重要的時間差分學習演算法：
1. Q-Learning：離策略（Off-policy）演算法，學習最優策略
2. SARSA：在策略（On-policy）演算法，學習當前策略

通過 50 次獨立實驗運行，收集統計數據並生成可視化結果。
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# 設置隨機種子以確保可重現性
np.random.seed(42)


# ============================================================================
# 第一部分：環境配置與基礎設置
# ============================================================================

class EnvironmentConfig:
    """
    懸崖行走環境配置類
    
    環境特性：
    - 4×12 網格 (48 個狀態)
    - 起點：(3,0)，終點：(3,11)
    - 懸崖區域：(3,1)-(3,10) 包含 10 個懸崖格子
    
    獎勵設置：
    - 普通步驟：-1
    - 掉入懸崖：-100
    - 到達目標：0
    """
    
    def __init__(self, num_runs: int = 50):
        """初始化環境配置"""
        # 環境參數
        self.env_name = "CliffWalking-v1"
        self.grid_size = (4, 12)
        self.num_states = 48
        self.num_actions = 4  # UP, RIGHT, DOWN, LEFT
        
        # 狀態位置定義
        self.start_state = 36      # (3, 0)
        self.goal_state = 47       # (3, 11)
        self.cliff_states = set(range(37, 47))  # (3, 1-10)
        
        # 獎勵設定
        self.step_reward = -1
        self.cliff_reward = -100
        self.goal_reward = 0
        
        # 超參數設定
        self.learning_rate = 0.1        # α：Q值更新步長
        self.discount_factor = 0.9      # γ：未來獎勵折扣因子
        self.epsilon = 0.1              # ε：ε-貪心探索率
        
        # 訓練參數
        self.episodes = 500             # 每次運行的回合數
        self.num_runs = num_runs        # 獨立運行次數


# ============================================================================
# 第二部分：Q-Learning 演算法實現
# ============================================================================

class QLearningAgent:
    """
    Q-Learning 代理實現
    
    演算法特徵：
    - 離策略（Off-policy）：學習最優策略，同時執行探索策略
    - Q值更新：Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
    - 關鍵：使用下一狀態的最大Q值（樂觀估計）
    
    優勢：
    - 收斂到最優值函數 Q*
    - 收斂速度快
    - 理論上保證收斂
    
    劣勢：
    - 訓練中容易掉懸崖
    - 在線學習時不安全
    - 對環境變化敏感
    """
    
    def __init__(self, config: EnvironmentConfig):
        """初始化 Q-Learning 代理"""
        self.config = config
        self.Q = defaultdict(lambda: np.zeros(config.num_actions))
        self.rewards_history = []
        self.episode_paths = []
    
    def select_action(self, state: int, training: bool = True) -> int:
        """
        ε-貪心動作選擇策略
        
        Args:
            state: 當前狀態
            training: 是否處於訓練模式（啟用探索）
            
        Returns:
            選擇的動作 (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        """
        if training and np.random.random() < self.config.epsilon:
            # 探索：隨機選擇 (10% 概率)
            return np.random.randint(0, self.config.num_actions)
        else:
            # 利用：選擇最優動作 (90% 概率)
            return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, terminated: bool) -> None:
        """
        Q-Learning 更新規則
        
        Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        
        Args:
            state: 當前狀態
            action: 執行的動作
            reward: 獲得的獎勵
            next_state: 下一狀態
            terminated: 是否終止
        """
        alpha = self.config.learning_rate
        gamma = self.config.discount_factor
        
        # Q-Learning 特徵：使用下一狀態的最大Q值
        max_next_q = 0.0 if terminated else np.max(self.Q[next_state])
        td_error = reward + gamma * max_next_q - self.Q[state][action]
        self.Q[state][action] += alpha * td_error
    
    def train(self, env: gym.Env) -> None:
        """
        訓練 Q-Learning 代理
        
        Args:
            env: Gymnasium 環境實例
        """
        self.rewards_history = []
        self.episode_paths = []
        
        for episode in range(self.config.episodes):
            state, _ = env.reset()
            total_reward = 0
            path_length = 0
            
            while True:
                # 選擇並執行動作
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                path_length += 1
                
                # 更新Q值
                self.update(state, action, reward, next_state, terminated or truncated)
                
                state = next_state
                if terminated or truncated:
                    break
            
            self.rewards_history.append(total_reward)
            self.episode_paths.append(path_length)
    
    def get_learned_policy(self) -> Dict[int, int]:
        """獲取學習到的策略"""
        policy = {}
        for state in range(self.config.num_states):
            if state in self.Q:
                policy[state] = np.argmax(self.Q[state])
            else:
                policy[state] = 0
        return policy


# ============================================================================
# 第三部分：SARSA 演算法實現
# ============================================================================

class SARSAAgent:
    """
    SARSA 代理實現
    
    演算法特徵：
    - 在策略（On-policy）：學習當前執行的策略
    - Q值更新：Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
    - 關鍵：使用實際執行的下一動作的Q值
    
    優勢：
    - 在線學習安全性高
    - 訓練過程穩定
    - 對現實環境友好
    
    劣勢：
    - 收斂到當前策略的值函數，非最優
    - 收斂速度較慢
    - 性能依賴於探索策略
    """
    
    def __init__(self, config: EnvironmentConfig):
        """初始化 SARSA 代理"""
        self.config = config
        self.Q = defaultdict(lambda: np.zeros(config.num_actions))
        self.rewards_history = []
        self.episode_paths = []
    
    def select_action(self, state: int, training: bool = True) -> int:
        """
        ε-貪心動作選擇策略
        
        Args:
            state: 當前狀態
            training: 是否處於訓練模式（啟用探索）
            
        Returns:
            選擇的動作 (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        """
        if training and np.random.random() < self.config.epsilon:
            # 探索：隨機選擇 (10% 概率)
            return np.random.randint(0, self.config.num_actions)
        else:
            # 利用：選擇最優動作 (90% 概率)
            return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, next_action: int, terminated: bool) -> None:
        """
        SARSA 更新規則
        
        Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
        
        Args:
            state: 當前狀態
            action: 執行的動作
            reward: 獲得的獎勵
            next_state: 下一狀態
            next_action: 下一狀態要執行的動作
            terminated: 是否終止
        """
        alpha = self.config.learning_rate
        gamma = self.config.discount_factor
        
        # SARSA 特徵：使用實際下一動作的Q值（而非最大值）
        next_q = 0.0 if terminated else self.Q[next_state][next_action]
        td_error = reward + gamma * next_q - self.Q[state][action]
        self.Q[state][action] += alpha * td_error
    
    def train(self, env: gym.Env) -> None:
        """
        訓練 SARSA 代理
        
        Args:
            env: Gymnasium 環境實例
        """
        self.rewards_history = []
        self.episode_paths = []
        
        for episode in range(self.config.episodes):
            state, _ = env.reset()
            action = self.select_action(state, training=True)
            total_reward = 0
            path_length = 0
            
            while True:
                # 執行動作
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                path_length += 1
                
                # 選擇下一動作
                next_action = self.select_action(next_state, training=True)
                
                # 更新Q值
                self.update(state, action, reward, next_state, next_action, 
                           terminated or truncated)
                
                state = next_state
                action = next_action
                if terminated or truncated:
                    break
            
            self.rewards_history.append(total_reward)
            self.episode_paths.append(path_length)
    
    def get_learned_policy(self) -> Dict[int, int]:
        """獲取學習到的策略"""
        policy = {}
        for state in range(self.config.num_states):
            if state in self.Q:
                policy[state] = np.argmax(self.Q[state])
            else:
                policy[state] = 0
        return policy


# ============================================================================
# 第四部分：實驗執行與統計分析
# ============================================================================

class AlgorithmComparison:
    """
    演算法比較研究類
    
    功能：
    - 執行多個獨立實驗運行
    - 收集統計數據
    - 生成比較報告
    - 可視化結果
    """
    
    def __init__(self, config: EnvironmentConfig):
        """初始化比較研究"""
        self.config = config
        self.env = gym.make(config.env_name)
        self.ql_results = []
        self.sarsa_results = []
        self.ql_agents = []
        self.sarsa_agents = []
    
    def run_experiments(self) -> None:
        """運行獨立實驗"""
        print("=" * 80)
        print("Q-Learning 與 SARSA 演算法之比較研究")
        print("=" * 80)
        print(f"\n配置參數：")
        print(f"  環境：{self.config.env_name}")
        print(f"  每次運行的回合數：{self.config.episodes}")
        print(f"  獨立運行次數：{self.config.num_runs}")
        print(f"  學習率 (α)：{self.config.learning_rate}")
        print(f"  折扣因子 (γ)：{self.config.discount_factor}")
        print(f"  探索率 (ε)：{self.config.epsilon}")
        print()
        
        for run in range(self.config.num_runs):
            print(f"實驗 {run + 1}/{self.config.num_runs}：", end=" ")
            
            # 訓練 Q-Learning 代理
            ql_agent = QLearningAgent(self.config)
            ql_agent.train(self.env)
            self.ql_results.append(ql_agent.rewards_history)
            self.ql_agents.append(ql_agent)
            ql_last_50 = np.mean(ql_agent.rewards_history[-50:])
            print(f"Q-Learning 平均={ql_last_50:6.2f}", end=" | ")
            
            # 訓練 SARSA 代理
            sarsa_agent = SARSAAgent(self.config)
            sarsa_agent.train(self.env)
            self.sarsa_results.append(sarsa_agent.rewards_history)
            self.sarsa_agents.append(sarsa_agent)
            sarsa_last_50 = np.mean(sarsa_agent.rewards_history[-50:])
            print(f"SARSA 平均={sarsa_last_50:6.2f}")
        
        self.env.close()
    
    def analyze_results(self) -> Dict:
        """分析實驗結果"""
        # 計算平均曲線
        ql_avg = np.mean(self.ql_results, axis=0)
        sarsa_avg = np.mean(self.sarsa_results, axis=0)
        
        # 最後 50 回合的數據
        ql_last_50 = ql_avg[-50:]
        sarsa_last_50 = sarsa_avg[-50:]
        
        results = {
            'qlearning': {
                'rewards': ql_avg,
                'mean': np.mean(ql_last_50),
                'std': np.std(ql_last_50),
                'max': np.max(ql_last_50),
                'min': np.min(ql_last_50),
            },
            'sarsa': {
                'rewards': sarsa_avg,
                'mean': np.mean(sarsa_last_50),
                'std': np.std(sarsa_last_50),
                'max': np.max(sarsa_last_50),
                'min': np.min(sarsa_last_50),
            }
        }
        
        return results
    
    def print_comparison_results(self) -> None:
        """打印詳細比較結果"""
        results = self.analyze_results()
        
        print("\n" + "=" * 80)
        print("實驗結果統計")
        print("=" * 80)
        
        print("\nQ-LEARNING (離策略演算法)：")
        print(f"  平均獎勵 (最後 50 回合)：{results['qlearning']['mean']:8.2f}")
        print(f"  標準差：                 {results['qlearning']['std']:8.2f}")
        print(f"  最大獎勵：               {results['qlearning']['max']:8.2f}")
        print(f"  最小獎勵：               {results['qlearning']['min']:8.2f}")
        
        print("\nSARSA (在策略演算法)：")
        print(f"  平均獎勵 (最後 50 回合)：{results['sarsa']['mean']:8.2f}")
        print(f"  標準差：                 {results['sarsa']['std']:8.2f}")
        print(f"  最大獎勵：               {results['sarsa']['max']:8.2f}")
        print(f"  最小獎勵：               {results['sarsa']['min']:8.2f}")
        
        # 計算差異
        mean_diff = results['qlearning']['mean'] - results['sarsa']['mean']
        std_diff = results['qlearning']['std'] - results['sarsa']['std']
        
        print("\n比較分析：")
        print(f"  Q-Learning 平均獎勵劣勢：{mean_diff:8.2f} 分")
        print(f"  Q-Learning 方差劣勢：    {std_diff:8.2f}")
        print(f"  SARSA 穩定性優勢：       {(std_diff/results['qlearning']['std']*100):8.1f}%")
        
        # 結論
        print("\n🎯 結論：")
        if results['sarsa']['mean'] > results['qlearning']['mean']:
            print(f"  ✓ SARSA 在懸崖環境中表現更優")
            print(f"  ✓ 原因：Q-Learning 的樂觀估計導致頻繁掉崖")
            print(f"  ✓ SARSA 採用更保守策略，避免高風險區域")
        else:
            print(f"  ✓ Q-Learning 找到更優的策略")
            print(f"  ✓ 但訓練過程不穩定")
        
        return results


# ============================================================================
# 第五部分：可視化與輸出
# ============================================================================

def plot_results(comparator: AlgorithmComparison) -> None:
    """生成比較圖表"""
    results = comparator.analyze_results()
    
    print("\n📊 正在生成圖表...\n")
    
    ql_rewards = results['qlearning']['rewards']
    sarsa_rewards = results['sarsa']['rewards']
    
    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q-Learning vs SARSA 演算法對比分析', fontsize=16, fontweight='bold')
    
    # 1. 累積獎勵曲線
    ax = axes[0, 0]
    episodes = range(1, len(ql_rewards) + 1)
    ax.plot(episodes, ql_rewards, label='Q-Learning', color='red', linewidth=2, alpha=0.7)
    ax.plot(episodes, sarsa_rewards, label='SARSA', color='blue', linewidth=2, alpha=0.7)
    ax.set_xlabel('Episodes (回合)', fontsize=10)
    ax.set_ylabel('Average Reward (平均獎勵)', fontsize=10)
    ax.set_title('Learning Curves Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 移動平均（50 回合）
    ax = axes[0, 1]
    window = 50
    ql_ma = np.convolve(ql_rewards, np.ones(window)/window, mode='valid')
    sarsa_ma = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
    episodes_ma = range(window, len(ql_rewards) + 1)
    ax.plot(episodes_ma, ql_ma, label='Q-Learning', color='red', linewidth=2)
    ax.plot(episodes_ma, sarsa_ma, label='SARSA', color='blue', linewidth=2)
    ax.set_xlabel('Episodes (回合)', fontsize=10)
    ax.set_ylabel('Average Reward (平均獎勵)', fontsize=10)
    ax.set_title('50-Episode Moving Average', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 性能對比柱狀圖
    ax = axes[1, 0]
    metrics = ['Mean', 'Std Dev', 'Max']
    ql_values = [
        results['qlearning']['mean'],
        results['qlearning']['std'],
        results['qlearning']['max']
    ]
    sarsa_values = [
        results['sarsa']['mean'],
        results['sarsa']['std'],
        results['sarsa']['max']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, ql_values, width, label='Q-Learning', color='red', alpha=0.7)
    ax.bar(x + width/2, sarsa_values, width, label='SARSA', color='blue', alpha=0.7)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 最後 50 回合的獎勵分布
    ax = axes[1, 1]
    ql_last_50 = np.array(ql_rewards[-50:])
    sarsa_last_50 = np.array(sarsa_rewards[-50:])
    ax.boxplot([ql_last_50, sarsa_last_50], tick_labels=['Q-Learning', 'SARSA'])
    ax.set_ylabel('Reward', fontsize=10)
    ax.set_title('Reward Distribution (Last 50 Episodes)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('qlearning_vs_sarsa_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 圖表已保存：qlearning_vs_sarsa_comparison.png\n")


# ============================================================================
# 第六部分：主程序入口
# ============================================================================

def main():
    """主程序入口"""
    # 創建配置
    config = EnvironmentConfig(num_runs=50)
    
    # 運行實驗
    comparator = AlgorithmComparison(config)
    comparator.run_experiments()
    
    # 打印結果
    results = comparator.print_comparison_results()
    
    # 生成可視化
    plot_results(comparator)
    
    print("=" * 80)
    print("✨ 研究完成！")
    print("=" * 80)
    print("\n📊 輸出文件：")
    print("  • qlearning_vs_sarsa_comparison.png - 演算法對比圖表")
    print("\n📝 詳細信息請參閱項目文檔")


if __name__ == "__main__":
    main()
