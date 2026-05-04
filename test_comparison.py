#!/usr/bin/env python
"""快速測試版本 - 只運行 3 次實驗以驗證圖表生成"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 設置隨機種子
np.random.seed(42)

class EnvironmentConfig:
    def __init__(self, num_runs=3):
        self.env_name = "CliffWalking-v1"
        self.num_states = 48
        self.num_actions = 4
        self.start_state = 36
        self.goal_state = 47
        self.cliff_states = set(range(37, 47))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.episodes = 500
        self.num_runs = num_runs

class QLearningAgent:
    def __init__(self, config):
        self.config = config
        self.Q = defaultdict(lambda: np.zeros(config.num_actions))
        self.rewards_history = []
    
    def select_action(self, state, training=True):
        if training and np.random.random() < self.config.epsilon:
            return np.random.randint(0, self.config.num_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, terminated):
        alpha = self.config.learning_rate
        gamma = self.config.discount_factor
        
        max_next_q = 0.0 if terminated else np.max(self.Q[next_state])
        td_error = reward + gamma * max_next_q - self.Q[state][action]
        self.Q[state][action] += alpha * td_error
    
    def train(self, env):
        self.rewards_history = []
        for episode in range(self.config.episodes):
            state, _ = env.reset()
            total_reward = 0
            
            while True:
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                self.update(state, action, reward, next_state, terminated or truncated)
                state = next_state
                if terminated or truncated:
                    break
            
            self.rewards_history.append(total_reward)

class SARSAAgent:
    def __init__(self, config):
        self.config = config
        self.Q = defaultdict(lambda: np.zeros(config.num_actions))
        self.rewards_history = []
    
    def select_action(self, state, training=True):
        if training and np.random.random() < self.config.epsilon:
            return np.random.randint(0, self.config.num_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action, terminated):
        alpha = self.config.learning_rate
        gamma = self.config.discount_factor
        
        next_q = 0.0 if terminated else self.Q[next_state][next_action]
        td_error = reward + gamma * next_q - self.Q[state][action]
        self.Q[state][action] += alpha * td_error
    
    def train(self, env):
        self.rewards_history = []
        for episode in range(self.config.episodes):
            state, _ = env.reset()
            action = self.select_action(state, training=True)
            total_reward = 0
            
            while True:
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                next_action = self.select_action(next_state, training=True)
                self.update(state, action, reward, next_state, next_action, terminated or truncated)
                state = next_state
                action = next_action
                if terminated or truncated:
                    break
            
            self.rewards_history.append(total_reward)

def main():
    config = EnvironmentConfig(num_runs=3)
    env = gym.make("CliffWalking-v1")
    
    print("🤖 運行快速測試（3 次實驗）...\n")
    
    ql_all = []
    sarsa_all = []
    
    for run in range(config.num_runs):
        print(f"📊 第 {run + 1}/3 次實驗")
        
        ql = QLearningAgent(config)
        ql.train(env)
        ql_all.append(ql.rewards_history)
        print(f"  Q-learning 完成：最後 50 回合平均 = {np.mean(ql.rewards_history[-50:]):.2f}")
        
        sarsa = SARSAAgent(config)
        sarsa.train(env)
        sarsa_all.append(sarsa.rewards_history)
        print(f"  SARSA 完成：最後 50 回合平均 = {np.mean(sarsa.rewards_history[-50:]):.2f}\n")
    
    ql_avg = np.mean(ql_all, axis=0)
    sarsa_avg = np.mean(sarsa_all, axis=0)
    
    # 繪製圖表
    print("\n📊 正在生成圖表...\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q-learning vs SARSA 演算法對比分析', fontsize=16, fontweight='bold')
    
    # 1. 累積獎勵曲線
    ax = axes[0, 0]
    episodes = range(1, len(ql_avg) + 1)
    ax.plot(episodes, ql_avg, label='Q-learning', color='red', linewidth=2, alpha=0.7)
    ax.plot(episodes, sarsa_avg, label='SARSA', color='blue', linewidth=2, alpha=0.7)
    ax.set_xlabel('Episodes (回合)', fontsize=10)
    ax.set_ylabel('Average Reward (平均獎勵)', fontsize=10)
    ax.set_title('學習曲線對比', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 移動平均（50 回合）
    ax = axes[0, 1]
    window = 50
    ql_ma = np.convolve(ql_avg, np.ones(window)/window, mode='valid')
    sarsa_ma = np.convolve(sarsa_avg, np.ones(window)/window, mode='valid')
    episodes_ma = range(window, len(ql_avg) + 1)
    ax.plot(episodes_ma, ql_ma, label='Q-learning', color='red', linewidth=2)
    ax.plot(episodes_ma, sarsa_ma, label='SARSA', color='blue', linewidth=2)
    ax.set_xlabel('Episodes (回合)', fontsize=10)
    ax.set_ylabel('Average Reward (平均獎勵)', fontsize=10)
    ax.set_title(f'{window} 回合移動平均', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 性能對比
    ax = axes[1, 0]
    ql_last_50 = ql_avg[-50:]
    sarsa_last_50 = sarsa_avg[-50:]
    
    metrics = ['平均', '最大', '最小']
    ql_values = [np.mean(ql_last_50), np.max(ql_last_50), np.min(ql_last_50)]
    sarsa_values = [np.mean(sarsa_last_50), np.max(sarsa_last_50), np.min(sarsa_last_50)]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, ql_values, width, label='Q-learning', color='red', alpha=0.7)
    ax.bar(x + width/2, sarsa_values, width, label='SARSA', color='blue', alpha=0.7)
    ax.set_ylabel('獎勵', fontsize=10)
    ax.set_title('最後 50 回合的性能對比', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 箱線圖
    ax = axes[1, 1]
    ax.boxplot([ql_last_50, sarsa_last_50], labels=['Q-learning', 'SARSA'])
    ax.set_ylabel('獎勵', fontsize=10)
    ax.set_title('獎勵分布對比', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('qlearning_vs_sarsa_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 圖表已保存：qlearning_vs_sarsa_comparison.png\n")
    
    # 打印統計結果
    print("\n" + "=" * 70)
    print("結果對比")
    print("=" * 70)
    print(f"\nQ-learning (最後 50 回合):")
    print(f"  平均獎勵: {np.mean(ql_last_50):.2f}")
    print(f"  標準差: {np.std(ql_last_50):.2f}")
    print(f"  最大: {np.max(ql_last_50):.2f}")
    print(f"  最小: {np.min(ql_last_50):.2f}")
    
    print(f"\nSARSA (最後 50 回合):")
    print(f"  平均獎勵: {np.mean(sarsa_last_50):.2f}")
    print(f"  標準差: {np.std(sarsa_last_50):.2f}")
    print(f"  最大: {np.max(sarsa_last_50):.2f}")
    print(f"  最小: {np.min(sarsa_last_50):.2f}")
    
    print("\n✨ 測試完成！")

if __name__ == "__main__":
    main()
