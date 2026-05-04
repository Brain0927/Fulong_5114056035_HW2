#!/usr/bin/env python
"""
Cliff Walking Environment Visualization
========================================
Visualizes the Cliff Walking environment grid and learned policies
from Q-Learning and SARSA agents.

Shows:
1. Environment layout with start, goal, and cliff
2. Q-Learning policy visualization
3. SARSA policy visualization
4. Value function heatmaps
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from typing import Tuple, Dict


class CliffEnvironmentVisualizer:
    """Visualize Cliff Walking environment and policies."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.grid_height = 4
        self.grid_width = 12
        self.start_pos = (3, 0)
        self.goal_pos = (3, 11)
        self.cliff_positions = [(3, i) for i in range(1, 11)]
        
        # Action names for visualization
        self.action_names = ['↑', '→', '↓', '←']  # UP, RIGHT, DOWN, LEFT
        
    def draw_environment(self, ax, title: str = "Cliff Walking Environment"):
        """Draw the basic environment layout."""
        ax.set_xlim(-0.5, self.grid_width - 0.5)
        ax.set_ylim(-0.5, self.grid_height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Draw grid
        for i in range(self.grid_height + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
        for j in range(self.grid_width + 1):
            ax.axvline(j - 0.5, color='black', linewidth=0.5)
        
        # Draw cliff
        for row, col in self.cliff_positions:
            rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1, 
                                    linewidth=1, edgecolor='black', 
                                    facecolor='red', alpha=0.4)
            ax.add_patch(rect)
            ax.text(col, row, 'C', ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        
        # Draw start
        circle_start = patches.Circle(self.start_pos[::-1], 0.3, 
                                     color='green', alpha=0.8)
        ax.add_patch(circle_start)
        ax.text(self.start_pos[1] - 0.4, self.start_pos[0] + 0.35, 'S', 
               fontsize=9, fontweight='bold', color='white')
        
        # Draw goal
        circle_goal = patches.Circle(self.goal_pos[::-1], 0.3, 
                                    color='blue', alpha=0.8)
        ax.add_patch(circle_goal)
        ax.text(self.goal_pos[1], self.goal_pos[0], 'G', 
               fontsize=9, fontweight='bold', color='white')
        
        ax.set_xticks(range(self.grid_width))
        ax.set_yticks(range(self.grid_height))
        ax.set_xticklabels(range(self.grid_width))
        ax.set_yticklabels(range(self.grid_height))
        ax.grid(True, alpha=0.3)
        
    def draw_policy(self, ax, Q_table: Dict, title: str = "Policy Visualization"):
        """Draw policy arrows on the environment."""
        self.draw_environment(ax, title)
        
        # Draw policy arrows
        for state in range(48):
            row = state // self.grid_width
            col = state % self.grid_width
            
            # Skip cliff and goal
            if (row, col) in self.cliff_positions or (row, col) == self.goal_pos:
                continue
            
            # Get best action
            if state in Q_table:
                best_action = np.argmax(Q_table[state])
                action_symbol = self.action_names[best_action]
                ax.text(col, row, action_symbol, ha='center', va='center',
                       fontsize=10, fontweight='bold', color='black')
            
    def draw_value_heatmap(self, ax, Q_table: Dict, title: str = "State Value Function"):
        """Draw heatmap of state values."""
        self.draw_environment(ax, title)
        
        # Create value grid
        value_grid = np.zeros((self.grid_height, self.grid_width))
        for state in range(48):
            row = state // self.grid_width
            col = state % self.grid_width
            
            if (row, col) in self.cliff_positions:
                value_grid[row, col] = -100  # Cliff value
            elif (row, col) == self.goal_pos:
                value_grid[row, col] = 0     # Goal value
            elif state in Q_table:
                value_grid[row, col] = np.max(Q_table[state])
            else:
                value_grid[row, col] = 0
        
        # Draw heatmap
        im = ax.imshow(value_grid, cmap='RdYlGn_r', alpha=0.6, aspect='auto',
                      extent=[-0.5, self.grid_width - 0.5, 
                             self.grid_height - 0.5, -0.5])
        
        # Add value labels
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if (i, j) not in self.cliff_positions and (i, j) != self.goal_pos:
                    value_text = f'{value_grid[i, j]:.1f}'
                    ax.text(j, i, value_text, ha='center', va='center',
                           fontsize=7, fontweight='bold', color='black', alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Value', fontsize=9)


def train_agent_for_visualization(agent_type: str, num_episodes: int = 500) -> Dict:
    """Train an agent and return its Q-table for visualization."""
    from collections import defaultdict
    
    env = gym.make("CliffWalking-v1")
    Q = defaultdict(lambda: np.zeros(4))
    
    alpha = 0.1      # Learning rate
    gamma = 0.9      # Discount factor
    epsilon = 0.1    # Exploration rate
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        
        if agent_type == "Q-Learning":
            # Q-Learning
            while True:
                # Epsilon-greedy
                if np.random.random() < epsilon:
                    action = np.random.randint(0, 4)
                else:
                    action = np.argmax(Q[state])
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # Q-Learning update: use max Q-value of next state
                max_next_q = np.max(Q[next_state]) if not (terminated or truncated) else 0
                td_error = reward + gamma * max_next_q - Q[state][action]
                Q[state][action] += alpha * td_error
                
                state = next_state
                if terminated or truncated:
                    break
        
        else:  # SARSA
            # SARSA
            if np.random.random() < epsilon:
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(Q[state])
            
            while True:
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                if np.random.random() < epsilon:
                    next_action = np.random.randint(0, 4)
                else:
                    next_action = np.argmax(Q[next_state])
                
                # SARSA update: use actual next action's Q-value
                next_q = Q[next_state][next_action] if not (terminated or truncated) else 0
                td_error = reward + gamma * next_q - Q[state][action]
                Q[state][action] += alpha * td_error
                
                state = next_state
                action = next_action
                if terminated or truncated:
                    break
    
    env.close()
    return dict(Q)


def main():
    """Main visualization function."""
    print("🎮 Training agents for visualization...")
    print("  Training Q-Learning agent...")
    ql_Q = train_agent_for_visualization("Q-Learning", num_episodes=500)
    print("  Training SARSA agent...")
    sarsa_Q = train_agent_for_visualization("SARSA", num_episodes=500)
    
    print("\n📊 Generating Cliff Walking environment visualization...")
    
    visualizer = CliffEnvironmentVisualizer()
    
    # Create figure with 6 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Cliff Walking Environment: Q-Learning vs SARSA Policy Visualization',
                fontsize=14, fontweight='bold')
    
    # Row 1: Q-Learning
    visualizer.draw_environment(axes[0, 0], "Q-Learning: Environment Layout")
    visualizer.draw_policy(axes[0, 1], ql_Q, "Q-Learning: Learned Policy")
    visualizer.draw_value_heatmap(axes[0, 2], ql_Q, "Q-Learning: Value Function")
    
    # Row 2: SARSA
    visualizer.draw_environment(axes[1, 0], "SARSA: Environment Layout")
    visualizer.draw_policy(axes[1, 1], sarsa_Q, "SARSA: Learned Policy")
    visualizer.draw_value_heatmap(axes[1, 2], sarsa_Q, "SARSA: Value Function")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.8, label='Start (S)'),
        Patch(facecolor='blue', alpha=0.8, label='Goal (G)'),
        Patch(facecolor='red', alpha=0.4, label='Cliff (C)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('cliff_walking_policies.png', dpi=300, bbox_inches='tight')
    print("✅ Policy visualization saved: cliff_walking_policies.png\n")
    
    # Create a simplified 2-subplot comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Cliff Walking: Policy Comparison',
                 fontsize=14, fontweight='bold')
    
    visualizer.draw_policy(axes2[0], ql_Q, "Q-Learning Policy")
    visualizer.draw_policy(axes2[1], sarsa_Q, "SARSA Policy")
    
    plt.tight_layout()
    plt.savefig('cliff_walking_policy_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Policy comparison saved: cliff_walking_policy_comparison.png\n")
    
    # Create value heatmap comparison
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('Cliff Walking: Value Function Comparison',
                 fontsize=14, fontweight='bold')
    
    visualizer.draw_value_heatmap(axes3[0], ql_Q, "Q-Learning Value Function")
    visualizer.draw_value_heatmap(axes3[1], sarsa_Q, "SARSA Value Function")
    
    plt.tight_layout()
    plt.savefig('cliff_walking_value_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Value function comparison saved: cliff_walking_value_comparison.png\n")
    
    print("=" * 70)
    print("ALL CLIFF WALKING VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. cliff_walking_policies.png - Complete 6-subplot comparison")
    print("  2. cliff_walking_policy_comparison.png - Side-by-side policies")
    print("  3. cliff_walking_value_comparison.png - Value function heatmaps")


if __name__ == "__main__":
    main()
