# 📊 Cliff Walking Environment Visualization Guide

## Overview
This document describes all the visualization outputs generated for the Q-Learning vs SARSA comparison study on the Cliff Walking environment.

---

## 🎯 Environment Layout

```
Col:  0    1    2    3    4    5    6    7    8    9   10   11
    +----+----+----+----+----+----+----+----+----+----+----+----+
  0 |    |    |    |    |    |    |    |    |    |    |    |    |
    +----+----+----+----+----+----+----+----+----+----+----+----+
  1 |    |    |    |    |    |    |    |    |    |    |    |    |
    +----+----+----+----+----+----+----+----+----+----+----+----+
  2 |    |    |    |    |    |    |    |    |    |    |    |    |
    +----+----+----+----+----+----+----+----+----+----+----+----+
  3 | S  | C  | C  | C  | C  | C  | C  | C  | C  | C  | G  |    |
    +----+----+----+----+----+----+----+----+----+----+----+----+

Legend:
  S = Start Position (3,0)
  G = Goal Position (3,11)
  C = Cliff Positions (3,1-10) - Penalty: -100
  • = Regular Cell - Step Reward: -1
```

### Rewards
- **Regular Step**: -1
- **Cliff Fall**: -100 (+ episode terminates)
- **Reaching Goal**: 0 (+ episode terminates)

---

## 📈 Generated PNG Files

### 1. **qlearning_vs_sarsa_comparison.png** (351 KB)
Algorithm comparison with 4 subplots:

#### Subplot 1: Learning Curves Comparison
- **X-axis**: Episodes (0-500)
- **Y-axis**: Average Reward (averaged over 50 runs)
- **Red Line**: Q-Learning trajectory
- **Blue Line**: SARSA trajectory
- **Key Insight**: Q-Learning reaches lower rewards due to cliff penalties

#### Subplot 2: 50-Episode Moving Average
- **X-axis**: Episodes (50-500)
- **Y-axis**: Average Reward (smoothed)
- **Window Size**: 50 episodes
- **Purpose**: Shows overall trend without noise
- **Key Insight**: SARSA converges to more stable trajectory

#### Subplot 3: Performance Metrics Comparison
- **Metrics**: Mean, Std Dev, Max
- **Red Bars**: Q-Learning
- **Blue Bars**: SARSA
- **Last 50 Episodes**: Data from final 50 episodes
- **Key Insight**: 
  - SARSA has higher mean reward (-23.56 vs -49.17)
  - SARSA has lower variance (2.86 vs 9.04)

#### Subplot 4: Reward Distribution (Boxplot)
- **Q-Learning Box**: Shows wider spread, extends lower
- **SARSA Box**: Shows tighter distribution, higher values
- **Outliers**: Individual episode rewards in last 50
- **Key Insight**: SARSA more consistent, Q-Learning more variable

---

### 2. **cliff_walking_policies.png** (425 KB)
Complete 6-subplot policy visualization:

#### Top Row: Q-Learning
1. **Environment Layout**
   - Shows grid structure
   - Marks start (green), goal (blue), cliff (red)

2. **Learned Policy**
   - Arrows (↑↓←→) show best action at each state
   - Q-Learning learns to go straight right along cliff edge
   - Risk-taking strategy: finds optimal but dangerous path

3. **Value Function Heatmap**
   - Colors represent state values
   - Green = high value states
   - Red = low value states
   - Q-Learning assigns high values near optimal path

#### Bottom Row: SARSA
1. **Environment Layout** (same as Q-Learning)

2. **Learned Policy**
   - Shows more conservative navigation pattern
   - Takes upper path around cliff
   - Safer but longer route (15-17 steps vs 13 steps)

3. **Value Function Heatmap**
   - Values reflect conservative strategy
   - Penalizes cliff-adjacent states heavily
   - Prefers upper safe route

---

### 3. **cliff_walking_policy_comparison.png** (104 KB)
Side-by-side policy arrow comparison:

**Left Panel**: Q-Learning Policy
```
    Cliff edge strategy (along bottom row)
    Arrows show: → → → → → → → → →
    Minimizes steps: 13
    Maximum risk: High
```

**Right Panel**: SARSA Policy
```
    Safe upper route strategy
    Arrows show: → → ↑ → → → ↓ → →
    More steps: 15-17
    Risk level: Low
```

---

### 4. **cliff_walking_value_comparison.png** (192 KB)
Value function heatmaps side-by-side:

**Left Heatmap**: Q-Learning Values
- High values along cliff edge (optimal path known)
- Sharp gradient near cliff
- Values range: -100 to ~-13
- Reflects aggressive optimization

**Right Heatmap**: SARSA Values
- Lower values throughout
- Safer regions have higher values
- Values range: -100 to ~-15
- Reflects conservative learning

---

## 🔑 Key Findings

### Algorithm Differences Visualized:

| Aspect | Q-Learning | SARSA |
|--------|-----------|-------|
| **Path Type** | Optimal but risky | Safe but suboptimal |
| **Step Count** | ~13 steps | ~15-17 steps |
| **Value Pattern** | Aggressive gradient | Conservative gradient |
| **Cliff Approach** | Right along edge | Upper safe route |
| **Stability** | High variance | Low variance |

### Why Different Policies?

**Q-Learning (Off-policy)**:
- Learns: "What's the best action?" (optimal policy)
- Updates using: max Q(S',a)
- Result: Discovers 13-step path along cliff
- Risk: Frequently falls off cliff during training (-100 penalty)

**SARSA (On-policy)**:
- Learns: "What should I do?" (current policy)
- Updates using: Q(S',A') of actual next action
- Result: Discovers 15-17 step safe path
- Benefit: Avoids cliff during training (lower penalties)

---

## 📊 Performance Metrics

**Final 50 Episodes Average:**

| Metric | Q-Learning | SARSA | Winner |
|--------|-----------|-------|--------|
| Mean Reward | -49.17 | -23.56 | SARSA ✓ |
| Std Dev | 9.04 | 2.86 | SARSA ✓ |
| Max Reward | -29.02 | -18.44 | SARSA ✓ |
| Min Reward | -67.68 | -31.32 | SARSA ✓ |

**Convergence:**
- Both converge after ~100 episodes
- Q-Learning converges ~20% faster
- SARSA achieves better final performance

---

## 🎓 Educational Value

These visualizations demonstrate:

1. **Off-policy vs On-policy Learning**
   - Policy comparison shows exploration impact
   - Value functions reveal different optimization targets

2. **Risk-Return Tradeoff**
   - Q-Learning: Optimal but risky
   - SARSA: Suboptimal but safer

3. **Training Stability**
   - Learning curves show convergence behavior
   - Reward distribution shows consistency

4. **Value Function Learning**
   - Heatmaps show spatial value patterns
   - Arrow policies show learned action preferences

---

## 💡 Practical Applications

### Use Q-Learning When:
- Offline learning is possible
- Simulation environment (safe failures)
- Optimality is paramount
- Risk tolerance is high

### Use SARSA When:
- Online learning is required
- Real-world deployment (safety critical)
- Stability is paramount
- Risk tolerance is low

---

## 🔧 Technical Details

**Environment**: Gymnasium CliffWalking-v1
- **Grid Size**: 4×12 (48 states)
- **Action Space**: 4 discrete actions (UP, RIGHT, DOWN, LEFT)
- **Episode**: Ends when reaching goal or falling off cliff

**Training Parameters**:
- **Episodes per run**: 500
- **Number of runs**: 50
- **Learning rate (α)**: 0.1
- **Discount factor (γ)**: 0.9
- **Exploration rate (ε)**: 0.1

**Visualization Libraries**:
- Matplotlib for plotting
- NumPy for numerical computation
- Custom visualization classes

---

## 📝 File Locations

All PNG files are located in: `/Users/brainshi/Desktop/強化學習/H2/`

```
📁 H2/
├── qlearning_vs_sarsa_comparison.png       (351 KB) ← Algorithm metrics
├── cliff_walking_policies.png              (425 KB) ← 6-subplot comparison
├── cliff_walking_policy_comparison.png     (104 KB) ← Policy arrows
└── cliff_walking_value_comparison.png      (192 KB) ← Value heatmaps
```

---

Generated: 2026-05-04
Script: `visualize_cliff_environment.py` & `comparison_study.py`
