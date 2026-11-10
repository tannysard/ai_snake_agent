# Autonomous Snake AI with Deep Q-Learning (DQN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-Environment-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

An autonomous Reinforcement Learning agent that learns to play the classic Snake game completely from scratch using **Deep Q-Networks (DQN)**, built with PyTorch and Pygame. 

The agent starts with zero heuristic rules or pathfinding algorithms (such as A* or Dijkstra). Through trial, error, and scalar reward feedback, it develops an emergent strategy for food acquisition, self-preservation, and obstacle avoidance—peaking at a high score of **90**.

---

## Architecture & Reinforcement Learning Design
┌────────────────────────────────────────────────────────┐
│                       Environment                      │
│               (Custom Pygame Engine)                   │
└─────────────▲────────────────────────────┬─────────────┘
│                            │
Action   │                            │ State (11-dim)
[S, R, L]  │                            │ Reward (+10, -10, 0)
│                            │ Done (Bool)
┌─────────────┴────────────────────────────▼─────────────┐
│                          Agent                         │
│        Linear_QNet (PyTorch) + Experience Replay       │
└────────────────────────────────────────────────────────┘
### 1. Relative Action Space
Instead of fixed directional inputs (`UP`, `DOWN`, `LEFT`, `RIGHT`) which can trigger illegal 180° reversals, actions are evaluated **relative to the snake's current heading vector**:
* `[1, 0, 0]`: Continue Straight
* `[0, 1, 0]`: Turn Right (Clockwise 90°)
* `[0, 0, 1]`: Turn Left (Counter-Clockwise 90°)

### 2. 11-Dimensional State Vector
Rather than feeding raw pixels into high-overhead Convolutional Neural Networks (CNNs), the agent operates on a compact 11-element boolean feature vector:
* **Danger Lookahead (3):** `[danger_straight, danger_right, danger_left]` (checks 1 block ahead for wall boundaries and body collisions).
* **Current Heading (4):** `[moving_left, moving_right, moving_up, moving_down]` (one-hot encoded).
* **Relative Food Location (4):** `[food_left, food_right, food_up, food_down]` (boolean indicators relative to snake head coordinates).

### 3. Bellman Optimality Update
The agent approximates the optimal action-value function $Q^*(s, a)$ via a feedforward neural network (`Linear_QNet`). Network weights are updated using the Bellman equation:

$$Q_{\text{new}}(s, a) = R + \gamma \max_{a'} Q(s', a')$$

*(Where $Q_{\text{new}}(s, a) = R$ if the transition terminates in a terminal state).*

* **Discount Factor ($\gamma$):** `0.9` (balances immediate food acquisition with extended survival).
* **Loss Function:** Mean Squared Error ($\text{MSE}$) between predicted $Q(s, a)$ and target $Q_{\text{new}}(s, a)$.
* **Optimizer:** Adam with learning rate $\alpha = 0.001$.

### 4. Experience Replay & Exploration Strategy
* **Experience Replay Buffer:** Past transitions $(s, a, r, s', \text{done})$ are accumulated in a circular deque buffer (capacity: 100,000). At the end of each game episode, a random batch of 1,000 transitions is sampled to break temporal correlations between successive frames.
* **$\epsilon$-Greedy Policy:** Balances exploration and exploitation via dynamic epsilon decay:
  $$\epsilon = \max(0, 80 - \text{episode})$$
  Ensures the agent shifts from random trial-and-error to deterministic policy exploitation as learning stabilizes.

---

## Project Structure

```text
├── agent.py         # RL Agent logic, state parsing, memory buffer, and main training loop
├── game.py          # Custom Pygame environment (step function, relative turns, starvation checks)
├── model.py         # PyTorch Linear_QNet architecture and QTrainer optimization pipeline
├── helper.py        # Real-time Matplotlib performance plotting utility
├── model/
│   └── model.pth    # Saved PyTorch checkpoint weights for the peak-performing run
├── requirements.txt # Project dependencies
└── README.md
Installation & SetupClone the repository:Bashgit clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>
Create and activate a virtual environment:Bashpython -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
Install dependencies:Bashpip install -r requirements.txt
Running the Training PipelineTo launch the environment and start training the agent with real-time score plotting:Bashpython agent.py
Training Dynamics Across Episodes:Episodes 1–20: High exploration ($\epsilon \approx 80$). Random navigation, frequent wall collisions, and low survival rates.Episodes 20–60: The agent learns basic survival mechanics, avoiding immediate 1-step collisions and tracking food coordinates.Episodes 80+: Exploration turns off ($\epsilon \le 0$). The model exploits learned Q-values, executing targeted food runs and reaching sustained scores between 30–70+, with recorded peaks reaching 90.Key Learnings & Future RoadmapDead-End Trapping: Because the 11-dimensional state space only inspects danger 1 step ahead, the agent is susceptible to trapping itself in self-enclosed loops at high body lengths ($>60$).Double DQN (DDQN): Decoupling action selection from action evaluation using a target network to mitigate overestimation bias.Multi-Step Lookahead: Expanding the state space to 2-step danger inspection or integrating spatial flood-fill features to detect dead ends before entering them.LicenseDistributed under the MIT License. See LICENSE for more information.
<FollowUp label="Want a test.py script to watch your saved model.pth play without retraining?" qu
