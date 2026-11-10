# Autonomous Snake Agent with Deep Q-Learning

A reinforcement-learning agent that learns to play the classic Snake game using a Deep Q-Network (DQN), PyTorch, and Pygame.

The agent starts without hand-written pathfinding rules such as A* or Dijkstra. It learns from game transitions and scalar rewards, gradually improving its ability to collect food, avoid collisions, and survive for longer episodes. The documented training run reached a peak score of **90**.

## Highlights

- Custom Snake environment built with Pygame
- Relative action space that prevents illegal 180-degree turns
- Compact 11-feature state representation instead of raw-pixel input
- Feed-forward PyTorch Q-network
- Experience replay for less-correlated training samples
- Epsilon-greedy exploration that decays during training
- Live score and mean-score plotting with Matplotlib

## How the Agent Works

### Environment and actions

At each step, the agent chooses one action relative to the snake's current heading:

| Action | Meaning |
| --- | --- |
| `straight` | Continue in the current direction |
| `right` | Turn 90 degrees clockwise |
| `left` | Turn 90 degrees counter-clockwise |

Using relative actions avoids presenting the network with moves that would immediately reverse the snake into itself.

### State representation

The agent receives an 11-element feature vector:

1. **Danger ahead (3):** collision risk straight, to the right, and to the left
2. **Current direction (4):** one-hot flags for left, right, up, and down
3. **Food position (4):** whether the food is left, right, above, or below the head

The danger features check the next grid cell for wall or body collisions. This compact representation is inexpensive to train, but it does not provide complete information about distant dead ends or the full board layout.

### Reward design

The environment provides a positive reward for eating food, a negative reward for dying, and a neutral reward for an ordinary step. The values used by the documented implementation are:

- **Food:** `+10`
- **Death:** `-10`
- **Ordinary move:** `0`

### DQN update

The Q-network estimates the value of each action. Training uses the Bellman target:

\[
Q_{target}(s,a) = r + \gamma \max_{a'} Q(s', a')
\]

For terminal transitions, the future-value term is omitted. The documented training configuration uses:

| Setting | Value |
| --- | --- |
| Discount factor (`gamma`) | `0.9` |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Loss | Mean squared error |
| Replay capacity | `100,000` transitions |
| End-of-game batch size | `1,000` transitions |
| Exploration schedule | `max(0, 80 - episode)` |

## Architecture

```text
┌──────────────────────┐
│   Pygame environment │
└──────────┬───────────┘
           │ state, reward, done
           ▼
┌──────────────────────┐
│       DQN agent      │
│  Linear_QNet + Adam  │
└──────────┬───────────┘
           │ relative action
           └──────────────► environment
```

The implementation is described as four main responsibilities:

- `game.py`: Snake environment, movement, collisions, food, and scoring
- `agent.py`: state extraction, action selection, replay memory, and training loop
- `model.py`: `Linear_QNet` and the Q-learning trainer
- `helper.py`: live training metrics and plotting

## Project Structure

The intended project layout is:

```text
.
├── agent.py
├── game.py
├── model.py
├── helper.py
├── requirements.txt
├── model/
│   └── model.pth
└── README.md
```

`model/model.pth` is a saved checkpoint when a trained model has been produced. Checkpoints and generated plots may not be included in every checkout.

## Installation

### Requirements

- Python 3.8 or newer
- A working Pygame display for visual training
- pip

### Setup

```bash
git clone https://github.com/tannysard/ai_snake_agent.git
cd ai_snake_agent

python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Run Training

Start the training loop with:

```bash
python agent.py
```

The environment opens a Pygame window while the agent trains. The helper module displays the current score and running mean score. Training may take many episodes before the policy becomes stable.

## Training Behavior

Typical learning stages are:

- **Early episodes:** exploration dominates, so collisions and low scores are expected.
- **Middle episodes:** the agent begins avoiding immediate collisions and following the food direction.
- **Later episodes:** exploration decreases and the learned Q-values drive more consistent food-seeking behavior.

Results depend on the random seed, runtime, hardware, and exact implementation. The peak score of 90 is a reference from the documented run, not a guaranteed result.

## Limitations and Future Work

Because the state only checks immediate danger and relative food direction, the agent can still enter long-term traps that are not visible in the current feature vector. Potential improvements include:

- Double DQN with a separate target network
- Multi-step returns or prioritized experience replay
- Two-step or multi-step danger features
- Flood-fill or reachable-area features for dead-end detection
- A separate evaluation mode for replaying a saved checkpoint
- Configurable seeds and command-line training parameters

## License

No `LICENSE` file is currently included in the repository. Add a license file before distributing the project under a specific license.

## Acknowledgements

This project is intended as a compact example of applying Deep Q-Learning to a discrete game environment with a small, hand-designed state space.
