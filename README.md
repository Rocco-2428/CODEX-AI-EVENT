# 🤖 CODEX AI EVENT — Bangalore Wumpus World

A competitive AI programming event submission featuring two core components:
a **Bangalore-themed Wumpus World** with A\* pathfinding and a **NanoGrad** minimal autograd engine built from scratch.

---

## 📁 Project Structure

```
CODEX-AI-EVENT/
├── wumpus_world.py      # Bangalore Wumpus World simulation with A* pathfinding
├── nanograd.py          # Minimal scalar autograd engine + small MLP
├── validator.py         # Official test suite runner
├── test_cases.json      # Test cases for validation
├── team_config.json     # Team configuration (seed, grid layout, team ID)
└── requirements.txt     # Python dependencies
```

---

## 🗺️ Bangalore Wumpus World

A Pygame-based grid simulation where an AI agent navigates a 5×10 Bangalore-themed world filled with hazards, using **A\* pathfinding** to find the optimal route to the goal.

### World Elements

| Element | Symbol | Effect |
|---|---|---|
| 🟢 Goal | `GOAL` | Reach this to win |
| ⚫ Pit | `PIT` | Instant game over |
| 🔴 Traffic Light | `SIGNAL` | Triggers a 1.5-second delay |
| 🟫 Cow | `COW` | Collision resets agent to start; cell becomes forbidden |
| 🟡 Agent | Circle | Starts at `(0, bottom-left)` |

### Percepts

The agent perceives clues in adjacent cells:

- **`~` (Breeze)** — a pit is nearby
- **`M` (Moo)** — a cow is nearby  
- **`L` (Light)** — a traffic signal is nearby

### A\* Pathfinding

The agent uses **A\* with Manhattan distance heuristic** and weighted cell costs:

- Pits → `∞` (impassable)
- Previously collided cow cells → `∞` (forbidden after collision)
- Traffic lights → cost `20`
- Cows → cost `50` (avoidable but expensive)
- Empty cells → random weight `1–15`

On a cow collision, the cell is marked forbidden and the agent **automatically replans** from the start.

### Controls

| Key | Action |
|---|---|
| `SPACE` | Run A\* and begin auto-navigation |
| `↑ ↓ ← →` | Manual agent movement |
| `V` | Toggle A\* visualization overlay |
| `R` | Reset the world |
| `C` | Clear forbidden cells |
| `ESC` | Quit |

### A\* Visualization

When enabled, the grid overlays:
- 🔵 **Blue** — open set (cells being considered)
- ⬛ **Gray** — closed set (already explored)
- 🟡 **Yellow** — chosen path

---

## 🧠 NanoGrad — Minimal Autograd Engine

A from-scratch scalar automatic differentiation engine inspired by [micrograd](https://github.com/karpathy/micrograd), supporting forward and backward passes through a computation graph.

### Features

- `Value` class with full scalar autograd support
- Supported operations: `+`, `-`, `*`, `/`, `**`, `relu`
- Reverse-mode backpropagation via topological sort
- Small MLP (`Neuron`, `MLP` classes) built on top of `Value`

### Quick Example

```python
from nanograd import Value

x = Value(2.0)
y = Value(3.0)
z = x * y + x**2

z.backward()
print(x.grad)  # dz/dx = y + 2x = 7.0
print(y.grad)  # dz/dy = x = 2.0
```

### Built-in Tests

```bash
python nanograd.py
```

Runs two built-in tests:
1. **Basic Operations** — validates gradients for arithmetic expressions
2. **MLP Training** — trains a small network to learn `y = 2x + 3`

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

> Dependencies include `pygame` and `numpy`.

### Configure Your Team

Edit `team_config.json` before running:

```json
{
  "seed": 42,
  "team_id": "teamX",
  "grid_config": {
    "traffic_lights": 3,
    "cows": 3,
    "pits": 3
  }
}
```

- **`seed`** — controls world generation (reproducible layout)
- **`team_id`** — your team identifier for the event
- **`grid_config`** — number of each hazard type to place

---

## 🚀 Running the Project

### Wumpus World

```bash
python wumpus_world.py
```

### NanoGrad Tests

```bash
python nanograd.py
```

### Validator (Official Test Suite)

```bash
python validator.py
```

---

## 🏆 Event Context

This project was built for the **CODEX AI Event**, combining:
- Classical AI concepts (knowledge-based agents, percepts, environment modeling)
- Search algorithms (A\* with heuristics and dynamic replanning)
- Deep learning fundamentals (autograd from scratch, backpropagation)

---

## 📄 License

This project is submitted as part of a competitive AI event. All code is original unless otherwise noted.
