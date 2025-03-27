# Battleship Reinforcement Learning Project

This project implements and compares different agents for playing the Battleship game, using reinforcement learning and other AI techniques.

## Project Structure

- `battleship_env.py`: The Battleship game environment (using OpenAI Gym interface)
- `random_agent.py`: Implementation of a random guessing agent
- `smart_agent.py`: Implementation of an agent with simple explore-exploit strategy
- `qlearning_agent.py`: Implementation of a Q-Learning reinforcement learning agent
- `model_comparison.py`: Tools to compare different agents
- `battleship_rl.ipynb`: Jupyter notebook for running experiments
- `requirements.txt`: Required Python packages

## Installation

```bash
pip install -r requirements.txt
```

## Running the Project

You can either:

1. Run the Jupyter notebook:
```bash
jupyter notebook battleship_5x5.ipynb
```

2. Run individual agent scripts:
```bash
python random_agent.py
python smart_agent.py
python qlearning_agent.py
```

3. Compare model performance:
```bash
python model_comparison.py
```

## Game Configuration

The Battleship game is configurable with the following parameters:
- `grid_size`: Size of the game grid (default: 5x5)
- `ships_config`: List of ship sizes (default: [3, 2] for 5x5 grid)

## Agents

### 1. Random Agent
- Makes completely random guesses
- Serves as a baseline for comparison

### 2. Smart Agent
- Uses a simple explore-exploit strategy
- When a hit is found, focuses on adjacent cells
- Returns to random exploration after sinking a ship

### 3. Q-Learning Agent
- Uses reinforcement learning to improve over time
- Learns optimal strategies through experience
- Features configurable hyperparameters:
  - Learning rate
  - Discount factor
  - Exploration rate and decay

## Results

Results from training and testing are saved in the `battleship_results` directory, including:
- Performance metrics (CSV files)
- Learning curves
- Comparison plots

## Comparative Analysis

The model comparison framework allows for fair evaluation of different strategies on the same test scenarios, producing:
- Box plots of shots distribution
- Bar charts of average shots and win rates
- Line plots of performance per episode

## Team Project Structure

This project is designed for a team of 5 members:
1. Game environment development
2. Random agent implementation
3. Smart agent implementation
4. Q-Learning agent implementation
5. Model comparison and analysis

Each component can be developed independently and then integrated for the final comparative analysis.