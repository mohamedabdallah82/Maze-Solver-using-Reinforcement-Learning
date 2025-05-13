# Maze Reinforcement Learning Project

This project implements a Q-Learning algorithm to train an agent to navigate through a maze. The agent learns to find the optimal path from a starting position to a goal position while avoiding walls.

## Features

- Interactive GUI for training configuration and visualization
- Real-time training progress monitoring
- Q-Learning implementation with customizable parameters
- Visualization of Q-table and learning curves
- Save and load trained Q-tables
- Detailed training logs

## Project Structure

```
maze_solver/
├── assets/
│   ├── jerry.png
│   ├── cheese.png
│   └── tom.png
├── output/
│   ├── models/
│   ├── train_info/
│   └── visualizations/
├── src/
│   ├── agents/
│   │   └── agent.py
│   ├── environment/
│   │   └── Environment.py
│   ├── training/
│   │   └── train.py
│   └── utils/
│       └── visualize.py
├── main.py
```

## Requirements

- Python 3.8 or higher
- Required packages (install using `pip install -r requirements.txt`):
  - numpy>=1.24.0
  - gymnasium>=0.29.1
  - matplotlib>=3.7.0
  - pygame>=2.5.0
  - gym>=0.26.0

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Algo2
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the program:
```bash
python main.py
```

2. Configure training parameters in the GUI:
   - Grid Size: Size of the maze (e.g., 6x6)
   - Number of Walls: Number of obstacles in the maze
   - Cell Size: Size of each cell in pixels
   - Episodes: Number of training episodes
   - Max Steps: Maximum steps per episode
   - Learning Rate: Q-learning learning rate
   - Discount Factor: Future reward discount factor
   - Exploration Rate: Initial exploration rate
   - Exploration Decay: Rate at which exploration decreases

3. Click "Start Training" to begin the training process

4. Monitor training progress in real-time:
   - Progress bar shows completion percentage
   - Text area displays detailed episode information
   - Success rate and rewards are tracked

5. Use visualization buttons to:
   - View the Q-table visualization
   - Plot learning curves

## Output Files

- Training logs are saved in `output/train_info/navigation.txt`
- Trained Q-tables are saved in `output/models/q_table.npy`
- Visualizations are saved in `output/visualizations/`

## Contributing

 - [Mohamed Abdallah](https://github.com/mohamedabdallah82)
 - [Menna Essam](https://github.com/mennaessam187)
 - [Abdulrahman Dahshan](https://github.com/Abdulrahman-Dahshan)
 - [Ahmed Mohamed](https://github.com/0xAMF)
 - [Mahmoud Aboelsalhen](https://github.com/sal7en)
