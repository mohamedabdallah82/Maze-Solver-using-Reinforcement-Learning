import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curve(rewards_history, steps_history, success_rate):
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_path = os.path.join(project_root, "output", "visualizations", "learning_curves.png")

    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot rewards
    ax1.plot(rewards_history, label='Episode Reward')
    ax1.set_title('Rewards per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot steps
    ax2.plot(steps_history, label='Steps per Episode', color='orange')
    ax2.set_title('Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Number of Steps')
    ax2.grid(True)
    
    # Plot success rate
    ax3.plot(success_rate, label='Success Rate', color='green')
    ax3.set_title('Success Rate over Time')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate')
    ax3.grid(True)
    
    # Add moving averages
    window_size = 10
    if len(rewards_history) >= window_size:
        rewards_ma = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
        steps_ma = np.convolve(steps_history, np.ones(window_size)/window_size, mode='valid')
        
        ax1.plot(range(window_size-1, len(rewards_history)), rewards_ma, 
                label=f'{window_size}-Episode Moving Average', color='red')
        ax2.plot(range(window_size-1, len(steps_history)), steps_ma, 
                label=f'{window_size}-Episode Moving Average', color='red')
    
    # Add legends
    ax1.legend()
    ax2.legend()
    ax3.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_q_table(q_table, grid_size):
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_path = os.path.join(project_root, "output", "visualizations", "q_table_visualization.png")
    
    # Create a figure with 4 subplots (one for each action)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    action_names = ['Up', 'Down', 'Left', 'Right']
    
    for action in range(4):
        # Create a grid to store Q-values
        q_grid = np.zeros(grid_size)
        
        # Fill in Q-values for each state
        for state in q_table:
            if state in q_table:
                q_grid[state] = q_table[state][action]
        
        # Plot heatmap
        im = axes[action].imshow(q_grid, cmap='viridis')
        axes[action].set_title(f'Q-values for {action_names[action]}')
        plt.colorbar(im, ax=axes[action])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 