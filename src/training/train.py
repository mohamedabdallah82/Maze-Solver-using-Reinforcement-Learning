import numpy as np
from src.environment.Environment import myMazeEnv
from src.agents.agent import QLearningAgent
import time
from datetime import timedelta
import os

def train(episodes=1000, grid_size=(6,6), number_of_walls=10, max_steps_per_episode=100, load_previous=False, callback=None):
    # Create environment
    env = myMazeEnv(render_mode="human", grid_size=grid_size, number_of_walls=number_of_walls)
    
    # Create agent
    state_size = env.observation_space.shape[0]  # (x, y) position
    action_size = env.action_space.n  # up, down, left, right
    agent = QLearningAgent(state_size, action_size)
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load previous Q-table if requested and exists
    q_table_path = os.path.join(project_root, "output", "models", "q_table.npy")
    if load_previous and os.path.exists(q_table_path):
        print("Loading previous Q-table...")
        agent.load_q_table(q_table_path)
        print("Previous Q-table loaded successfully!")
    
    # Training metrics
    rewards_history = []
    steps_history = []
    success_rate = []
    successful_episodes = 0
    
    # Timing metrics
    start_time = time.time()
    action_names = ['Up', 'Down', 'Left', 'Right']
    
    # Create navigation log file
    log_path = os.path.join(project_root, "output", "train_info", "navigation.txt")
    prog_file = open(log_path, 'w')
    
    for episode in range(episodes):
        episode_start_time = time.time()
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        # Print and save episode start
        episode_info = f"\nEpisode {episode + 1}/{episodes} started\nStarting position: {state}"
        # print(episode_info)
        prog_file.write(episode_info + "\n")
        
        while not done and steps < max_steps_per_episode:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # Print and save step information
            step_info = f"Step {steps + 1}:\nAction: {action_names[action]}\nCurrent position: {state}\nNext position: {next_state}\nReward: {reward:.2f}"
            # print(step_info)
            prog_file.write(step_info + "\n")
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if reward == 1.0:  # Reached goal
                successful_episodes += 1
                goal_msg = "  Goal reached!"
                # print(goal_msg)
                prog_file.write(goal_msg + "\n")
            elif steps >= max_steps_per_episode:
                max_steps_msg = "  Max steps reached!"
                # print(max_steps_msg)
                prog_file.write(max_steps_msg + "\n")
        
        # Calculate episode time
        episode_time = time.time() - episode_start_time
        
        # Record metrics
        rewards_history.append(total_reward)
        steps_history.append(steps)
        success_rate.append(successful_episodes / (episode + 1))
        
        # Calculate total elapsed time
        total_elapsed_time = time.time() - start_time
        elapsed_time_str = str(timedelta(seconds=int(total_elapsed_time)))
        
        # Print and save episode summary
        summary = f"\nEpisode {episode + 1} Summary:\nSuccess Rate: {success_rate[-1]:.2%}\nTotal Reward: {total_reward:.2f}\nSteps taken: {steps}\nEpisode time: {episode_time:.2f} seconds\nTotal time elapsed: {elapsed_time_str}\nExploration Rate: {agent.exploration_rate:.2f}\n" + "-" * 50
        # print(summary)
        prog_file.write(summary + "\n")
        
        # Call the callback function if provided
        if callback is not None:
            if not callback(episode, total_reward, steps, success_rate[-1], episode_time):
                break
    
    # Print and save final training summary
    total_time = time.time() - start_time
    final_summary = f"\nTraining Complete!\nTotal training time: {str(timedelta(seconds=int(total_time)))}\nFinal success rate: {success_rate[-1]:.2%}\nAverage steps per episode: {np.mean(steps_history):.2f}\nAverage reward per episode: {np.mean(rewards_history):.2f}"
    # print(final_summary)
    prog_file.write(final_summary + "\n")
    
    # Close the log file
    prog_file.close()
    
    # Save the trained Q-table
    agent.save_q_table(q_table_path)
    
    env.close()
    return agent