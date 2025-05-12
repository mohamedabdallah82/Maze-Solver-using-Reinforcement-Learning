import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
from src.training.train import train
from src.utils.visualize import visualize_q_table, plot_learning_curve
from src.utils.visualize_navigation import visualize_navigation
import time
from datetime import timedelta

class MazeTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Training Configuration")
        self.root.geometry("800x600")
        
        # Create main frames
        self.config_frame = ttk.LabelFrame(root, text="Training Configuration", padding="10")
        self.config_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        self.progress_frame = ttk.LabelFrame(root, text="Training Progress", padding="10")
        self.progress_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        
        # Configure grid weights
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)
        
        # Initialize variables
        self.grid_size = tk.StringVar(value="6")
        self.num_walls = tk.StringVar(value="10")
        self.cell_size = tk.StringVar(value="80")
        self.episodes = tk.StringVar(value="1000")
        self.max_steps = tk.StringVar(value="100")
        self.learning_rate = tk.StringVar(value="0.1")
        self.discount_factor = tk.StringVar(value="0.95")
        self.exploration_rate = tk.StringVar(value="1.0")
        self.exploration_decay = tk.StringVar(value="0.995")
        self.load_previous = tk.BooleanVar(value=False)
        
        # Training metrics
        self.rewards_history = []
        self.steps_history = []
        self.success_rate = []
        self.current_q_table = None
        self.start_time = None
        
        # Create configuration widgets
        self.create_config_widgets()
        
        # Create progress widgets
        self.create_progress_widgets()
        
        # Training thread
        self.training_thread = None
        self.is_training = False
        
    def create_config_widgets(self):
        # Grid size
        ttk.Label(self.config_frame, text="Grid Size:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(self.config_frame, textvariable=self.grid_size, width=10).grid(row=0, column=1, sticky="w", pady=2)
        
        # Number of walls
        ttk.Label(self.config_frame, text="Number of Walls:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(self.config_frame, textvariable=self.num_walls, width=10).grid(row=1, column=1, sticky="w", pady=2)
        
        # Cell size
        ttk.Label(self.config_frame, text="Cell Size:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(self.config_frame, textvariable=self.cell_size, width=10).grid(row=2, column=1, sticky="w", pady=2)
        
        # Episodes
        ttk.Label(self.config_frame, text="Number of Episodes:").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(self.config_frame, textvariable=self.episodes, width=10).grid(row=3, column=1, sticky="w", pady=2)
        
        # Max steps
        ttk.Label(self.config_frame, text="Max Steps per Episode:").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Entry(self.config_frame, textvariable=self.max_steps, width=10).grid(row=4, column=1, sticky="w", pady=2)
        
        # Learning rate
        ttk.Label(self.config_frame, text="Learning Rate:").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Entry(self.config_frame, textvariable=self.learning_rate, width=10).grid(row=5, column=1, sticky="w", pady=2)
        
        # Discount factor
        ttk.Label(self.config_frame, text="Discount Factor:").grid(row=6, column=0, sticky="w", pady=2)
        ttk.Entry(self.config_frame, textvariable=self.discount_factor, width=10).grid(row=6, column=1, sticky="w", pady=2)
        
        # Exploration rate
        ttk.Label(self.config_frame, text="Exploration Rate:").grid(row=7, column=0, sticky="w", pady=2)
        ttk.Entry(self.config_frame, textvariable=self.exploration_rate, width=10).grid(row=7, column=1, sticky="w", pady=2)
        
        # Exploration decay
        ttk.Label(self.config_frame, text="Exploration Decay:").grid(row=8, column=0, sticky="w", pady=2)
        ttk.Entry(self.config_frame, textvariable=self.exploration_decay, width=10).grid(row=8, column=1, sticky="w", pady=2)
        
        # Load previous
        ttk.Checkbutton(self.config_frame, text="Load Previous Q-table", variable=self.load_previous).grid(row=9, column=0, columnspan=2, sticky="w", pady=2)
        
        # Start/Stop button
        self.start_button = ttk.Button(self.config_frame, text="Start Training", command=self.toggle_training)
        self.start_button.grid(row=10, column=0, columnspan=2, pady=10)
        
        # Visualization buttons
        self.visualize_frame = ttk.LabelFrame(self.config_frame, text="Visualizations", padding="5")
        self.visualize_frame.grid(row=11, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Button(self.visualize_frame, text="Show Q-table", command=self.show_q_table).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(self.visualize_frame, text="Show Learning Curves", command=self.show_learning_curves).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(self.visualize_frame, text="Show Navigation Results", command=self.show_navigation_results).grid(row=1, column=0, columnspan=2, padx=5, pady=2)
        
    def create_progress_widgets(self):
        # Progress frame
        progress_container = ttk.Frame(self.progress_frame)
        progress_container.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Progress slider
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_slider = ttk.Progressbar(
            progress_container,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            length=300
        )
        self.progress_slider.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        # Progress percentage label
        self.progress_label = ttk.Label(progress_container, text="0%")
        self.progress_label.grid(row=0, column=1, sticky="w")
        
        # Progress text
        self.progress_text = scrolledtext.ScrolledText(self.progress_frame, width=50, height=30)
        self.progress_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights
        self.progress_frame.grid_columnconfigure(0, weight=1)
        self.progress_frame.grid_rowconfigure(1, weight=1)
        progress_container.grid_columnconfigure(0, weight=1)
        
    def update_progress(self, progress):
        self.progress_var.set(progress)
        self.progress_label.config(text=f"{progress:.1f}%")
        
    def log_progress(self, message):
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)
        
    def show_q_table(self):
        if self.current_q_table is not None:
            grid_size = (int(self.grid_size.get()), int(self.grid_size.get()))
            visualize_q_table(self.current_q_table, grid_size)
            self.log_progress("Q-table visualization generated and saved in 'output/visualizations/q_table_visualization.png'")
        else:
            self.log_progress("No Q-table available. Please train the agent first.")
            
    def show_learning_curves(self):
        if self.rewards_history:
            plot_learning_curve(self.rewards_history, self.steps_history, self.success_rate)
            self.log_progress("Learning curves generated and saved in 'output/visualizations/learning_curves.png'")
        else:
            self.log_progress("No training data available. Please train the agent first.")
        
    def show_navigation_results(self):
        try:
            visualize_navigation()
            self.log_progress("Navigation visualization generated and saved in 'output/visualizations/navigation_visualization.png'")
        except Exception as e:
            self.log_progress(f"Error generating navigation visualization: {str(e)}")
        
    def toggle_training(self):
        if not self.is_training:
            self.start_training()
        else:
            self.stop_training()
            
    def start_training(self):
        self.is_training = True
        self.start_button.config(text="Stop Training")
        self.progress_text.delete(1.0, tk.END)
        self.update_progress(0)
        
        # Reset training metrics
        self.rewards_history = []
        self.steps_history = []
        self.success_rate = []
        self.current_q_table = None
        self.start_time = time.time()
        
        # Get parameters
        params = {
            'episodes': int(self.episodes.get()),
            'grid_size': (int(self.grid_size.get()), int(self.grid_size.get())),
            'number_of_walls': int(self.num_walls.get()),
            'max_steps_per_episode': int(self.max_steps.get()),
            'load_previous': self.load_previous.get()
        }
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self.run_training, args=(params,))
        self.training_thread.start()
        
    def stop_training(self):
        self.is_training = False
        self.start_button.config(text="Start Training")
        
    def run_training(self, params):
        try:
            # Custom callback function to update GUI
            def update_callback(episode, reward, steps, success_rate, episode_time=None):
                if not self.is_training:
                    return False
                    
                # Update progress slider and label
                progress = (episode + 1) / params['episodes'] * 100
                self.root.after(0, lambda: self.update_progress(progress))
                
                # Update training metrics
                self.rewards_history.append(reward)
                self.steps_history.append(steps)
                self.success_rate.append(success_rate)
                
                # Calculate total elapsed time
                total_elapsed_time = time.time() - self.start_time
                elapsed_time_str = str(timedelta(seconds=int(total_elapsed_time)))
                
                # Log progress
                message = f"Episode {episode + 1}/{params['episodes']}\n"
                message += f"Reward: {reward:.2f}\n"
                message += f"Steps: {steps}\n"
                message += f"Success Rate: {success_rate:.2%}\n"
                if episode_time is not None:
                    message += f"Episode Time: {episode_time:.2f} seconds\n"
                message += f"Total Time Elapsed: {elapsed_time_str}\n"
                message += "-" * 50
                self.root.after(0, lambda: self.log_progress(message))
                
                return True
            
            # Start training
            agent = train(**params, callback=update_callback)
            self.current_q_table = agent.q_table
            
        except Exception as e:
            self.root.after(0, lambda: self.log_progress(f"Error: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.start_button.config(text="Start Training"))
            self.is_training = False

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeTrainingGUI(root)
    root.mainloop() 