import os
import subprocess
from datetime import datetime

# List of tasks and algorithms to test
tasks = [
    "CartPole-v1",
    # "LunarLander-v2",
    "HalfCheetah-v2",
    "Ant-v2",
    "Walker2d-v2"
]
algorithms = ["ppo_clipped", "ppo_kl", "trpo", "actor_critic", "reinforce"]

# Parameters for training and testing
num_train_episodes = 1000
num_play_episodes = 50

# Define paths to the training and play scripts
train_script = "train.py"
play_script = "play.py"

# Timestamp for the start of this experiment run
experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Prepare log directory
log_dir = "experiment_logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "log.txt")

# Clear log file at the beginning of each experiment run
with open(log_file_path, "w") as log_file:
    log_file.write(f"Experiment Start Time: {experiment_time}\n")
    log_file.write("Task, Algorithm, Mean Return\n")

# Function to train a model
def train_model(task, algorithm):
    print(f"Training {algorithm} on {task}...")
    train_command = [
        "python", train_script,
        "--task", task,
        "--algo", algorithm
    ]
    subprocess.run(train_command)

# Function to test a model and get mean return
def test_model(task, algorithm):
    print(f"Testing {algorithm} on {task}...")
    play_command = [
        "python", play_script,
        "--task", task,
        "--algo", algorithm,
        "--num_episodes", str(num_play_episodes)
    ]
    result = subprocess.run(play_command, capture_output=True, text=True)
    
    # Extract the mean return from play script output
    for line in result.stdout.splitlines():
        if "Mean Return over" in line:
            mean_return = float(line.split(":")[-1].strip())
            return mean_return
    return None

# Main experiment loop
for task in tasks:
    for algorithm in algorithms:
        # Train the model
        # train_model(task, algorithm)
        
        # Test the model and get mean return
        mean_return = test_model(task, algorithm)
        
        # Log the results
        if mean_return is not None:
            with open(log_file_path, "a") as log_file:
                log_file.write(f"{task}, {algorithm}, {mean_return}\n")
            print(f"Logged: {task}, {algorithm}, Mean Return: {mean_return}")
        else:
            print(f"Failed to retrieve mean return for {task} with {algorithm}")

print(f"Experiment complete. Results saved to {log_file_path}")
