import matplotlib.pyplot as plt
import numpy as np
import json

def create_training_loss_plots():
    """
    Create training loss curves for lift, stack, and sort tasks using actual training data
    """
    # Actual training data from logs
    tasks = {
        'lift': {
            'epochs': list(range(10, 101, 10)),  # Every 10 epochs: 10,20,30,...,100
            'train_loss': [1.0352, 1.0293, 1.0300, 1.0280, 1.0275, 1.0312, 1.0283, 1.0286, 1.0282, 1.0274],
            'val_loss': [],  # No separate validation loss in logs
            'final_loss': 1.0274,
            'task_name': 'Lift Task'
        },
        'stack': {
            'epochs': list(range(10, 101, 10)),  # Every 10 epochs: 10,20,30,...,100
            'train_loss': [0.9956, 0.9931, 0.9927, 0.9924, 0.9924, 0.9924, 0.9923, 0.9922, 0.9923, 0.9921],
            'val_loss': [],  # No separate validation loss in logs
            'final_loss': 0.9921,
            'task_name': 'Stack Task'
        },
        'sort': {
            'epochs': list(range(10, 101, 10)),  # Every 10 epochs: 10,20,30,...,100
            'train_loss': [1.0120, 1.0115, 1.0112, 1.0111, 1.0110, 1.0110, 1.0111, 1.0110, 1.0110, 1.0109],
            'val_loss': [],  # No separate validation loss in logs
            'final_loss': 1.0109,
            'task_name': 'Sort Task'
        }
    }

    # Create subplots - adjust based on available data
    num_tasks = sum(1 for task in tasks.values() if task['train_loss'])
    if num_tasks == 0:
        print("No training data available. Please provide training logs.")
        return

    fig, axes = plt.subplots(1, num_tasks, figsize=(6*num_tasks, 5))
    if num_tasks == 1:
        axes = [axes]  # Make it iterable

    fig.suptitle('Real Robot Training Loss Curves - ACT Policy', fontsize=16, fontweight='bold')

    colors = ['blue', 'red', 'green']
    plot_idx = 0

    for i, (task_key, task_data) in enumerate(tasks.items()):
        if not task_data['train_loss']:  # Skip tasks without data
            continue

        ax = axes[plot_idx]

        # Plot training loss
        ax.plot(task_data['epochs'], task_data['train_loss'],
               label='Training Loss', color=colors[i], linewidth=2, marker='o', markersize=4)

        # If validation loss is available, plot it too
        if task_data['val_loss']:
            ax.plot(task_data['epochs'], task_data['val_loss'],
                   label='Validation Loss', color=colors[i], linestyle='--', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{task_data["task_name"]}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add final loss annotation
        final_loss = task_data['final_loss']
        ax.annotate(f'Final Loss: {final_loss:.4f}',
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

        plot_idx += 1

    plt.tight_layout()
    plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_training_data_from_json(json_files):
    """
    Load training data from JSON files if available
    """
    training_data = {}

    for task, json_file in json_files.items():
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                training_data[task] = data
        except FileNotFoundError:
            print(f"Warning: {json_file} not found for {task} task")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {json_file}")

    return training_data

if __name__ == "__main__":
    # Example usage with JSON files (uncomment and modify paths when you have real data)
    # json_files = {
    #     'lift': 'checkpoints/lift_act/training_history.json',
    #     'stack': 'checkpoints/stack_act/training_history.json',
    #     'sort': 'checkpoints/sort_act/training_history.json'
    # }
    # training_data = load_training_data_from_json(json_files)

    # Create plots with example data
    create_training_loss_plots()