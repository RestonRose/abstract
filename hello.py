import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import copy

# Function to load JSON data
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

# Function to plot a single solution matrix
def plot_one_solution(ax, input_matrix, title):
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    ax.set_title(title)

# Function to plot a challenge (input and output)
def plot_one_challenge(ax, task, train_or_test, input_or_output, index):
    input_matrix = task[train_or_test][index][input_or_output]
    plot_one_solution(ax, input_matrix, f'{train_or_test} {input_or_output}')

# Function to plot the training and test data of a task
def plot_task(task, title):
    num_train = len(task['train'])
    num_test = len(task['test'])

    fig, axs = plt.subplots(2, num_train, figsize=(3 * num_train, 6))
    plt.suptitle(f'{title}:', fontsize=20, fontweight='bold', y=1)

    # Plot train input and output
    for j in range(num_train):
        plot_one_challenge(axs[0, j], task, 'train', 'input', j)
        plot_one_challenge(axs[1, j], task, 'train', 'output', j)

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')
    plt.tight_layout()
    plt.show()

# Function to plot the test input and corresponding solution
def plot_solution(task, solution, title):
    fig, axs = plt.subplots(2, 1, figsize=(3, 6))
    plt.suptitle(f'{title}:', fontsize=20, fontweight='bold', y=1)

    # Plot the test input
    plot_one_challenge(axs[0], task, 'test', 'input', 0)

    # Plot the solution output
    solution_matrix = solution[0]
    plot_one_solution(axs[1], solution_matrix, 'Solution output')

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')
    plt.tight_layout()
    plt.show()

# Load data
training_challenges = load_json('arc-agi_training_challenges.json')
training_solutions = load_json('arc-agi_training_solutions.json')
evaluation_challenges = load_json('arc-agi_evaluation_challenges.json')
evaluation_solutions = load_json('arc-agi_evaluation_solutions.json')
test_challenges = load_json('arc-agi_test_challenges.json')

# Prepare colormap and normalization
cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF']
)
norm = colors.Normalize(vmin=0, vmax=10)

# Example usage: Plot the first training challenge and its solution
for i in range(0, 1):
    task_name = list(training_solutions.keys())[i]
    task = training_challenges[task_name]
    solution = training_solutions[task_name]
    
    plot_task(task, "Training Challenge")
    plot_solution(task, solution, "Training Solution")