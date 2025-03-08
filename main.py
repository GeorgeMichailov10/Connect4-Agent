from AlphaZero import AlphaZero
from Connect4 import Connect4
from Evaluator import Evaluator
from MCTS import MCTS, Config

import torch
import matplotlib.pyplot as plt
import numpy as np

config = Config()
game = Connect4()
alphazero = AlphaZero(game, config)
evaluator = Evaluator(alphazero)

# Evaluate pre training
evaluator.evaluate()

# Main training/eval loop
for epoch in range(config.training_epochs):
    alphazero.train()
    accuracy = evaluator.evaluate()
    if accuracy > 85.0:
        torch.save(alphazero.model.state_dict(), f'model_epoch{epoch}_accuracy_{int(accuracy * 10)}.pth')



x_values = np.linspace(0, 101 * len(evaluator.accuracies), len(evaluator.accuracies))
y_values = [acc * 100 for acc in evaluator.accuracies]

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, linewidth=2, marker='o', markersize=4, linestyle='-', color='#636EFA')

# Formatting
plt.xlabel('\nNumber of Games', fontsize=16)
plt.ylabel('Policy Evaluation Accuracy (%)', fontsize=16)
plt.title('Policy Evaluation\n', fontsize=24)
plt.grid(True, linestyle='--', linewidth=0.5, color='gray')

plt.show()