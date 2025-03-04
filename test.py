from Board import Connect4
from Models import CNNModel
from MCTS import mcts, Node

import torch
from torch.amp import autocast

model = CNNModel()
model = model.to(torch.float16)
model.to('cuda')
board = Connect4()

best_move, search_tree = mcts(board, model, num_simulations=10)
print("Best move:", best_move)