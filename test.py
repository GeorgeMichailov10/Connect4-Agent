from Board import Connect4
from Models import CNNModel

import torch

model = CNNModel()
board = Connect4()

state_tensor = board.encode_state_cnn()
policy, value = model(state_tensor)