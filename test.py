from Board import Connect4
from Models import CNNModel

import torch
from torch.amp import autocast

model = CNNModel()
model = model.to(torch.bfloat16)
board = Connect4()

state_tensor = board.encode_state_cnn()


with autocast(device_type='cpu', dtype=torch.bfloat16):
    policy, value = model(state_tensor)

print(policy, value)