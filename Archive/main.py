from AlphaZero import AlphaZero
from AlphaZero2 import AlphaZero2
from Connect4 import Connect4
from Evaluator import Evaluator
from MCTS import MCTS, Config

import torch
import matplotlib.pyplot as plt
import numpy as np

config = Config()
game = Connect4()
alphazero = AlphaZero2(game, config)


for epoch in range(config.training_epochs):
    alphazero.train()