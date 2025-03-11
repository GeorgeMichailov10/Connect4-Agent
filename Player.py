from Models import CNNModel
from MCTS import MCTS, Config
from Connect4 import Connect4

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Agent:
    def __init__(self, config:Config, model_weights_path, difficulty=None):
        self.config = config
        self.model = CNNModel(self.config).to(self.config.device)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.config.device))
        self.model.eval()
        self.game = Connect4()
        self.mcts = MCTS(self.model, self.game, self.config)

        self.set_difficulty(difficulty)

    def set_difficulty(self, difficulty):
        if difficulty == 'easy':
            self.search_iterations = 100
        elif difficulty == 'medium':
            self.search_iterations = 200
        elif difficulty == 'hard':
            self.search_iterations = 300
        else:
            self.search_iterations = None

    def select_action(self, state):
        if self.search_iterations is None:
            return self.select_action_without_mcts(state)
        else:
            return self.select_action_with_mcts(state)

    def select_action_without_mcts(self, state):
        state_tensor = torch.tensor(self.game.encode_state_cnn(state), dtype=torch.float).to(self.config.device)
        with torch.no_grad():
            value, policy = self.model(state_tensor)

        action_probs = F.softmax(policy.view(-1), dim=0).cpu().numpy()
        valid_actions = self.game.get_valid_actions(state)
        valid_action_probs = action_probs[valid_actions]

        best_action = valid_actions[np.argmax(valid_action_probs)]
        return best_action

    def select_action_with_mcts(self, state):
        action, _ = self.mcts.search(state, self.search_iterations)
        return action

