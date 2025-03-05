from dataclasses import dataclass
from Connect4 import Connect4
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Config:
    device = 'cuda'
    exploration_constant = 2
    temperature = 1.25
    dirichlet_alpha = 1.
    dirichlet_epsilon = 0.25

    training_epochs = 150
    games_per_epoch = 100
    minibatch_size = 128
    n_minibatches = 4

    mcts_start_search_iter = 30
    mcts_max_search_iter = 150
    mcts_search_increment = 1

class MCTS:
    def __init__(self, model, game: Connect4, config: Config):
        self.model = model
        self.game = game
        self.config = config

    def get_model_evaluations(self, valid_actions, state_tensor):
        with torch.no_grad():
            self.model.eval()
            value, logits = self.model(state_tensor)

        action_probabilities = F.softmax(logits.view(self.game.cols), dim=0).cpu().numpy()
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * self.game.cols)
        action_probabilities = ((1 - self.config.dirichlet_epsilon) * action_probabilities) + self.config.dirichlet_epsilon * noise

        mask = np.full(self.game.cols, False)
        mask[valid_actions] = True
        action_probabilities = action_probabilities[mask]

        action_probabilities /= np.sum(action_probabilities)

        return action_probabilities, value

    def search(self, state, total_iterations, temperature=None):
        root = Node(None, state, 1, self.game, self.config)

        # Step 1: Get Model evaluations on each move
        valid_actions = self.game.get_valid_actions(state)
        state_tensor = torch.tensor(self.game.encode_state_cnn(state), dtype=torch.float).unsqueeze(0).to(self.config.device)

        action_probabilities, value = self.get_model_evaluations(valid_actions, state_tensor)

        # Step 2: Create a child for each action with its prior probability
        for action, probability in zip(valid_actions, action_probabilities):
            child_state = -self.game.get_next_state(state, action)
            root.children[action] = Node(root, child_state, -1, self.game, self.config)
            root.children[action].prob = probability


        root.node_visits = 1
        root.total_score = value.item()

        # Step 3: Search with similar process to what just done
        for _ in range(total_iterations):
            current_node = root
            while not current_node.is_leaf():
                current_node = current_node.select_child()

                if not current_node.is_terminal():
                    current_node.expand()
                    valid_actions = self.game.get_valid_actions(current_node.state)
                    state_tensor = torch.tensor(self.game.encode_state_cnn(current_node.state), dtype=torch.float).unsqueeze(0).to(self.config.device)

                    action_probabilities, value = self.get_model_evaluations(valid_actions, state_tensor)

                    

        



class Node:
    def __init__(self, parent, state, player: int, game: Connect4, config: Config):
        self.parent = parent
        self.state = state
        self.player = player
        self.config = config
        self.game = game

        self.prob = 0
        self.children = {}
        self.node_visits = 0
        self.total_score = 0

    def expand(self):
        valid_actions = self.game.get_valid_actions()

        # If no more valid actions, no more exploration; set value and return
        if len(valid_actions) == 0:
            self.total_score = self.game.evaluate(self.state)
            return
        
        # Create a child for each action
        for action in valid_actions:
            child_state = -self.game.get_next_state(self.state, action)
            self.children[action] = Node(self, child_state, -self.player, self.game, self.config)

    def select_child(self):
        best_puct = -np.inf
        best_child = None
        for child in self.children.values():
            puct = self.calculate_puct(child)
            if puct > best_puct:
                best_puct = puct
                best_child = child
        return best_child
    
    def backprop(self, value):
        self.total_score += value
        self.node_visits += 1
        if self.parent is not None:
            self.parent.backprop(-value)

    def is_leaf(self):
        return len(self.children) == 0
    
    def is_terminal(self):
        return (self.node_visits != 0) and (len(self.children) == 0)

    def get_value(self):
        if self.node_visits == 0:
            return 0
        return self.total_score / self.node_visits
    
    def __str__(self):
        return (f"State:\n{self.state}\nProb: {self.prob}\nTo play: {self.to_play}" +
                f"\nNumber of children: {len(self.children)}\nNumber of visits: {self.n_visits}" +
                f"\nTotal score: {self.total_score}")