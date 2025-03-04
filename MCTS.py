import math
import torch
import torch.nn as nn
import numpy as np
import copy

from Board import Connect4

class Node:
    def __init__(self, state: Connect4, prior, parent=None, move=None):
        self.state = state
        self.prior = prior
        self.parent = parent
        self.children = {}
        self.move = move
        self.NODES = 0
        self.EVALUATION = 0.0
        self.MEAN = 0.0

    def is_terminal(self):
        return len(self.state.get_valid_moves()) == 0

    def legal_moves(self):
        return self.state.get_valid_moves()

    # Build the tree
    def expand(self, model):
        if self.is_terminal():
            return

        moves = self.legal_moves()
        state_tensor = self.state.encode_state_cnn()
        state_tensor = state_tensor.to('cuda')
        with torch.no_grad():
            policy_logits, _ = model(state_tensor)
        policy = torch.exp(policy_logits).squeeze(0).cpu().numpy()
        for move in moves:
            if move not in self.children:
                new_state = copy.deepcopy(self.state)
                new_state.make_move(move)
                self.children[move] = Node(
                    state=new_state,
                    prior=policy[move],
                    parent=self,
                    move=move
                )

    def select_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_move = None
        best_child = None

        total_N = sum(child.NODES for child in self.children.values()) # Exploration bonus
        for move, child in self.children.items():
            # PUCT: Q + c_puct * p * sqrt(total_N) / (1 + N)
            curr_score = child.MEAN + c_puct * child.prior * math.sqrt(total_N) / (1 + child.NODES)
            if curr_score > best_score:
                best_score = curr_score
                best_move = move
                best_child = child
        return best_move, best_child
        
    def backprop(self, value):
        self.NODES += 1
        self.EVALUATION += value
        self.MEAN = self.EVALUATION / self.NODES
        if self.parent is not None:
            self.parent.backprop(-value)
    
def mcts(root_state: Connect4, model, num_simulations=100, c_puct=1.0):
    root = Node(state=root_state, prior=1.0)
    root.expand(model)

    for epoch in range(num_simulations):
        node = root
        while node.children:
            _, node = node.select_child(c_puct)

            if not node.is_terminal() and not node.children:
                node.expand(model)

        if node.is_terminal():
            value = node.state.get_winner(col=node.move)
        else:
            state_tensor = node.state.encode_state_cnn()
            state_tensor = state_tensor.to('cuda')
            with torch.no_grad():
                _, value_tensor = model(state_tensor)
            value = value_tensor.item()
        node.backprop(value) 
    best_move = max(root.children.items(), key=lambda item: item[1].NODES)[0]
    return best_move, root   
