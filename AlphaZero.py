from Connect4 import Connect4
from MCTS import MCTS, Config
from Models import CNNModel

import torch
import torch.nn as nn
import numpy as np
import concurrent.futures as cf
import threading

class AlphaZero:
    def __init__(self, game: Connect4, config: Config, verbose=True):
        self.model_mutex = threading.Lock()
        self.memory_mutex = threading.Lock()
        self.total_games_mutex = threading.Lock()

        self.config = config
        self.model = CNNModel(config).to(config.device)
        self.game = [Connect4() for _ in range(self.config.cpu_threads)]
        self.mcts = [MCTS(self.model, self.game[instance], config, self.model_mutex) for instance in range(self.config.cpu_threads)]

        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss(reduction='mean')
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=0.0001)

        # Preallocate memory on gpu
        state_shape = game.encode_state_cnn(game.reset()).shape
        self.max_memory = config.minibatch_size * config.n_minibatches
        self.state_memory = torch.zeros(self.max_memory, *state_shape).to(config.device)
        self.value_memory = torch.zeros(self.max_memory, 1).to(config.device)
        self.policy_memory = torch.zeros(self.max_memory, game.cols).to(config.device)
        self.current_memory_index = 0
        self.memory_full = False

        self.search_iterations = config.mcts_start_search_iter

        self.verbose = verbose
        self.total_games = 0

    def train(self):
        with cf.ThreadPoolExecutor(max_workers=self.config.cpu_threads) as executor:
            futures = [executor.submit(self.self_play, thread_id) for thread_id in range(self.config.cpu_threads)]
            cf.wait(futures)
        self.learn()
        self.search_iterations = min(self.config.mcts_max_search_iter, self.search_iterations + self.config.mcts_search_increment)

    def self_play(self, thread_id):
        state = self.game[thread_id].reset()
        done = False
        while not done:
            action, root = self.mcts[thread_id].search(state, self.search_iterations)
            value = root.get_value()
            visits = np.zeros(self.config.n_cols)
            for child_action, child in root.children.items():
                visits[child_action] = child.node_visits
            visits /= np.sum(visits)
            self.append_to_memory(state, value, visits)

            if self.memory_full:
                return

            state, _, done = self.game.play_action(state, action)
            state = -state

        with self.total_games_mutex:
            self.total_games += 1

    def append_to_memory(self, state, value, visits):
        encoded_state = np.array(self.game.encode_state_cnn(state))
        encoded_state_augmented = np.array(self.game.encode_state_cnn(state[:, ::-1]))

        states_stack = np.stack((encoded_state, encoded_state_augmented), axis=0)
        visits_stack = np.stack((visits, visits[::-1]), axis=0)

        state_tensor = torch.tensor(states_stack, dtype=torch.float).to(self.config.device)
        visits_tensor = torch.tensor(visits_stack, dtype=torch.float).to(self.config.device)
        value_scalar = value.cpu().item()
        value_tensor = torch.tensor([value_scalar, value_scalar], dtype=torch.float).to(self.config.device).unsqueeze(1)

        with self.memory_mutex:
            if self.memory_full:
                return
            
            self.state_memory[self.current_memory_index:self.current_memory_index + 2] = state_tensor
            self.value_memory[self.current_memory_index:self.current_memory_index + 2] = value_tensor
            self.policy_memory[self.current_memory_index:self.current_memory_index + 2] = visits_tensor
            self.current_memory_index = (self.current_memory_index + 2) % self.max_memory
            if (self.current_memory_index == 0) or (self.current_memory_index == 1):
                self.memory_full = True

    def learn(self):
        self.model.train()

        batch_indices = np.arange(self.max_memory)
        np.random.shuffle(batch_indices)

        for batch_index in range(self.config.n_minibatches):
            start = batch_index * self.config.minibatch_size
            end = start + self.config.minibatch_size
            mb_indices = batch_indices[start:end]
            mb_states = self.state_memory[mb_indices]
            mb_value_targets = self.value_memory[mb_indices]
            mb_policy_targets = self.policy_memory[mb_indices]
            value_preds, policy_logits = self.model(mb_states)
            policy_loss = self.loss_ce(policy_logits, mb_policy_targets)
            value_loss = self.loss_mse(value_preds.view(-1), mb_value_targets.view(-1))
            loss = policy_loss + value_loss
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        self.memory_full = False
        self.model.eval()

        