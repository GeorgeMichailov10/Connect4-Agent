import ray
import numpy as np
import torch
import torch.nn as nn
from Connect4 import Connect4
from Evaluator import Evaluator
from MCTS import MCTS, Config
from Models import CNNModel

@ray.remote
class SampleGeneratorWorker:
    def __init__(self, config: Config, model_state_dict_ref):
        self.config = config
        self.model = CNNModel().to(self.config.device)
        self.model.load_state_dict(model_state_dict_ref)
        self.game = Connect4()
        self.mcts = MCTS(self.model, self.game, config)

    def self_play(self):
        pass

    def update_model(self, model_state_dict_ref):
        self.model.load_state_dict(ray.get(model_state_dict_ref))
        self.mcts = MCTS(self.model, self.game, self.config)

class AlphaZero2:
    def __init__(self, game: Connect4, config: Config, verbose=True):
        self.model = CNNModel().to(config.device)
        self.mcts = MCTS(self.model, game, config)
        self.game = game
        self.config = config

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

        ray.init(ignore_reinit_error=True)
        self.num_workers = self.config.cpu_procs
        model_state_ref = ray.put(self.model.state_dict())
        self.workers = [
            SampleGeneratorWorker.remote(config, model_state_ref)
            for _ in range(self.num_workers)
        ]

    # TODO: Save model
    def train(self, epochs):
        for epoch in range(epochs):
            futures = [worker.self_play_loop.remote() for worker in self.workers]
            results = ray.get(futures)
            for worker_samples in results:
                for sample in worker_samples:
                    self.append_to_memory(*sample)
                    if self.memory_full:
                        self.learn()

                        model_state_ref = ray.put(self.model.state_dict())
                        update_futures = [worker.update_model.remote(model_state_ref) for worker in self.workers]
                        ray.get(update_futures)
            self.search_iterations = min(
                self.config.mcts_max_search_iter, 
                self.search_iterations + self.config.mcts_search_increment
            )

    def append_to_memory(self, states_stack, value, visits_stack):
        state_tensor = torch.tensor(states_stack, dtype=torch.float).to(self.device)
        visits_tensor = torch.tensor(visits_stack, dtype=torch.float).to(self.device)
        value_tensor = torch.tensor([value, value], dtype=torch.float).to(self.device).unsqueeze(1)
        
        self.state_memory[self.current_memory_index:self.current_memory_index+2] = state_tensor
        self.value_memory[self.current_memory_index:self.current_memory_index+2] = value_tensor
        self.policy_memory[self.current_memory_index:self.current_memory_index+2] = visits_tensor
        
        self.current_memory_index = (self.current_memory_index + 2) % self.max_memory
        if self.current_memory_index in [0, 1]:
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