from AlphaZero import AlphaZero
import random
import torch
import numpy as np

class Evaluator:
    def __init__(self, alphazero: AlphaZero, num_examples=500, verbose=True):
        self.model = alphazero.model
        self.game = alphazero.game
        self.config = alphazero.config
        self.accuracies = []
        self.num_examples = num_examples
        self.verbose = verbose

        self.generate_examples()

    def select_action(self, state):
        valid_actions = self.game.get_valid_actions(state)
        if len(valid_actions) == 0:
            return None
        
        for action in valid_actions:
            next_state, reward, _ = self.game.play_action(state, action)
            if reward == 1:
                return action

        flipped_state = -state
        for action in valid_actions:
            next_state, reward, _ = self.game.play_action(flipped_state, action)
            if reward == 1:
                return action

        return random.choice(valid_actions)

    def generate_examples(self):
        winning_examples = self.generate_examples_for_condition('win')
        blocking_examples = self.generate_examples_for_condition('block')

        winning_example_states, winning_example_actions = zip(*winning_examples)
        blocking_example_states, blocking_example_actions = zip(*blocking_examples)

        target_states = np.concatenate([winning_example_states, blocking_example_states], axis=0)
        target_actions = np.concatenate([winning_example_actions, blocking_example_actions], axis=0)

        encoded_states = [self.game.encode_state_cnn(state) for state in target_states]
        self.X_target = torch.tensor(np.stack(encoded_states, axis=0), dtype=torch.float).to(self.config.device)
        self.y_target = torch.tensor(target_actions, dtype=torch.long).to(self.config.device)

    def generate_examples_for_condition(self, condition):
        examples = []
        while len(examples) < self.num_examples:
            state = self.game.reset()
            while True:
                action = self.select_action(state)
                if action is None:
                    break
                next_state, reward, done = self.game.play_action(state, action, player=1)                
                if condition == 'win' and reward == 1:
                    examples.append((state, action))
                    break                
                if done:
                    break
                state = next_state
                action = self.select_action(-state)
                if action is None:
                    break
                next_state, reward, done = self.game.play_action(state, action, player=-1)               
                if condition == 'block' and reward == -1:
                    examples.append((-state, action))
                    break                
                if done:
                    break
                state = next_state
        return examples

    def evaluate(self):
        with torch.no_grad():
            self.model.eval()
            _, logits = self.model(self.X_target)
            pred_actions = logits.argmax(dim=1)
            accuracy = (pred_actions == self.y_target).float().mean().item()     
        self.accuracies.append(accuracy)
        if self.verbose:
            print(f"Initial Evaluation Accuracy: {100 * accuracy:.1f}%")
        return int(1000 * accuracy)    