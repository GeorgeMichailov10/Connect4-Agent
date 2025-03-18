from ..Connect4 import Connect4
import random
import json
import os
import re

class TDLAgent:
    def __init__(self, lam=1, learning_rate=0.1, discount_factor=1.0, explore_rate=0.1, value_function_path = None):
        self.value_function = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.explore_rate = explore_rate
        self.lam = lam
        self.game = Connect4()

        if value_function_path is not None:
            self.load_value_function(value_function_path)

    def state_to_key(self, state):
        return json.dumps(state.tolist())

    def save_value_function(self):
        safe_lam = str(self.lam).replace(".", "_")
        safe_lr = str(self.learning_rate).replace(".", "_")
        safe_df = str(self.discount_factor).replace(".", "_")
        safe_er = str(self.explore_rate).replace(".", "_")

        directory = "./Agents/TDL/TDL_Value_Functions"
        filename = f"l_{safe_lam}_lr_{safe_lr}_df_{safe_df}_er_{safe_er}.json"
        path = os.path.join(directory, filename)
        os.makedirs(directory, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.value_function, f)
        print(f"Value function saved to {path}")

    def load_value_function(self, path):
        filename = os.path.basename(path)
        match = re.search(r"l_([\d_]+)_lr_([\d_]+)_df_([\d_]+)_er_([\d_]+)", filename)

        if match:
            self.lam = float(match.group(1).replace("_", "."))
            self.learning_rate = float(match.group(2).replace("_", "."))
            self.discount_factor = float(match.group(3).replace("_", "."))
            self.explore_rate = float(match.group(4).replace("_", "."))
        else:
            raise ValueError(f"Invalid filename format: {filename}")

        with open(path, 'r') as f:
            self.value_function = json.load(f)

        print(f"Loaded value function and parameters: lam={self.lam}, lr={self.learning_rate}, df={self.discount_factor}, er={self.explore_rate}")
   
    def get_action(self, state):
        actions = self.game.get_valid_actions(state)
        if random.random() < self.explore_rate:
            return random.choice(actions)
        
        best_action = None
        best_value = float('-inf')
        for action in actions:
            next_state = self.game.get_next_state(state, action)
            next_state_key = self.state_to_key(next_state)
            next_state_value = self.value_function.get(next_state_key, 0.0)
            if next_state_value > best_value:
                best_value = next_state_value
                best_action = action
        
        return best_action if best_action is not None else random.choice(actions)

    def self_play(self, games):
        for game in range(games):
            print(f'Starting Game: {game + 1}')
            state = self.game.empty_board()
            done = False
            eligibility = {} # Note to self: Need this for the lambda

            while not done:
                current_state_key = self.state_to_key(state)
                eligibility.setdefault(current_state_key, 0.0)

                action = self.get_action(state)
                next_state, reward, done = self.game.play_action(state, action)

                next_state_key = self.state_to_key(next_state)
                next_state_value = reward if done else self.value_function.get(next_state_key, 0.0)

                current_state_value = self.value_function.get(current_state_key, 0.0)

                td_error = reward + self.discount_factor * next_state_value - current_state_value
                eligibility[current_state_key] += 1

                for st in list(eligibility.keys()):
                    self.value_function[st] = self.value_function.get(st, 0.0) + self.learning_rate * td_error * eligibility[st]
                    eligibility[st] *= self.discount_factor * self.lam

                    if eligibility[st] < 1e-5: # Remove small traces for efficiency
                        del eligibility[st]

                state = self.game.flip_board(next_state)
        self.save_value_function() 