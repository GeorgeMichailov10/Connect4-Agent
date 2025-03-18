from ..Connect4 import Connect4
import numpy as np
import json
import os
import re

class QLearningAgent:
    def __init__(self, learning_rate = 0.1, discount_factor=0.95, exploration_rate=0.1, value_function_path=None):
        self.Q_table = {}
        self.game = Connect4()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.explore_rate = exploration_rate
        
        if value_function_path is not None:
            self.load_value_function(value_function_path)

    def state_to_key(self, state):
        return json.dumps(state.tolist())

    def load_value_function(self, path):
        filename = os.path.basename(path)
        match = re.search(r"lr_([\d_]+)_df_([\d_]+)_er_([\d_]+)", filename)

        if match:
            self.learning_rate = float(match.group(1).replace("_", "."))
            self.discount_factor = float(match.group(2).replace("_", "."))
            self.explore_rate = float(match.group(3).replace("_", "."))
        else:
            raise ValueError(f"Invalid filename format: {filename}")

        with open(path, 'r') as f:
            loaded_table = json.load(f)
            self.Q_table = {k: np.array(v) for k, v in loaded_table.items()}

        print(f"Loaded value function and parameters: lr={self.learning_rate}, df={self.discount_factor}, er={self.explore_rate}")

    def save_value_function(self):
        safe_lr = str(self.learning_rate).replace(".", "_")
        safe_df = str(self.discount_factor).replace(".", "_")
        safe_er = str(self.explore_rate).replace(".", "_")

        directory = "./Agents/QLearning/QL_Value_Functions"
        filename = f"lr_{safe_lr}_df_{safe_df}_er_{safe_er}.json"
        path = os.path.join(directory, filename)
        os.makedirs(directory, exist_ok=True)

        serializable_Q_table = {k: v.tolist() for k, v in self.Q_table.items()}
        with open(path, "w") as f:
            json.dump(serializable_Q_table, f)
        print(f"Value function saved to {path}")

    def get_q_values(self, state):
        key = self.state_to_key(state)
        if key not in self.Q_table:
            self.Q_table[key] = np.zeros(self.game.cols)
        return self.Q_table[key]
    
    def get_action(self, state):
        valid_actions = self.game.get_valid_actions(state)
        if len(valid_actions) == 0:
            return None
        
        if np.random.rand() < self.explore_rate:
            return np.random.choice(valid_actions)
        
        q_values = self.get_q_values(state)
        valid_q = {action: q_values[action] for action in valid_actions}
        best_action = max(valid_q, key=valid_q.get)
        return best_action
    
    def update(self, state, action, reward, next_state, done):
        q_values = self.get_q_values(state)
        if done or next_state is None:
            target = reward
        else:
            next_q_values = self.get_q_values(next_state)
            target = reward + self.discount_factor * np.max(next_q_values)
        q_values[action] += self.learning_rate * (target - q_values[action])

    def self_play(self, games, decay_rate=0.95):
        for game in range(games):
            state = self.game.empty_board()
            done = False

            while not done:
                action = self.get_action(state)
                next_state, reward, done = self.game.play_action(state, action)
                self.update(state, action, reward, next_state, done)
        
                state = self.game.flip_board(next_state)
            self.explore_rate = max(0.01, self.explore_rate * decay_rate)
            if (game + 1) % 1000 == 0:
                print(f"Episode {game + 1}/{games} completed.")
        self.save_value_function()








