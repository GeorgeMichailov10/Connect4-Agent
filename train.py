from Agents.Connect4 import Connect4
from Agents.QLearning.QLearning import QLearningAgent

agent = QLearningAgent(learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2)
agent.self_play(games=100000)