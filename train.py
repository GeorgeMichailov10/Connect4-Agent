from Agents.Connect4 import Connect4
from Agents.TemporalDifferenceLearning import TDLAgent

agent = TDLAgent(lam=5, learning_rate=0.01, discount_factor=0.9, explore_rate=0.1)
agent.self_play(games=10000)