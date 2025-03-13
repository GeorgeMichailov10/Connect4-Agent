from Agents.Connect4 import Connect4
from Agents.TemporalDifferenceLearning import TDLAgent

agent = TDLAgent(lam=10, learning_rate=0.001, discount_factor=0.8, explore_rate=0.2)
agent.self_play(games=100000)