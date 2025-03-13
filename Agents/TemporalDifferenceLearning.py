from Connect4 import Connect4

class TDLAgent:
    def __init__(self, learning_rate=0.1, discount_factor=1.0, explore_rate=0.1):
        self.value_function = {}