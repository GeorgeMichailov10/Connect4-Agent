from Player import Agent
from Connect4 import Connect4
from MCTS import Config

config = Config()
agent = Agent(config, './models/cnn_134_845.pth', difficulty='hard')

state = Connect4().reset()
done = False
turn  = 0

while not done:
    print(state)
    if turn == 0:
        print('Human move')
        action = int(input("Enter a valid move"))
    else:
        print('Agent thinking')
        action = agent.select_action(state)
        print("MODEL CHOSE:", action)

    next_state, reward, done = Connect4().play_action(state, action)
    state = Connect4().flip_board(next_state)

    if done == True:
        print('Game over')
    else:
        turn = 1 - turn
    print(turn)