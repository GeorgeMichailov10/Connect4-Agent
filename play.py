from Agents.Connect4 import Connect4
from Agents.TemporalDifferenceLearning import TDLAgent



agent = TDLAgent(value_function_path='./Agents/TDL_Value_Functions/l_15_lr_0_1_df_0_95_er_0_1.json')

state = Connect4().empty_board()
done = False
turn  = 0

while not done:
    print(state)
    if turn == 0:
        print('Human move')
        action = int(input("Enter a valid move: ")) - 1 # Not 0 indexed for simplicity atm
    else:
        print('Agent thinking')
        action = agent.get_action(state)
        print("MODEL CHOSE:", action)

    next_state, reward, done = Connect4().play_action(state, action)
    state = Connect4().flip_board(next_state)

    if done == True:
        print(f'Game over: {reward}')
    else:
        turn = 1 - turn
    print(turn)