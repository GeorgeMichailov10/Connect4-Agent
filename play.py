import torch
import numpy as np
import pygame
from Connect4 import Connect4
from MCTS import MCTS, Config, Node
from Models import CNNModel

def load_model(config: Config, model_path='./cnn_134_845'):
    model = CNNModel(config).to(config.device)
    state_dict = torch.load(model_path, map_location=config.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def model_move(mcts, state):
    search_iters = mcts.config.mcts_max_search_iter
    action, root = mcts.search(state, search_iters)
    return action

def play_game():
    config = Config()
    game = Connect4()
    #model = load_model(config)
    #mcts = MCTS(model, Connect4(), config)

    board = game.reset()
    current_player = 1
    done = False

    while not done:
        print(board)
        if current_player == 1:
            valid_moves = game.get_valid_actions(board)
            print("Valid moves:", valid_moves)
            move = None
            while move not in valid_moves:
                try:
                    move = int(input("Enter column (0-indexed): "))
                except ValueError:
                    print("Invalid input. Please enter an integer.")
        else:
            valid_moves = game.get_valid_actions(board)
            move = np.random.choice(valid_moves)
            print(f"AI chooses column: {move}")

        board, reward, done = game.play_action(board, move, player=current_player)
        if board is None:
            print("Move not possible. Game over!")
            break
        board = -board  
        current_player *= -1

    print(board)
    outcome = game.evaluate(board)
    if outcome == 1:
        print("Player 1 wins!")
    elif outcome == -1:
        print("Player 2 wins!")
    else:
        print("It's a draw!")



def main():
    play_game()

if __name__ == '__main__':
    main()