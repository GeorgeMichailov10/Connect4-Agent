import numpy as np
from scipy.signal import convolve2d

class Connect4:
    def __init__(self):
        self.rows = 6
        self.cols = 7

    #--------- Public Use ------------------------------

    def get_valid_actions(self, state):
        """
        Returns a list of columns (0-indexed) where a move may be played.
        """
        if self._evaluate(state) != 0:
            return np.array([], dtype=int)
        return np.array([col for col in range(state.shape[1]) if state[0][col] == 0], dtype=int)
    
    def get_next_state(self, state, action, player_value=1):
        """
        Returns the board state (unflipped) of how the board would look if this action was played.
        """
        legal_moves = self.get_valid_actions(state)
        if action not in legal_moves:
            return None
        row = np.where(state[:, action] == 0)[0][-1]
        new_state = state.copy()
        new_state[row, action] = player_value
        return new_state
    
    def play_action(self, state, action, player=1):
        """
        Plays action on the board and returns the next state, 1, 0, or -1 depending on result (0 if not done), and bool done.
        """
        next_state = self.get_next_state(state, action, player)
        if next_state is None:
            return None, 0, True
        game_score = self._evaluate(next_state)
        done = True if game_score != 0 or len(self.get_valid_actions(next_state)) == 0 else False
        return next_state, game_score, done
    
    def flip_board(self, state):
        """
        Returns flipped state of board.
        """
        return -state
     
    def empty_board(self):
        """
        Returns a blank starting state board.
        """
        return np.zeros([self.rows, self.cols], dtype=np.int8)

    #--------- Private Use -----------------------------
    
    def _evaluate(self, state):
        filter = np.ones((1, 4), dtype=int)
        horizontal_check = convolve2d(state, filter, mode='valid')
        vertical_check = convolve2d(state, filter.T, mode='valid')

        diagonal_filter = np.eye(4, dtype=int)
        diagonal1_check = convolve2d(state, diagonal_filter, mode='valid')
        diagonal2_check = convolve2d(state, np.fliplr(diagonal_filter), mode='valid')

        if any(cond.any() for cond in [horizontal_check == 4, vertical_check == 4, diagonal1_check == 4, diagonal2_check == 4]):
            return 1
        elif any(cond.any() for cond in [horizontal_check == -4, vertical_check == -4, diagonal1_check == -4, diagonal2_check == -4]):
            return -1
        return 0
    
    # ---------- Unused functions at the moment ------------------------
    
    def encode_state_cnn(self, state):
        three_channel_state = np.stack((state == 1, state == 0, state == -1)).astype(np.float16)
        if len(state.shape) == 3:
            three_channel_state = np.swapaxes(three_channel_state, 0, 1)
        return three_channel_state
    
    def encode_state_transformer(self, state):
        three_channel_state = np.stack((state == 1, state == 0, state == -1)).astype(np.float16)
        tokens = np.moveaxis(three_channel_state, 0, -1)
        tokens = tokens.reshape(-1, 3)
        return tokens  # Returns (self.rows * self.cols, 3)