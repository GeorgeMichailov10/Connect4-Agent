class Connect4:
    def __init__(self):
        self.board = [[0 for _ in range(7)] for _ in range(6)]
        self.current_player = 1

    def make_move(self, col):
        if col not in self.get_valid_moves():
            raise ValueError("Illegal Move")
        
        row = 5
        while row >= 0 and self.board[row][col] != 0:
            row += 1
        
        self.board[row][col] = self.current_player

        game_over, winner = self.check_win(row-1, col)
        if game_over:
            return winner
        else:
            self.switch_player()
            return None

    def check_win(self, row, col):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and self.board[r][c] == self.current_player:
                count += 1
                r += dr
                c += dc

            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and self.board[r][c] == self.current_player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 4:
                return True, self.current_player
        
        if all(self.board[0][c] != 0 for c in range(7)):
            return True, 0
        
        return False, None

    def switch_player(self):
        self.current_player *= -1
        self.board *= -1

    def get_valid_moves(self):
        return [col for col in range(7) if self.board[0][col] == 0]
    
    def encode_state_cnn(self):
        return self.board
    
    def encode_state_transformer(self):
        return [cell for row in self.board for cell in row]