from agent import Agent
from board import Game

# Allows a human to play against AlphaZero
class HumanAgent(Agent):
    
    def update_board(self, board : Game):
        self.board = board
        
    def pick_move(self):
        valid_action = True
        
        try:
            action = int(input('Choose a column to place the token (0 to 6): '))
            if action not in self.board.available_moves():
                valid_action = False
        except ValueError:
            valid_action = False
        
        
        while valid_action == False:
            print('The chosen action is not possible.')
            try:
                action = int(input('Please, choose a valid column: '))
                if action in self.board.available_moves():
                    valid_action = True
            except ValueError:
                continue
        
        return action
        
        