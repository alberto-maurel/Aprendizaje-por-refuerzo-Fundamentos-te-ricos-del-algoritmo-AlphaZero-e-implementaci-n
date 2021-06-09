from agent import Agent
from board import Game
from random import randint
from time import sleep

#Agent that just picks random movements
class RandomAgent(Agent):
    
    def update_board(self, board : Game):
        self.board = board
        
    def pick_move(self):
        #It just randomly picks one of the available actions
        sleep(0.5)
        moves = self.board.available_moves()
        action = moves[randint(0, len(moves) - 1)]
        
        return action
