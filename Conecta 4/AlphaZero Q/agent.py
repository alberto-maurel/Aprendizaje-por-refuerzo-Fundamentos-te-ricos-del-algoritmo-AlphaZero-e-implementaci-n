from abc import abstractmethod

class Agent:
    
    def __init__(self):
        pass
    
    @abstractmethod
    def update_board(self, board):
        pass
        
    @abstractmethod
    def pick_move(self):
        pass
