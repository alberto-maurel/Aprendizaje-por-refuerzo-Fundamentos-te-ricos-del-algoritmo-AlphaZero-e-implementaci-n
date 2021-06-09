from board import Game
from agent import Agent
from humanAgent import HumanAgent
from randomAgent import RandomAgent
from MCTS import *
from NeuralNetwork import Connect4Zero

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#AI VS RANDOM

def main():
    builder = Connect4Zero()
    
    #Load the weights of the model we are going to play against
    model = builder.load_model('nn_weights_c3/t4-125.h5')
    
    random_wins = 0
    ai_wins = 0
    
    first_ai = True
    #Uncomment if AlphaZero plays second
    #first_ai = False    
    
    board = Game()
    evaluation = model.predict([board.get_state()])    
    mcts = MCTS(evaluation[0][0])
    player = HumanAgent()
    
    finished = False
    draw = False
    while not finished and not draw: 
        board.print_board()           
        print('pol: ', model.predict([board.get_state()])[0][0])
        print('val: ', model.predict([board.get_state()])[1][0][0])
        if first_ai:
            if board.current_player == 0:
                action, state, mcts_distribution = mcts.makeMove(board, model)
            else:
                player.update_board(board)
                action = player.pick_move()
                
        else: 
            if board.current_player == 1:
                action, state, mcts_distribution = mcts.makeMove(board, model)
            else:
                player.update_board(board)
                action = player.pick_move()
                
        finished = board.make_move(action)
        if not finished and not board.is_possible_to_move():
            draw = True
        
        if first_ai and board.current_player == 0 or not first_ai and board.current_player == 1:
            mcts.update_board(board, action, model)
    
    if finished:
        if board.current_player == 0:
            print('Second player wins')
        else:
            print('First player wins')
    else:
        print('Draw')
        
    board.print_board()
    
if __name__ == "__main__":
    main()
