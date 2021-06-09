from board import Game
from agent import Agent
from humanAgent import HumanAgent
from randomAgent import RandomAgent

from multiprocessing import Process

from MCTS import *
from NeuralNetwork import Connect4Zero

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#AI VS RANDOM 
#Function to play Human vs AlphaZero Q
def main():
    builder = Connect4Zero()
    #Load the desired model
    model = builder.load_model('nn_weights/d5-999.h5')
    model.summary()
    
    random_wins = 0
    ai_wins = 0
    
    first_ai = True
    #Uncomment if you want AlphaZero Q to play second
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
                action, _, _, _ = mcts.makeMove(board, model)
            else:
                player.update_board(board)
                action = player.pick_move()
                
        else: 
            if board.current_player == 1:
                action, _, _, _ = mcts.makeMove(board, model)
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

'''
# AI VS AI 
#Function to play AlphaZero Q vs AlphaZero Q
NUM_PARTIDAS = 30
    
def main():
    builder = Connect4Zero()
    
    #Load the models
    model1 = builder.load_model('nn_weights/d5-350.h5')
    model2 = builder.load_model('nn_weights/d5-999.h5')
    
    first_player_wins = 0
    second_player_wins = 0
    
    for i in range(0, NUM_PARTIDAS):
        #First, we create an empty board
        board = Game()
        
        evaluation1 = model1.predict([board.get_state()]) 
        mcts1 = MCTS(evaluation1[0][0])
        evaluation2 = model2.predict([board.get_state()]) 
        mcts2 = MCTS(evaluation2[0][0])
        
        finished = False
        draw = False
        while not finished and not draw:            
            if board.current_player == 0:
                action, state, mcts_distribution, _ = mcts1.makeMove(board, model1)
            else:
                action, state, mcts_distribution, _ = mcts2.makeMove(board, model2)
                
            finished = board.make_move(action)
            if not finished and not board.is_possible_to_move():
                draw = True
                
            if board.current_player == 0:
                mcts1.update_board(board, action, model1)
            else:
                mcts2.update_board(board, action, model2)
        
        if finished:
            if board.current_player == 0:
                second_player_wins += 1
                print('Second player wins')
            else:
                first_player_wins += 1
                print('First player wins')
        else:
            print('Draw')
            
        board.print_board()
        print('FIRST: ', first_player_wins, ' SECOND: ', second_player_wins)
'''

if __name__ == "__main__":
    main()
