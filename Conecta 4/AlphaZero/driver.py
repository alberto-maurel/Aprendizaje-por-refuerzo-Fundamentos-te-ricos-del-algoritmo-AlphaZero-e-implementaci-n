from board import Game
from agent import Agent
from humanAgent import HumanAgent
from randomAgent import RandomAgent
from MCTS import MCTS
from NeuralNetwork import Connect4Zero

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

'''
Function to play the game between 2 agents
def play_game(player1 : Agent, player2: Agent):
    #En primer lugar creamos un juego en la posición inicial
    board = Game()
    
    while True:
        board.print_board()
        player1.update_board(board)
        action = player1.pick_move()
        finished = board.make_move(action)
        
        if finished == True:
            board.print_board()
            print('Player 1 wins')
            break
        
        if board.is_possible_to_move() == False:
            board.print_board()
            print('Draw')
            break
        
        board.print_board()
        player2.update_board(board)
        action = player2.pick_move()
        finished = board.make_move(action)
        
        if finished == True:
            board.print_board()
            print('Player 2 wins')
            break
        
        if board.is_possible_to_move() == False:
            board.print_board()
            print('Draw')
            break
'''
        
#Function to play a game between 2 AlphaZero's
def self_play(model):
    #First, we create an empty board
    board = Game()
    
    #Then, we build the MCTS
    evaluation = model.predict([board.get_state()])
    mcts = MCTS(evaluation[0][0])
    #mcts2 = MCTS(evaluation[0][0])
    
    states = []
    distributions = []
    rewards = []
    
    finished = False
    draw = False
    while not finished and not draw:                
        action, state, mcts_distribution = mcts.makeMove(board, model)
        states.append(state)
        distributions.append(mcts_distribution)
        
        #Update the board, finish if the game has ended
        finished = board.make_move(action)
        
        if not finished and not board.is_possible_to_move():
            draw = True
        
    board.print_board()
    
    states.append(board.get_state())
    distributions.append(distributions[len(distributions) - 1])
         
    #Update the rewards
    if finished:
        #If not draw, each state receives a +-1 reward depending on who has won
        for i in range(0, len(states)):
            if i % 2 == len(states) % 2:
                rewards.append([1.0])
            else:
                rewards.append([-1.0])
    else:
        print('DRAW')
        #Else, every state gets a 0 reward
        for i in range(0, len(states)):
            rewards.append([0.0])
    
    return states, distributions, rewards        
  
#For plotting the loss  
def print_history(history):
    print(history.history.keys())
    # summarize history for loss 
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
  
def main():
    #Create the neural network
    builder = Connect4Zero()
     
    #Building a model from scratch
    model = builder.build()
    builder.save_model(model, 'nn_weights/d3-0.h5')
    
    #Loading a partially trained model
    #model = builder.load_model('nn_weights/d3-217.h5')
    #model.summary()
    
    states_data = []
    distributions_data = []
    rewards_data = []
    
    for i in range(0, 250):
        states_data_tmp = []
        distributions_data_tmp = []
        rewards_data_tmp = []
        for _ in range(0, 10):
            states, distributions, rewards = self_play(model)
            states_data_tmp.extend(states)
            distributions_data_tmp.extend(distributions)
            rewards_data_tmp.extend(rewards)
        
        states_data.extend(states_data_tmp)
        distributions_data.extend(distributions_data_tmp)
        rewards_data.extend(rewards_data_tmp)
        
        indexes = random.sample(range(0, len(states_data)), min(500, len(states_data)))
        states_training = [states_data[i] for i in indexes]
        distributions_training = [distributions_data[i] for i in indexes]
        rewards_training = [rewards_data[i] for i in indexes]
        
        history = model.fit(x=np.array(states_training), y=[np.array(distributions_training), np.array(rewards_training)], batch_size = 32, epochs = 8)
        print_history(history)
        builder.save_model(model, 'nn_weights/d3-' + str(i) + '.h5')
    
if __name__ == "__main__":
    main()
