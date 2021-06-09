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


#Function to play a training game between 2 AlphaZero's
def self_play(model):
    #First, create an empty board
    board = Game()
    
    #Then, build the MCTS
    evaluation = model.predict([board.get_state()])
    mcts = MCTS(evaluation[0][0])
    
    states = []
    distributions = []
    q_values = []
    
    finished = False
    draw = False
    while not finished and not draw:                
        #The MCTS decides what movement should be played and the required data
        # is gathered
        action, state, mcts_distribution, q_val = mcts.makeMove(board, model)
        states.append(state)
        distributions.append(mcts_distribution)
        
        #In this variation, we can directly store the Q value, instead of waiting
        # until the game ends
        q_values.append(q_val) 
        
        #Update the board, finish if the game has ended
        finished = board.make_move(action)
        
        if not finished and not board.is_possible_to_move():
            draw = True
        
    board.print_board()
    
    states.append(board.get_state())
    distributions.append(distributions[len(distributions) - 1])
    q_values.append(-1)
         
    if finished:
        print('PLAYER ', board.current_player, ' WINS')
        
    else:
        print('DRAW')

    
    return states, distributions, q_values        

    
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

    
NUM_ITERATIONS = 1000
NUM_GAMES_PER_ITERATION = 5
   

#Function to train AlphaZero Q
def main():
    #Create the neural network
    builder = Connect4Zero()
     
    #Uncomment to train a fresh model
    model = builder.build()
    #builder.save_model(model, 'nn_weights/d5-0.h5')
    
    #Uncomment to keep training a partially trained model
    #model = builder.load_model('nn_weights/d5-955.h5')
    #model.summary()
    
    states_data = []
    distributions_data = []
    rewards_data = []
    
    for i in range(0, NUM_ITERATIONS):
        states_data_tmp = []
        distributions_data_tmp = []
        rewards_data_tmp = []
        
        #Generate the training games
        #Modify the lower bound of the range if you want to keep training a model
        for _ in range(0, NUM_GAMES_PER_ITERATION):
            states, distributions, rewards = self_play(model)
            states_data_tmp.extend(states)
            distributions_data_tmp.extend(distributions)
            rewards_data_tmp.extend(rewards)
        
        #Store this iteration's data
        states_data.extend(states_data_tmp)
        distributions_data.extend(distributions_data_tmp)
        rewards_data.extend(rewards_data_tmp)
        
        #Train the model and store this iterations weights        
        history = model.fit(x=np.array(states_data), y=[np.array(distributions_data), np.array(rewards_data)], batch_size = 16, epochs = 3)
        print_history(history)
        builder.save_model(model, 'nn_weights/d5-' + str(i) + '.h5')
        
        
        states_data = []
        distributions_data = []
        rewards_data = []
        
    
if __name__ == "__main__":
    main()