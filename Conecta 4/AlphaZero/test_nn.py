from NeuralNetwork import Connect4Zero
from board import Game
import time
import numpy as np
import matplotlib.pyplot as plt

DRAW_THRESHOLD = 0.0

def test_value(PATH):
    builder = Connect4Zero()
    model = builder.load_model(PATH)
    
    file = open("data/connect-4-first-player.data", "r")
    labels = []
    states = []
    
    labels_data = []
    states_data = []
    
    total_time = 0
    for line in file.readlines():
        line_trim = line.split(',')
        
        ntokens = 0
        board = [[[0 for _ in range(0,3)] for _ in range(0,7)] for _ in range(0,6)]
        
        for col in range(0, 7):
            for fil in range(0, 6):
                char = line_trim[6*col+fil]
                
                if char == 'x':
                    board[fil][col][0] = 1
                    ntokens += 1
                    
                elif char == 'o':
                    board[fil][col][1] = 1
                    ntokens += 1
          
        if ntokens % 2 == 1:
            for col in range(0, 7):
                for fil in range(0, 6):
                    board.decoded_state[fil][col][2] = 1
        
        if ntokens % 2 == 1:
            print('Segundo')
        
        states.append(board)
        
        if line_trim[42] == 'win\n':
            #if ntokens % 2 == 0:
            labels.append(1)
            #else:
            #    labels.append(-1)
        elif line_trim[42] == 'loss\n':
            #if ntokens % 2 == 0:
            labels.append(-1)
            #else:
            #    labels.append(1)
        else:
            labels.append(0)
            
            
        if len(labels) % 1000 == 0:
            labels_data.extend(labels)
            labels = []
            states_data.extend(states)
            states = []
            
        

    labels_data.extend(labels)
    states_data.extend(states)
    
    correct_label = [0, 0, 0]
    wrong_label = [0, 0, 0]
    
    idx = 0
    for state,label in zip(states_data, labels_data):
        value = model.predict([state])[1][0][0]
        
        if value > DRAW_THRESHOLD:
            if label == 1:
                correct_label[0] += 1
            else:
                wrong_label[0] += 1
        elif value < -DRAW_THRESHOLD:
            if label == -1:
                correct_label[2] += 1
            else:
                wrong_label[2] += 1
        else:
            if label == 0:
                correct_label[1] += 1
            else:
                wrong_label[1] += 1
                
        #idx += 1
        #if idx % 1000 == 0:
        #    print(idx)
    
    '''
    if correct_label[0] + wrong_label[0] > 0:
        print('WIN: AC: ', correct_label[0], ' WA: ', wrong_label[0], ' ', correct_label[0]/(correct_label[0] + wrong_label[0]))
    if correct_label[1] + wrong_label[1] > 0:
        print('DRAW: AC: ', correct_label[1], ' WA: ', wrong_label[1], ' ', correct_label[1]/(correct_label[1] + wrong_label[1]))
    if correct_label[2] + wrong_label[2] > 0:
        print('LOSE: AC: ', correct_label[2], ' WA: ', wrong_label[2], ' ', correct_label[2]/(correct_label[2] + wrong_label[2]))
    print('OVERALL: AC: ', correct_label[0]+correct_label[1]+correct_label[2], 
                  ' WA: ', wrong_label[0]+wrong_label[1]+wrong_label[2], ' ', 
                  (correct_label[0]+correct_label[1]+correct_label[2])/
                  (correct_label[0]+correct_label[1]+correct_label[2]+wrong_label[0]+wrong_label[1]+wrong_label[2]))               
    '''
    return correct_label, wrong_label

def line_graph_initial_policy():
    builder = Connect4Zero()
    
    board = Game()
    
    values = [[],[],[],[],[],[],[]]
    RANGE = 220
    axis = [i for i in range(0,RANGE)]
    
    for i in range(0, RANGE):
        model = builder.load_model('nn_weights/d3-' + str(i) + '.h5')
        policy = model.predict([board.get_state()])
        
        for j in range(0,7):
            values[j].append(policy[0][0][j])
        
        if i % 10 == 0:
            print(i)
        
    values = np.array(values).transpose()
    plt.subplots(figsize=(10,8))
    plt.plot(axis, values)
    plt.legend(['0','1','2','3','4','5','6'])
    plt.savefig('policy_initial_board.png')
    plt.show

def line_graph_initial_value():
    builder = Connect4Zero()
    
    board = Game()
    
    values = []
    RANGE = 220
    axis = [i for i in range(0,RANGE)]
    
    for i in range(0, RANGE):
        model = builder.load_model('nn_weights/d3-' + str(i) + '.h5')
        policy = model.predict([board.get_state()])
        values.append(policy[1][0][0])
        
        if i % 10 == 0:
            print(i)
        
    values = np.array(values).transpose()
    plt.subplots(figsize=(10,8))
    plt.plot(axis, values)
    #plt.legend(['inferior izquierda','inferior derecha','superior izquierda','superior derecha'])
    plt.savefig('value_initial_board.png')
    plt.show

def line_graph_prediction_accuracy():
    values = [[],[],[]]
    RANGE = 201
    
    for i in range(0,RANGE,5):
        print(i)
        model_path = 'nn_weights/d3-' + str(i) + '.h5'
        res = test_value(model_path)
        values[0].append(res[0][0]/(res[0][0] + res[1][0]))
        values[1].append(res[0][2]/(res[0][2] + res[1][2]))
        values[2].append((res[0][0] + res[0][2])/(res[0][0] + res[0][2] + res[1][0] + res[1][2]))
        
    axis = [i for i in range(0,RANGE,5)]
    
    #values = np.array(values).transpose()
    plt.subplots(figsize=(10,8))
    plt.plot(axis, values[0], color = 'g')
    plt.plot(axis, values[1], color = 'r')
    plt.plot(axis, values[2], color = 'b')
    plt.legend(['victorias', 'derrotas', 'total'])
    plt.ylim([0, 1])
    plt.savefig('S5 accuracy_evolution_alphazero.png')
    plt.show
    
def test_board1():
    builder = Connect4Zero()
    model = builder.load_model('nn_weights/d3-150.h5')
    board = Game()
    
    board.make_move(1)
    board.make_move(5)
    board.make_move(1)
    board.make_move(2)
    board.make_move(2)
    board.make_move(3)
    board.make_move(2)
    board.make_move(3)
    board.make_move(3)
    board.make_move(0)
    board.make_move(4)
    board.make_move(4)
    board.make_move(4)
    
    board.print_board()
    
    
    print('Valor: ',model.predict([board.get_state()]))
    
def test_board():
    builder = Connect4Zero()
    model = builder.load_model('nn_weights/d3-150.h5')
    board = Game()
    
    board.make_move(0)
    board.make_move(1)
    board.make_move(0)
    board.make_move(1)
    board.make_move(2)
    board.make_move(0)
    board.make_move(5)
    board.make_move(3)
    board.make_move(2)
    board.make_move(5)
    board.make_move(3)
    board.make_move(2)
    board.make_move(4)
    board.make_move(4)
    board.make_move(5)
    board.make_move(3)
    board.make_move(6)
    board.make_move(2)
    #board.make_move(6)
    
    board.print_board()
    
    
    print('Valor: ',model.predict([board.get_state()]))
    
    
def main():
    line_graph_prediction_accuracy()

if __name__ == "__main__":
    main()