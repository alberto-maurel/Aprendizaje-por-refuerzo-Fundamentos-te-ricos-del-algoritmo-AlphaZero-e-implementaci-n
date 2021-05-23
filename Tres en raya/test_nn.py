from NeuralNetwork import Connect4Zero
from board import Game
import time
import numpy as np
import matplotlib.pyplot as plt

CURRENT_MODEL = 'nn_weights_c3/t4-125.h5'

# Bunch of functions to plot several metrics. These are the functions the 
# memory's graphs are plotted with.

def plot_policy(board, model):
    val = model.predict([board.get_state()])[0][0]
    print(val)
    
    mat = [[val[6], val[7], val[8]],
           [val[3], val[4], val[5]],
           [val[0], val[1], val[2]]]

    plt.imshow(np.array(mat), cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.savefig(CURRENT_MODEL)
    plt.show()

def line_graph_evolution_2_mov_lateral():
    builder = Connect4Zero()
    
    board1 = Game()
    board1.make_move(1)
    
    board2 = Game()
    board2.make_move(3)
    
    board3 = Game()
    board3.make_move(5)
    
    board4 = Game()
    board4.make_move(7)
    
    boards = [board1, board2, board3, board4]
    
    values = [[],[],[],[]]
    RANGE = 126
    axis = [i for i in range(0,RANGE)]
    
    for i in range(0, RANGE):
        model = builder.load_model('nn_weights_c3/t4-' + str(i) + '.h5')
        for j in range(0, 4):
            policy = model.predict([boards[j].get_state()])[0][0]
            if j == 0:
                values[j].append(policy[0] + policy[2] + policy[4] + policy[7])
            elif j == 1:
                values[j].append(policy[0] + policy[4] + policy[5] + policy[6])
            elif j == 2:
                values[j].append(policy[2] + policy[3] + policy[4] + policy[8])
            elif j == 3:
                values[j].append(policy[1] + policy[4] + policy[6] + policy[8])
                
        if i % 10 == 0:
            print(i)
        
    values = np.array(values).transpose()
    plt.subplots(figsize=(10,8))
    plt.plot(axis, values)
    plt.legend(['inferior','izquierda','derecha','superior'])
    plt.savefig('2-mov-lateral_2.png')
    plt.show

def line_graph_evolution_2_mov_esquina():
    builder = Connect4Zero()
    
    board1 = Game()
    board1.make_move(0)
    
    board2 = Game()
    board2.make_move(2)
    
    board3 = Game()
    board3.make_move(6)
    
    board4 = Game()
    board4.make_move(8)
    
    boards = [board1, board2, board3, board4]
    
    values = [[],[],[],[]]
    RANGE = 126
    axis = [i for i in range(0,RANGE)]
    
    for i in range(0, RANGE):
        model = builder.load_model('nn_weights_c3/t4-' + str(i) + '.h5')
        for j in range(0, 4):
            policy = model.predict([boards[j].get_state()])[0][0]
            values[j].append(policy[4])
        if i % 10 == 0:
            print(i)
        
    values = np.array(values).transpose()
    plt.subplots(figsize=(10,8))
    plt.plot(axis, values)
    plt.legend(['inferior izquierda','inferior derecha','superior izquierda','superior derecha'])
    plt.savefig('2-mov-esquina.png')
    plt.show
    
def line_graph_evolution_2_mov_centro():
    builder = Connect4Zero()
    
    board = Game()
    board.make_move(4)

    values=[]
    RANGE = 126
    axis = [i for i in range(0,RANGE)]
    
    for i in range(0, RANGE):
        model = builder.load_model('nn_weights_c3/t4-' + str(i) + '.h5')
        for j in range(0, 1):
            policy = model.predict([board.get_state()])[0][0]
            values.append(policy[0] + policy[2] + policy[6] + policy[8])
        if i % 10 == 0:
            print(i)
        
    values = np.array(values).transpose()
    plt.subplots(figsize=(10,8))
    plt.plot(axis, values)
    #plt.legend([0,2,6,8])
    plt.savefig('2-mov-centro-2.png')
    plt.show
        
def second_position_evaluation():
    builder = Connect4Zero()
    
    
    model = builder.load_model(CURRENT_MODEL)
    
    board = Game()
    board.make_move(8)
    board.print_board()
    
    
    print('Valor: ',model.predict([board.get_state()])[1][0][0])
    plt.close()
    plot_policy(board, model)
    
def position_evaluation():
    builder = Connect4Zero()
    
    model = builder.load_model(CURRENT_MODEL)
    
    board = Game()
    
    
    print('Valor: ',model.predict([board.get_state()])[1][0][0])
    plot_policy(board, model)

def main():    
    position_evaluation()
    
if __name__ == "__main__":
    main()