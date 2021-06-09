fil = [0,1,1,1,0,-1,-1,-1]
col = [-1,-1,0,1,1,1,0,-1]

#Class to represent the Connect 4 board
class Game:
    
    #Create an empty board
    def __init__(self):
        self.current_player = 0
        self.decoded_state = [[[0 for _ in range(0,2)] for _ in range(0,7)] for _ in range(0,6)]
        self.next_token = [0 for _ in range(0,7)]
        self.current_depth = 0
    
    #Prints a human-friendly representation of the board
    def print_board(self):
        print('Last move from', 'O' if self.current_player == 0 else 'X')
        for i in range(5, -1, -1):
            for j in range(0,7):
                print('|', end='')
                
                if self.decoded_state[i][j][0] == 1:
                    print('X', end='')
                elif self.decoded_state[i][j][1] == 1:
                    print('O', end='')
                else:
                    print(' ', end='')
            print('|')

    #Function that returns what movements are possible from the current state
    def available_moves(self):
        moves = []
        for j in range(0,7):
            if self.next_token[j] < 6:
                moves.append(j)
        return moves
    
    #Checks whether it's possible to move or not
    def is_possible_to_move(self):
        return (False if self.current_depth == 42 else True)
    
    #Executes the movement and returns True if the player has won the game 
    # (returns False otherwise)
    def make_move(self, column):
        #First, we check whether the movement is possible or not
        if self.next_token[column] == 6:
            self.print_board()
            raise Exception('Column ' , column, ' is already full')
            
        #If the column is not full, we place the token in that column
        #Update the decoded state
        self.decoded_state[self.next_token[column]][column][self.current_player] = 1
        
        adj = [0,0,0,0]
        
        #Check if the movement has led to a win
        x = column
        y = self.next_token[column]
        
        for k in range(0, len(fil)):
            for l in range(1, 7):
                xsig = x + l * col[k]
                ysig = y + l * fil[k]
                if xsig >= 0 and xsig < 7 and ysig >= 0 and ysig < 6:
                    if self.decoded_state[ysig][xsig][self.current_player] == 1:
                        adj[k % 4] = adj[k % 4] + 1
                    else:
                        break
                else:
                    break
                                
        #Update the next token's position
        self.next_token[column] = self.next_token[column] + 1 
        
        #Change the player
        self.current_player = 1 - self.current_player
        
        self.current_depth = self.current_depth + 1
        
        return True if max(adj) >= 3 else False
    
    #Returns the current state with the representation the neural network is expecting
    def get_state(self):
        return [[[(self.decoded_state[i][j][k] if k < 2 else (self.current_player)) for k in range(0,3)] for j in range(0,7)] for i in range(0,6)]
        