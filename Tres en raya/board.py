fil = [0,1,1,1,0,-1,-1,-1]
col = [-1,-1,0,1,1,1,0,-1]

class Game:
    
    def __init__(self):
        self.current_player = 0
        self.decoded_state = [[[0 for _ in range(0,2)] for _ in range(0,3)] for _ in range(0,3)]
        self.current_depth = 0
    
    #Prints a human-friendly representation of the board
    def print_board(self):
        print('Last move from', 'O' if self.current_player == 0 else 'X')
        for i in range(2, -1, -1):
            for j in range(0,3):
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
        for i in range(0,3):
            for j in range(0,3):
                if self.decoded_state[i][j][0] == 0 and self.decoded_state[i][j][1] == 0:
                    moves.append(3*i+j)
                
        return moves
    
    #Checks whether it's possible to move or not
    def is_possible_to_move(self):
        return (False if len(self.available_moves()) == 0 else True)
    
    #Executes the movement and returns True if the player has won the game
    def make_move(self, action):
        y = int(action / 3)     
        x = int(action % 3)
        
        #First, we check whether the movement is possible or not
        if self.decoded_state[y][x][0] == 1 or self.decoded_state[y][x][1] == 1:
            self.print_board()
            raise Exception('Action ' , action, ' is not possible')
            
        #If the column is not full, we place the token in that column
        #Update the decoded state
        self.decoded_state[y][x][self.current_player] = 1
        
        adj = [0,0,0,0]
        
        #Check if the movement has lead to a win        
        for k in range(0, len(fil)):
            for l in range(1, 3):
                xsig = x + l * col[k]
                ysig = y + l * fil[k]
                if xsig >= 0 and xsig < 3 and ysig >= 0 and ysig < 3:
                    if self.decoded_state[ysig][xsig][self.current_player] == 1:
                        adj[k % 4] = adj[k % 4] + 1
                    else:
                        break
                else:
                    break
        
        #Change the player
        self.current_player = 1 - self.current_player
        
        self.current_depth = self.current_depth + 1
        
        return True if max(adj) >= 2 else False
    
    #Returns the current state with the representation the nn is expecting
    def get_state(self):
        return [[[(self.decoded_state[i][j][k] if k < 2 else (self.current_player)) for k in range(0,3)] for j in range(0,3)] for i in range(0,3)]
        