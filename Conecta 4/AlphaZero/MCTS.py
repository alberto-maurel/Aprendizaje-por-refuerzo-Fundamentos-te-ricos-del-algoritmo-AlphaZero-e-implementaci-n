import math
import copy
import numpy
import time

NUM_ACTIONS = 7
NUM_SIMULATIONS = 200
C_PUCT = 1.0
eps = 0.25
PROF = 4
DIR_ALPHA = 1.5

#Class that stores the information needed for each pair (state, action)
class Edge:
    
    #Initialize a new edge. We just need the probability estimated 
    # by the neural network
    def __init__(self, prob):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = prob
        self.nextState = None
    
    #Updating an edge once the expansion has taken place. We just need
    # the predicted value by the neural network
    def updateEdge(self, val):
        self.n = self.n + 1
        self.w = self.w + val
        self.q = self.w / self.n        
        

class MCTS:
    #Each MCTS represents a different state of the game. However, we don't need to store 
    #   the board, just the outgoing edges.    
    
    def __init__(self, probs):
        dirich = numpy.random.dirichlet([DIR_ALPHA for _ in range(0,7)], 1)[0]
        self.edges = []
        for i in range(0, NUM_ACTIONS):
            self.edges.append(Edge((1 - eps) * probs[i] + eps * dirich[i]))
        
    #Simulate one game from the current state to a leaf node        
    def simulation(self, board, model):
        # Node of the MCTS we are currently in
        # We start at the root of the MCTS and we down until we find a leaf node
        current_node = self
        
        # Current state of the game. It matches with state represented by 'current_node' 
        current_board = copy.deepcopy(board)
        
        # Actions taken through this simulation
        # They are used to update the MCTS after the simulation
        actions = []        
        
        # This loop traverses the tree until we find a non existing state
        while(True):
            #First, we decide what action we should take at the current state
            sum_visits = math.sqrt(sum(map(lambda edge: edge.n, current_node.edges)))
            
            utility_vect = []
            for edge in current_node.edges:
                utility_vect.append(C_PUCT * edge.p * sum_visits / (1 + edge.n))
                
            q_values_vect = map(lambda edge: edge.q, current_node.edges)
            
            upper_confidence_vector = []
            for qst, ust in  zip(q_values_vect, utility_vect):
                upper_confidence_vector.append(qst + ust)
                
            #Mask the unavailable actions
            possible_act = current_board.available_moves()
            for i in range(0, NUM_ACTIONS):
                if i not in possible_act:
                    upper_confidence_vector[i] = -numpy.inf
            
            action_taken = numpy.argmax(upper_confidence_vector)
            
            actions.append(action_taken)
            
            win = current_board.make_move(action_taken)
            
            #We haven't reached a leaf node yet
            if(current_node.edges[action_taken].nextState != None):
                # Move to the next state and keep traversing the MCTS
                current_node = current_node.edges[action_taken].nextState
            
            #We have reached a leaf node, so we have to expand it
            else:
                #Win
                if win:
                    if len(actions) % 2 == 0:
                        total_value = -1
                    else:
                        total_value = 1
                #Draw
                elif not current_board.is_possible_to_move():
                    total_value = 0
                
                else:
                    # Evaluate the position
                    evaluation = model.predict([current_board.get_state()])
                    total_value = evaluation[1][0][0]
                    if len(actions) % 2 == 1:
                            total_value = -total_value
                    
                    # If it's neither a direct win nor a draw we append the new node
                    current_node.edges[action_taken].nextState = MCTS(evaluation[0][0])
                    
                break
            
        # Update all the edges we have gone through during the simulation
        current_node = self
        for action in actions:
            current_node.edges[action].updateEdge(total_value)
            current_node = current_node.edges[action].nextState
            total_value = -total_value
            
    # Choose a movement guided by the MCTS
    def makeMove(self, board, model):
        
        #First, we run several simulations to populate our MCTS
        for _ in range(0, NUM_SIMULATIONS):
            self.simulation(board, model)

        #print(list(map(lambda edge: edge.n, self.edges)))

        if board.current_depth < PROF:
            #TAU = (1.0 if board.current_depth < 7 else 0.01)
            TAU = 1
    
            sum_tot = 0
            for edge in self.edges:
                sum_tot = sum_tot + edge.n**(1/TAU)
            
            probs = []
            for edge in self.edges:
                probs.append((edge.n**(1/TAU)) / sum_tot)
            
            action = numpy.random.choice(a=[i for i in range(0, NUM_ACTIONS)], p=probs)
        
        else:
            action = numpy.argmax(list(map(lambda edge: edge.n, self.edges)))
            probs = [0.0 for _ in range(0,7)]
            probs[action] = 1.0

        curr_board = copy.deepcopy(board)
        #We have to take care if the next state is the winning one
        win = curr_board.make_move(action) 
        
        if not win and curr_board.is_possible_to_move():               
            self.__dict__ = self.edges[action].nextState.__dict__
        
        return action, copy.deepcopy(board).get_state(), probs
    
    
    def update_board(self, board, action, model):
        if self.edges[action].nextState == None:
            evaluation = model.predict([board.get_state()])
            self.edges[action].nextState = MCTS(evaluation[0][0])
            
        self.__dict__ = self.edges[action].nextState.__dict__