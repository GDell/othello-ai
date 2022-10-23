'''
    Siyan
    CS5001
    Fall 2018
    November 28, 2018
'''

import copy
from shutil import move
from telnetlib import GA
from othello_game_master import score
from othello_game_master.board import Board
from othello_game_master.modes import GameModes
from utils import load_model
from run_model import predict_move
import random
import turtle
import time
import json
import operator


# Define all the possible directions in which a player's move can flip 
# their adversary's tiles as constant (0 – the current row/column, 
# +1 – the next row/column, -1 – the previous row/column)
MOVE_DIRS = [(-1, -1), (-1, 0), (-1, +1),
             (0, -1),           (0, +1),
             (+1, -1), (+1, 0), (+1, +1)]

class Othello(Board):
    ''' Othello class.
        Attributes: current_player, an integer 0 or 1 to represent two 
                    different players (the user and the computer)
                    num_tiles, a list of integers for number of tiles each 
                    player has
                    n, an integer for nxn board
                    all other attributes inherited from class Board
        n (integer) is optional in the __init__ function
        current_player, num_tiles and all other inherited attributes 
        are not taken in the __init__

        Methods: initialize_board, make_move, flip_tiles, has_tile_to_flip, 
                 has_legal_move, get_legal_moves, is_legal_move, 
                 is_valid_coord, run, play, make_random_move, 
                 report_result, __str__ , __eq__ and all other methods 
                 inherited from class Board
    '''

    def __init__(self, \
                model_gen: int, \
                n: int = 8, \
                game_mode: GameModes = GameModes.RANDOM_VS_RANDOM, \
                model = load_model(), \
                train_session: bool = False, 
                draw: bool = True):
        '''
            Initilizes the attributes. 
            Only takes one optional parameter; others have default values.
        '''
        Board.__init__(self, n)
        self.current_player = 0
        self.num_tiles = [2, 2]
        self.current_move_index = 0
        self.game_mode = game_mode
        self.epoch = 27
        self.model = model
        self.n = n
        self.model_gen = model_gen
        self.train_session = train_session
        self.draw = draw
        self.record = {
            'snapshots':[],
            'selected_moves':[],
            'move_choices':[]
        }
        

    def initialize_board(self):
        ''' Method: initialize_board
            Parameters: self
            Returns: nothing
            Does: Draws the first 4 tiles in the middle of the board
                  (the size of the board must be at least 2x2).
        '''
        if self.n < 2:
            return

        coord1 = int(self.n / 2 - 1)
        coord2 = int(self.n / 2)
        initial_squares = [(coord1, coord2), (coord1, coord1),
                           (coord2, coord1), (coord2, coord2)]
        
        for i in range(len(initial_squares)):
            color = i % 2
            row = initial_squares[i][0]
            col = initial_squares[i][1]
            self.board[row][col] = color + 1
            self.draw_tile(initial_squares[i], color)
            
    
    def make_move(self):
        ''' Method: make_move
            Parameters: self
            Returns: nothing
            Does: Draws a tile for the player's next legal move on the 
                  board and flips the adversary's tiles. Also, updates the 
                  state of the board (1 for black tiles and 2 for white 
                  tiles), and increases the number of tiles of the current 
                  player by 1.
        '''
        if self.is_legal_move(self.move):
            self.board[self.move[0]][self.move[1]] = self.current_player + 1
            self.num_tiles[self.current_player] += 1
            self.draw_tile(self.move, self.current_player)
            self.flip_tiles()
    

    def flip_tiles(self):
        ''' Method: flip_tiles
            Parameters: self
            Returns: nothing
            Does: Flips the adversary's tiles for current move. Also, 
                  updates the state of the board (1 for black tiles and 
                  2 for white tiles), increases the number of tiles of 
                  the current player by 1, and decreases the number of 
                  tiles of the adversary by 1.
        '''
        curr_tile = self.current_player + 1 
        for direction in MOVE_DIRS:
            if self.has_tile_to_flip(self.move, direction):
                i = 1
                while True:
                    row = self.move[0] + direction[0] * i
                    col = self.move[1] + direction[1] * i
                    if self.board[row][col] == curr_tile:
                        break
                    else:
                        self.board[row][col] = curr_tile
                        self.num_tiles[self.current_player] += 1
                        self.num_tiles[(self.current_player + 1) % 2] -= 1
                        self.draw_tile((row, col), self.current_player)
                        i += 1


    def has_tile_to_flip(self, move, direction):
        ''' Method: has_tile_to_flip
            Parameters: self, move (tuple), direction (tuple)
            Returns: boolean 
                     (True if there is any tile to flip, False otherwise)
            Does: Checks whether the player has any adversary's tile to flip
                  with the move they make.

                  About input: move is the (row, col) coordinate of where the 
                  player makes a move; direction is the direction in which the 
                  adversary's tile is to be flipped (direction is any tuple 
                  defined in MOVE_DIRS).
        '''
        i = 1
        if self.current_player in (0, 1) and \
           self.is_valid_coord(move[0], move[1]):
            curr_tile = self.current_player + 1
            while True:
                row = move[0] + direction[0] * i
                col = move[1] + direction[1] * i
                if not self.is_valid_coord(row, col) or \
                    self.board[row][col] == 0:
                    return False
                elif self.board[row][col] == curr_tile:
                    break
                else:
                    i += 1
        return i > 1


    def has_legal_move(self):
        ''' Method: has_legal_move
            Parameters: self
            Returns: boolean 
                     (True if the player has legal move, False otherwise)
            Does: Checks whether the current player has any legal move 
                  to make.
        '''
        for row in range(self.n):
            for col in range(self.n):
                move = (row, col)
                if self.is_legal_move(move):
                    return True
        return False
    

    def get_legal_moves(self):
        ''' Method: get_legal_moves
            Parameters: self
            Returns: a list of legal moves that can be made
            Does: Finds all the legal moves the current player can make.
                  Every move is a tuple of coordinates (row, col).
        '''
        moves = []
        for row in range(self.n):
            for col in range(self.n):
                move = (row, col)
                if self.is_legal_move(move):
                    moves.append(move)
        return moves


    def is_legal_move(self, move):
        ''' Method: is_legal_move
            Parameters: self, move (tuple)
            Returns: boolean (True if move is legal, False otherwise)
            Does: Checks whether the player's move is legal.

                  About input: move is a tuple of coordinates (row, col).
        '''
        if move != () and self.is_valid_coord(move[0], move[1]) \
           and self.board[move[0]][move[1]] == 0:
            for direction in MOVE_DIRS:
                if self.has_tile_to_flip(move, direction):
                    return True
        return False


    def is_valid_coord(self, row, col):
        ''' Method: is_valid_coord
            Parameters: self, row (integer), col (integer)
            Returns: boolean (True if row and col is valid, False otherwise)
            Does: Checks whether the given coordinate (row, col) is valid.
                  A valid coordinate must be in the range of the board.
        '''
        if 0 <= row < self.n and 0 <= col < self.n:
            return True
        return False


    def run(self):
        
        ''' Method: run
            Parameters: self
            Returns: nothing
            Does: Starts the game, sets the user to be the first player,
                  and then alternate back and forth between the user and 
                  the computer until the game is over.
        '''
        if self.current_player not in (0, 1):
            print('Error: unknown player. Quit...')
            return
        
        self.current_player = 0
        if self.game_mode in [GameModes.PLAYER_VS_MODEL, GameModes.PLAYER_VS_RANDOM]:
            turtle.onscreenclick(self.play)
        else:
            print('AI\'s turn.')
            self.play()
        turtle.mainloop()


    def get_model_move(self, moves, board, variation_flag = False):
        prediction, predicted_moves = predict_move(self.model, board)

        predicted_moves_value_dict = {}
        for move in predicted_moves:
            predicted_moves_value_dict[move['move']] = move['value']

        possible_moves = []
        for move in moves:
            possible_moves.append({
                'move': move, 'value': predicted_moves_value_dict[move]
            })

        possible_moves.sort(key=operator.itemgetter('value'), reverse=True)

        num_moves = len(possible_moves)
        move_index = 0
        if variation_flag: 
            rank_choice_range = range(3) if num_moves >= 3 else range(num_moves)
            move_index = random.choice(rank_choice_range)
        
        return possible_moves[move_index]['move']


    def write_trial_file(self, file_name, data, move_index):
        f = open(f"./data/model_gen_{self.model_gen}/epoch_{self.epoch}/{file_name}_{move_index}.txt", "w")
        f.write(data)
        f.close()


    def write_training_data(self):
        for round in range(0, len(self.record['snapshots'])):
            self.write_trial_file("board", self.record['snapshots'][round], round)
            self.write_trial_file("selected_move", self.convert_move_to_matrix(self.record['selected_moves'][round]), round)
            self.write_trial_file("move_choices", str(self.record['move_choices'][round]), round)


    def reset_board(self):
        self.current_player = 0
        self.num_tiles = [2, 2]
        self.current_move_index = 0
        self.record = {
            'snapshots':[],
            'selected_moves':[],
            'move_choices':[]
        }
        self.board = [[0] * self.n for i in range(self.n)]
        self.move = ()
        self.draw_board()
        self.initialize_board()
        return


    def convert_move_to_matrix(self, move):
        rows, cols = (8,8)
        m = [[0 for i in range(cols)] for j in range(rows)]
        m[move[0]][move[1]] = 1 
        board_str = ''
        for row in m:
            board_str += str(row) + '\n'
        return board_str 


    def play(self, x = None, y = None):
        ''' Method: play
            Parameters: self, x (float), y (float)
            Returns: nothing
            Does: Plays alternately between the user's turn and the computer's
                  turn. The user plays the first turn. For the user's turn, 
                  gets the user's move by their click on the screen, and makes 
                  the move if it is legal; otherwise, waits indefinitely for a 
                  legal move to make. For the computer's turn, just makes a 
                  random legal move. If one of the two players (user/computer)
                  does not have a legal move, switches to another player's 
                  turn. When both of them have no more legal moves or the 
                  board is full, reports the result, saves the user's score 
                  and ends the game.

                  About the input: (x, y) are the coordinates of where 
                  the user clicks.
        '''

        # Take a snapshot of the game
        snapshot = self.__str__()
        moves = self.get_legal_moves()
        chosen_move = ""

        # Player 1's turn
        if moves:
            if self.game_mode == GameModes.RANDOM_VS_RANDOM:
                self.move = random.choice(moves)
                chosen_move = self.move

            elif self.game_mode in [GameModes.MODEL_VS_RANDOM, GameModes.MODEL_VS_MODEL]:
                self.move = self.get_model_move(moves, self.board, variation_flag=True)
                chosen_move = self.move
                self.make_move()

            elif self.game_mode in [GameModes.PLAYER_VS_MODEL]:
                self.get_coord(x, y)
                if self.is_legal_move(self.move):
                    turtle.onscreenclick(None)
                    self.make_move()
                else:
                    return

            self.record['snapshots'].append(snapshot)
            self.record['selected_moves'].append(chosen_move)
            self.record['move_choices'].append(str(moves))

            self.current_move_index += 1


        # Player 2's turn
        while True:
            self.current_player = 1
            if self.has_legal_move():
                if self.game_mode in [GameModes.RANDOM_VS_RANDOM, GameModes.MODEL_VS_RANDOM]:
                    self.make_random_move()

                elif self.game_mode in [GameModes.MODEL_VS_MODEL, GameModes.PLAYER_VS_MODEL]:
                    # Reverse state of board to feed to model for computers turn.
                    player_two_board = copy.deepcopy(self.board)
                    for row in range(0, len(player_two_board)): 
                        for column in range(0, len(player_two_board[row])): 
                            val = player_two_board[row][column]
                            if val == 1: 
                                val = 2
                            elif val == 2: 
                                val = 1
                            player_two_board[row][column] = val

                    legal_moves = self.get_legal_moves()
                    self.move = self.get_model_move(legal_moves, player_two_board, variation_flag=False)
                    self.make_move()


                self.current_player = 0
                if self.has_legal_move():  
                    break
            else:
                break    
        
        
    
        # Switch back to the user's turn
        self.current_player = 0

        # Check whether the game is over
        if not self.has_legal_move() or sum(self.num_tiles) == self.n ** 2:

            turtle.onscreenclick(None)
            
            print('-----------')
            is_win = self.report_result()

            if self.train_session:
                if not is_win:
                    self.reset_board()
                    self.run()
                    return
                
                self.epoch += 1

                if self.epoch < 100:
                    self.reset_board()
                    self.run()
                    return

            # name = input('Enter your name for posterity\n')
            # if not score.update_scores(name, self.num_tiles[0]):
            #     print('Your score has not been saved.')
            print('Thanks for playing Othello!')
            close = input('Close the game screen? Y/N\n')
            if close == 'Y':
                turtle.bye()
            elif close != 'N':
                print('Quit in 3s...')
                turtle.ontimer(turtle.bye, 3000)
        else:
            if self.game_mode in [GameModes.PLAYER_VS_MODEL, GameModes.PLAYER_VS_RANDOM]:
                turtle.onscreenclick(self.play)
            else:
                self.play()
            

    def make_random_move(self):
        ''' Method: make_random_move
            Parameters: self
            Returns: nothing
            Does: Makes a random legal move on the board.
        '''
        moves = self.get_legal_moves()
        if moves:
            self.move = random.choice(moves)
            self.make_move()


    def report_result(self):
        ''' Method: report_result
            Parameters: self
            Returns: nothing
            Does: Announces the winner and reports the final number of
                  tiles each play has.
        '''
        win = False
        print('GAME OVER!!')

        if self.num_tiles[0] > self.num_tiles[1]:
            print('YOU WIN!!',
                  'You have %d tiles, but the computer only has %d!' 
                  % (self.num_tiles[0], self.num_tiles[1]))
            win = True
            if self.train_session:
                self.write_training_data()
                self.write_trial_file("game_result", str(win), self.current_move_index)
        elif self.num_tiles[0] < self.num_tiles[1]:
            print('YOU LOSE...',
                  'The computer has %d tiles, but you only have %d :(' 
                  % (self.num_tiles[1], self.num_tiles[0]))

        else:
            print("IT'S A TIE!! There are %d of each!" % self.num_tiles[0])

        return win 


    def __str__(self):
        ''' 
            Returns a printable version of the current status of the 
            game to print.
        '''
        player_str = 'Current player: ' + str(self.current_player + 1) + '\n'
        num_tiles_str = '# of black tiles -- 1: ' + str(self.num_tiles[0]) + \
                        '\n' + '# of white tiles -- 2: ' + \
                        str(self.num_tiles[1]) + '\n'
        board_str = Board.__str__(self)
        printable_str = player_str + num_tiles_str + board_str

        return printable_str


    def __eq__(self, other):
        '''
            Compares two instances. 
            Returns True if they have both the same board attribute and 
            current player, False otherwise.
        '''
        return Board.__eq__(self, other) and self.current_player == \
        other.current_player

