'''
    Siyan
    CS5001
    Fall 2018
    November 30, 2018
'''

from othello_game_master import othello
from othello_game_master.modes import GameModes

def run_game():
    # Initializes the game
    # game = othello.Othello(game_mode = GameModes.MODEL_VS_MODEL, model_gen = 1, train_session = True)
    game = othello.Othello(game_mode = GameModes.MODEL_VS_MODEL, model_gen = 1, train_session = True)

    game.draw_board()
    game.initialize_board()
    print(game.__str__())

    # Starts playing the game
    # The user makes a move by clicking one of the squares on the board
    # The computer makes a random legal move every time
    # Game is over when there are no more lagal moves or the board is full
    game.run()



if __name__ == "__main__":
    run_game()
    