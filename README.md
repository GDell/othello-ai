## Start:

'conda activate othello-ai'
'make run'

Install packages 

'conda install <x>'


## Input: 

1. 8x8 matrix - Current board state (64 total vectors)
2. Possible moves

Values - 0, 1, 2 


## Output:

8x8 matrix 

A selection of all possible choices for the next move, the liklihood they lead to victory


## Labels: 

0 or 1 
Loss at the end of the game, retained at the end of the game?
Win at the end of the game, retained at the end of the game?


## Training: 

8x8 Input nodes

8x8 Output nodes

The model would train against itself. 

Initially, the model will have entirely randomized weights. 

Feed it a board, the system picks its best move of the possible moves. Repeat and see who wins. Store all data for training sets.


The process of training against itself will go like this: 

1. Start with random weights. Feed the board and available moves to the network to play many games. 
2. Take this data and re-train the model. 


## Dependencies 

This model the following Othello game engine: 
othello-game <https://github.com/SiyanH/othello-game>
