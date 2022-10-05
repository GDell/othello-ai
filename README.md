## Start:

'conda activate othello-ai'
'make run'

Install packages 

'conda install <x>'


## Input: 

8x8 matrix - Current board state

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

Feed it a board, the system picks its best move. Repeat and see who wins. Store all data for training sets.


## Dependencies 

othello-game <https://github.com/SiyanH/othello-game>
