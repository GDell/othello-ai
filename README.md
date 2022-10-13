## Requirements

conda -- <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>

## Start:

'conda activate othello-ai'
'make run'

Install packages 

'conda install <x>'


## Input: 

1. 8x8 matrix - Current board state (64 total vectors)
2. Possible moves? 
3. Other features? ....

Values - 0, 1, 2 


## Output:

Array of 64 float values, converted to 8x8 matrix.

A selection of all possible choices for the next move, the liklihood they lead to victory.


## Labels: 

The move given the board input that led to a victory at the end of the game. (Trained only on games won)


## Training: 

8x8 Input nodes

64 Output nodes

The model would produce training data sets by playing against itself. 

Initially, the model will have entirely randomized weights. 

Feed it a board, the system picks its best move of the possible moves. Repeat and see who wins. Store all data for training sets.


The process of training against itself will go like this: 

1. Start with random weights. Feed the board and available moves to the network to play many games. 
2. Take this data and re-train the model. 


Do we just avoid data from games where we lost? Only train on decision made in games where we won? Then train against itself. 

OR do we incorporate the win/loss into the loss function? 


## Dependencies 

Credit to https://github.com/SiyanH for the Othello Game engine used in this project.
    - othello-game <https://github.com/SiyanH/othello-game>


## Resources

- <https://www.tensorflow.org/tutorials/images/cnn>
- <https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid>

