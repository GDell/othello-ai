## Requirements

conda -- <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>

## Start:

'conda activate <env>'
'make run'

## Model Input: 

1. 8x8 matrix - Current board state (64 total vectors)
2. Possible moves? 
3. Other features? ....

Values - 0, 1, 2 


## Model Output:

Array of 64 float values, converted to 8x8 matrix.

A selection of all possible choices for the next move, the liklihood they lead to victory.


## Training: 

8x8 Input nodes

64 Output nodes

The model produces training data by playing against itself. 

Initially, the model is initialized with randomized weights. 

Each move, the board state is fed to the model and it recommends the next move. 

Move choices and board states are saved for training.


Self Training: 

1. Start with random weights. Feed the board and available moves to the network to play many games. 
2. Take this data and re-train the model. 


## Dependencies 

Credit to https://github.com/SiyanH for the Othello Game engine used in this project.
    - othello-game <https://github.com/SiyanH/othello-game>

## Resources

- <https://www.tensorflow.org/tutorials/images/cnn>
- <https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid>

