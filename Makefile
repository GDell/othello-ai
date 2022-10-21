make save_env: 
	conda env export | grep -v "^prefix: " > environment.yml

game:
	python game.py


model: 
	python model.py


train: 
	python train_model.py


run_model: 
	python run_model.py

