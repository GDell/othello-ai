build: 
	pip install -r requirements.py

run_model:
	export BOARD_SIZE=64
	python keras_model.py


othello:
	python app.py


train: 
	python train_model.py