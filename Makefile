build: 
	pip install -r requirements.py

run_model:
	export BOARD_SIZE=64
	python keras_model.py

train: 
	python app.py