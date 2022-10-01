build: 
	pdm update
	pdm build

run:
	eval $(pdm venv activate othello-ai)
	pdm run start
	