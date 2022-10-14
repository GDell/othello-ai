from enum import Enum


class GameModes(Enum):
    PLAYER_VS_RANDOM = "player_vs_random"
    RANDOM_VS_RANDOM = "random_vs_random"
    MODEL_VS_MODEL = "model_vs_model"
    PLAYER_VS_MODEL = "player_vs_model"
    MODEL_VS_RANDOM = "model_vs_random"


if __name__ == "__main__":
    modes = GameModes
    print([mode for mode, member in modes.__members__.items()]) # ['PLAYER_VS_RANDOM', 'RANDOM_VS_RANDOM', 'MODEL_VS_MODEL', 'PLAYER_VS_MODEL']
    print(modes.PLAYER_VS_MODEL.name) # --> PLAYER_VS_MODEL
    