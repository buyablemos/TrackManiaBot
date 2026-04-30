import os

class Config:
    DATETIME_FORMAT: str = '%m-%d_%H-%M'

    class Paths:
        MAP_PREFIX: str = "maps"
        MAP: str =   "zigzags"        # Verify that the map here is the same as the one in your .env file
        MAP_BLOCKS_PATH: str = os.path.join(MAP_PREFIX, MAP, "ordered_blocks.json")
        MAP_LAYOUT_PATH: str = os.path.join(MAP_PREFIX, MAP, "layout.txt")

        MODELS_PATH: str = "models"
        LATEST_MODEL_PATH: str = os.path.join(MODELS_PATH, "latest")
        STAT_FILE_NAME: str = "stats.json"
        DQN_MODEL_FILE_NAME: str = "model.pth"
        DQN_REPLAY_FILE_NAME: str = "replay.pt"
        ACTOR_FILE_NAME: str = "actor.pth"
        CRITIC_FILE_NAME: str = "critic.pth"

        @staticmethod
        def get_map():
            return {"ordered_blocks": Config.Paths.MAP_BLOCKS_PATH,
                    "list_of_blocks": Config.Paths.MAP_LAYOUT_PATH}

    class Game:
        NUMBER_OF_CLIENTS: int = 1
        TMI_WINDOW_NAME: str = "TrackMania Nations Forever (TMInterface 1.4.3)"
        WINDOW_NAME: str = "TrackMania Nations Forever"
        PROCESS_NAME: str = "TmForever.exe"

        BLOCK_SIZE: int = 32

        NUMBER_OF_ACTIONS_PER_SECOND: int = 10
        INTERVAL_BETWEEN_ACTIONS: int = (1000 // NUMBER_OF_ACTIONS_PER_SECOND) // 10 * 10
        GAME_SPEED: int = 16
        RESTART_INTERVAL_SECONDS: int = 60 * 60 * 4

        CURRICULUM_LEARNING: bool = False
        UNLOCK_ALL_STATES: bool = False

        REWARD_PER_MS: float = -6 / 5000
        REWARD_PER_METER_ALONG_CENTERLINE: float = 1 / 20

        STATES_INTERVAL: int = 5000

        # Final bonus
        MAX_BONUS: float = 10
        PER_SEC_RATIO: float = 1  # one-second difference gives 30% more reward
        TIME_REF: int = 240 * 1000

    class PPO:
        LEARNING_RATE: float = 0.0003
        GAMMA: float = 0.99
        LAMBDA: float = 0.95
        EPSILON: float = 0.2
        C1: float = 1.0
        C2: float = 0.01
        MEMORY_SIZE: int = 128
        BATCH_SIZE: int = 32
        EPOCHS: int = 4     # Number of times to train on a given memory batch

        @staticmethod
        def get_hyperparameters():
            return {
                "learning_rate": Config.PPO.LEARNING_RATE,
                "gamma": Config.PPO.GAMMA,
                "lambda": Config.PPO.LAMBDA,
                "epsilon": Config.PPO.EPSILON,
                "c1": Config.PPO.C1,
                "c2": Config.PPO.C2,
                "memory_size": Config.PPO.MEMORY_SIZE,
                "batch_size": Config.PPO.BATCH_SIZE,
                "epochs": Config.PPO.EPOCHS,
            }

    class DQN:
        LEARNING_RATE: float = 1e-3     # Only used if no learning rate schedule is defined
        LEARNING_RATE_SCHEDULE: list[tuple[int, float]] = [(0, 1e-3),
                                                           (3_000_000, 5e-5),
                                                           (12_000_000, 5e-5),
                                                           (15_000_000, 1e-5)]
        # GAMMA: float = 0.99
        GAMMA_SCHEDULE: list[tuple[int, float]] = [(0, 0.99),
                                                   (1_500_000, 0.999),
                                                   (2_500_000, 1)]

        NUMBER_OF_QUANTILES: int = 8
        N_COS: int = 64 # Number of cosine embedding dimensions
        KAPPA: float = 1.0

        MAX_MEMORY: int = 100_000
        MIN_MEMORY: int = 10_000
        BATCH_SIZE: int = 512

        EPSILON_SCHEDULE: list[tuple[int, float]] = [(0, 1),
                                                     (50_000, 1),
                                                     (500_000, 0.05),
                                                     (3_000_000, 0.03)]

        EPSILON_BOLTZMANN_SCHEDULE: list[tuple[int, float]] = [(0, 0.15),
                                                               (3_000_000, 0.03)]

        TAU_EPSILON_BOLTZMANN: float = 0.01

        UPDATE_TARGET_EVERY: int = 2
        TAU: float = 0.02

        ALPHA: float = 0.6
        BETA_START: float = 0.4
        BETA_MAX: float = 1.0
        BETA_INCREMENT_STEPS: int = 40000

        N_STEPS: int = 3

        ENABLE_NOISY_NETWORK: bool = False
        ENABLE_DUELING_NETWORK: bool = True
        NOISY_NETWORK_SIGMA_START: float = 0.5

        @staticmethod
        def get_hyperparameters():
            return {
                "learning_rate": Config.DQN.LEARNING_RATE,
                "learning_rate_schedule": Config.DQN.LEARNING_RATE_SCHEDULE,
                #"gamma": Config.DQN.GAMMA,
                "gamma_schedule": Config.DQN.GAMMA_SCHEDULE,
                "number_of_quantiles": Config.DQN.NUMBER_OF_QUANTILES,
                "n_cos": Config.DQN.N_COS,
                "kappa": Config.DQN.KAPPA,
                "max_memory": Config.DQN.MAX_MEMORY,
                "min_memory": Config.DQN.MIN_MEMORY,
                "batch_size": Config.DQN.BATCH_SIZE,
                "epsilon_schedule": Config.DQN.EPSILON_SCHEDULE,
                "epsilon_boltzmann_schedule": Config.DQN.EPSILON_BOLTZMANN_SCHEDULE,
                "tau_epsilon_boltzmann": Config.DQN.TAU_EPSILON_BOLTZMANN,
                "update_target_every": Config.DQN.UPDATE_TARGET_EVERY,
                "tau": Config.DQN.TAU,
                "alpha": Config.DQN.ALPHA,
                "beta_start": Config.DQN.BETA_START,
                "beta_max": Config.DQN.BETA_MAX,
                "beta_increment_steps": Config.DQN.BETA_INCREMENT_STEPS,
                "n_steps": Config.DQN.N_STEPS,
                "enable_noisy_network": Config.DQN.ENABLE_NOISY_NETWORK,
                "enable_dueling_network": Config.DQN.ENABLE_DUELING_NETWORK,
                "noisy_network_sigma_start": Config.DQN.NOISY_NETWORK_SIGMA_START,
            }

    class Arch:
        INPUTS_DESC: list[str] = ["distance_to_corner_x", "section_rel_y" ,"velocity", "acceleration", "relative_yaw", "turning_rate",
                                  "next_turn", "second_edge_length", "second_turn", "third_edge_length", "third_turn",
                                  "pitch", "roll", "wheel_0_contact", "wheel_1_contact", "wheel_2_contact", "wheel_3_contact",
                                  "wheel_0_sliding", "wheel_1_sliding", "wheel_2_sliding", "wheel_3_sliding",]
        ENABLE_BRAKE: bool = False
        OUTPUTS_DESC: list[str] = ["release", "forward", "right", "left", "forward_right", "forward_left"]
        ACTIVATED_KEYS_PER_OUTPUT: list[tuple[int]] = [(1, 1, 1, 1), (1, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0), (1, 0, 0, 1), (1, 1, 0, 0)]
        if ENABLE_BRAKE:
            OUTPUTS_DESC += ["brake", "forward_brake", "right_brake", "left_brake", "forward_right_brake", "forward_left_brake"]
            ACTIVATED_KEYS_PER_OUTPUT += [(0, 0, 1, 0), (1, 0, 1, 0), (0, 0, 1, 1), (0, 1, 1, 0), (1, 0, 1, 1), (1, 1, 1, 0)]

        REWARD_DESC: str = "distance travelled projected on the section's x axis (progression on the track)"

        INPUT_SIZE: int = len(INPUTS_DESC)
        OUTPUT_SIZE: int = len(OUTPUTS_DESC)

        LAYER_SIZES: list[int] = [256, 256]
        VALUE_ADVANTAGE_LAYER_SIZE: int = 128

        @staticmethod
        def get_number_of_hidden_layers():
            base_layers = len(Config.Arch.LAYER_SIZES)
            if Config.DQN.ENABLE_DUELING_NETWORK:
                return base_layers + 1
            return base_layers

        @staticmethod
        def get_architecture_description():
            return {
                "inputs": Config.Arch.INPUTS_DESC,
                "outputs": Config.Arch.OUTPUTS_DESC,
                "input_size": Config.Arch.INPUT_SIZE,
                "output_size": Config.Arch.OUTPUT_SIZE,
                "layer_sizes": Config.Arch.LAYER_SIZES,
                "value_advantage_layer_size": Config.Arch.VALUE_ADVANTAGE_LAYER_SIZE,
                "number_of_hidden_layers": Config.Arch.get_number_of_hidden_layers(),
                "reward_description": Config.Arch.REWARD_DESC
            }