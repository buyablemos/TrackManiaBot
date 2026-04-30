import torch
import os
import json
import numpy as np
import concurrent.futures

from tminterface.client import Client
from tminterface.interface import TMInterface
from tminterface.structs import SimStateData
from collections import deque
from abc import ABC, abstractmethod
from time import sleep
from datetime import datetime

from .game_interaction import launch_map
from ..map_interaction.agent_position import AgentPosition
from ..config import Config
from ..utils.tm_logger import TMLogger
from ..utils.utils import get_device_info, get_random_states, copy_model_to_latest

class Agent(Client, ABC):
    def __init__(self, shared_dict, algorithm: str) -> None:
        """
        Constructor for the Agent class
        :param shared_dict: a dictionary to share data between threads
        :param algorithm: the algorithm used for training, either "DQN" or "PPO"
        """
        super(Agent, self).__init__()
        self.algorithm = algorithm

        self.agent_position = AgentPosition()

        #self.device = torch.device("cpu")
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.reward = 0.0

        self.prev_positions = deque(maxlen=5 * Config.Game.NUMBER_OF_ACTIONS_PER_SECOND)  # 5-second long memory
        self.current_state: torch.Tensor = torch.zeros(Config.Arch.INPUT_SIZE, dtype=torch.float, device=self.device)

        self.prev_velocity = None
        self.has_finished = False
        self.iterations = 0
        self.ready = False
        self.spawn_point = 0

        self.best_reward = 0.0
        self.personal_best = float("inf")
        self.save_pb = False
        self.previous_finish_time: str = ""

        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.shared_dict = shared_dict
        self.eval: bool = shared_dict["eval"]
        self.game_speed: int = shared_dict["game_speed"]
        self.logger: TMLogger = TMLogger(self.algorithm, get_device_info(self.device.type))

        self.random_states = get_random_states()
        self.unlocked_states = len(self.random_states) - 1 if Config.Game.UNLOCK_ALL_STATES else 0

        self.reward_requirements = self.agent_position.get_reward_requirements_for_checkpoint(len(self.random_states))

        self.total_time = 0
        self.current_run_start_time = 0

        self.epsilon = 0
        self.epsilon_boltzmann = 0

    def __str__(self) -> str:
        """
        Print the current state of the agent
        :return: the current state
        """
        if self.current_state is None:
            return "No state available"
        state = ""
        for name, value in zip(Config.Arch.INPUTS_DESC, self.current_state):
            state += f"{name}: {value:.2f}\n"

        return state

    def load_hyperparameters(self, path: str) -> dict:
        """
        Load the hyperparameters from the checkpoint
        :param path: the path to the checkpoint
        :return: the hyperparameters as a dictionary
        """
        hyperparameters = {}
        hyperparameters_path = os.path.join(path, Config.Paths.STAT_FILE_NAME)
        if os.path.exists(hyperparameters_path):
            with open(hyperparameters_path, "r") as f:
                stats = json.load(f)
                hyperparameters = stats["hyperparameters"]
                self.iterations = stats["statistics"]["total number of runs"]
                self.total_time = stats["statistics"]["training time"]["milliseconds"]
        return hyperparameters

    def on_registered(self, iface: TMInterface) -> None:
        """
        Called when the agent is registered to the TMInterface
        :param iface: the TMInterface instance
        :return: None
        """
        iface.set_timeout(-1)
        print(f"Registered to {iface.server_name}")
        iface.log(f"Loaded a {self.algorithm} agent")
        iface.execute_command("toggle_console")

    def _save_stats(self) -> str:
        """
        Save the statistics to a file
        :return: None
        """
        # Before dumping, if the directory is not set, ask the user for a directory
        result = self.logger.dump()

        if not result:
            print("Failed to save the model")
            return ""

        return result

    @abstractmethod
    def save_model(self, directory: str) -> None:
        """
        Save the model to a file
        :param directory: the directory to save the model to
        :return: None
        """
        pass

    @abstractmethod
    def get_learning_rate(self) -> float:
        """
        Get the learning rate of the model
        :return: the learning rate
        """
        pass

    def save(self) -> None:
        """
        Save the model and the statistics
        :return: None
        """
        directory = self.shared_dict["model_path"].value
        if not directory:
            directory = datetime.now().strftime("%m-%d_%H-%M")
            directory = os.path.join(Config.Paths.MODELS_PATH, directory)
            self.shared_dict["model_path"].value = directory

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.logger.set_directory(directory)
        self.logger.update_learning_rate(self.get_learning_rate())
        result = self._save_stats()
        if not result:
            print("Failed to save the model")
            return

        self.save_model(directory)
        copy_model_to_latest(directory)
        print(f"Model saved to {directory} and in the latest model directory")

    def update_state(self, simulation_state: SimStateData) -> None:
        """
        Update the state of the agent
        :param simulation_state: the simulation state
        :return: None
        """
        current_velocity = np.array(simulation_state.velocity)
        acceleration_scalar = 0
        velocity_norm = 0
        if self.prev_velocity is not None:
            delta_v = current_velocity - self.prev_velocity
            velocity_norm = np.linalg.norm(current_velocity)
            if velocity_norm > 1e-5:
                direction = current_velocity / velocity_norm
                acceleration_scalar = np.dot(delta_v, direction) / (Config.Game.INTERVAL_BETWEEN_ACTIONS / 1000)
        self.prev_velocity = current_velocity

        agent_absolute_position = (simulation_state.position[0], simulation_state.position[2])
        distance_to_corner, section_relative_y, next_turn, second_edge_length, second_turn, third_edge_length, third_turn = self.agent_position.get_relative_position_and_next_turns(
            agent_absolute_position)
        relative_yaw = self.agent_position.get_car_orientation(simulation_state.yaw_pitch_roll[0],
                                                               agent_absolute_position)
        wheels = simulation_state.simulation_wheels
        turning_rate = simulation_state.scene_mobil.turning_rate
        pitch, roll = simulation_state.yaw_pitch_roll[1:3]
        self.current_state = torch.tensor([
            distance_to_corner,
            section_relative_y,
            velocity_norm / 50,
            acceleration_scalar / 15,
            relative_yaw,
            turning_rate,
            next_turn,
            second_edge_length,
            second_turn,
            third_edge_length,
            third_turn,
            pitch,
            roll,
            wheels[0].real_time_state.has_ground_contact,
            wheels[1].real_time_state.has_ground_contact,
            wheels[2].real_time_state.has_ground_contact,
            wheels[3].real_time_state.has_ground_contact,
            wheels[0].real_time_state.is_sliding,
            wheels[1].real_time_state.is_sliding,
            wheels[2].real_time_state.is_sliding,
            wheels[3].real_time_state.is_sliding,
        ], dtype=torch.float, device=self.device)

    def get_reward(self, simulation_state: SimStateData, done: torch.tensor, time: int) -> torch.Tensor:
        """
        Get the reward for the current state
        :param simulation_state: the simulation state
        :param done: whether the simulation is done
        :param time: The current time in milliseconds
        :return: the reward
        """
        if done.item() and not self.has_finished:
            return torch.tensor(Config.Game.REWARD_PER_MS * Config.Game.INTERVAL_BETWEEN_ACTIONS, device=self.device)

        if not self.prev_positions:
            return torch.tensor(0, device=self.device)

        prev_position = self.prev_positions[-1]
        current_position = (simulation_state.position[0], simulation_state.position[2])
        distance_advanced_along_centerline = self.agent_position.get_distance_reward(prev_position, current_position)

        current_reward = distance_advanced_along_centerline * Config.Game.REWARD_PER_METER_ALONG_CENTERLINE

        if self.has_finished:
            bonus = self.final_bonus(time)
            current_reward += bonus


        current_reward += Config.Game.REWARD_PER_MS * Config.Game.INTERVAL_BETWEEN_ACTIONS

        return torch.tensor(current_reward, device=self.device)

    def final_bonus(self, time: int):
        """
        Bonus = max_bonus * (per_sec_ratio) ** (time_ref - time)
        time < time_ref → bonus increases
        time > time_ref → bonus decreases
        :time: time in milliseconds
        :return: the final bonus
        """
        #ref = min(self.personal_best, Config.Game.TIME_REF)
        ref = Config.Game.TIME_REF
        return Config.Game.MAX_BONUS * (Config.Game.PER_SEC_RATIO ** ((ref - time) / 1000)) + 100


    def determine_done(self, simulation_state: SimStateData) -> torch.Tensor:
        """
        Determine if the simulation is done
        :param simulation_state: the simulation state
        :return: A tensor indicating if the simulation is done, 1.0 if done, 0.0 otherwise
        """

        if simulation_state.position[1] < 23.9:  # If the car is below the track
            return torch.tensor(1.0, device=self.device, dtype=torch.float)

        if simulation_state.player_info.race_finished:
            self.has_finished = True
            return torch.tensor(1.0, device=self.device, dtype=torch.float)

        if (self.prev_positions and len(self.prev_positions) == (5 * Config.Game.NUMBER_OF_ACTIONS_PER_SECOND)
                and np.linalg.norm(np.array(self.prev_positions[0]) - np.array(self.prev_positions[-1])) < 5):  # If less than 5 meters were travelled in the last 5 seconds
            return torch.tensor(1.0, device=self.device, dtype=torch.float)

        return torch.tensor(0.0, device=self.device, dtype=torch.float)

    def reset(self, iface: TMInterface, time: int) -> None:
        """
        Reset the agent
        :param iface: the TMInterface instance
        :param time: the run time of this run
        :return: None
        """
        if not self.eval:
            self.iterations += 1
            self.shared_dict["reward"].put(self.reward)
            run_duration = time - self.current_run_start_time
            self.logger.add_run(self.iterations, run_duration, self.reward, self.spawn_point, self.has_finished, self.epsilon, self.epsilon_boltzmann)
            self.total_time += run_duration

        if self.iterations % 10 ==  0:
            print(f"Iteration: {self.iterations:<8} reward: {self.reward:<8.2f}")

        if self.reward > self.best_reward:
            self.best_reward = self.reward
            self.unlocked_states = max(np.clip(np.searchsorted(self.reward_requirements, self.best_reward) - 1, 0, len(self.random_states) - 2), self.unlocked_states)

        self.reward = 0.0
        self.prev_positions.clear()
        self.prev_velocity = None
        self.current_state = torch.zeros(Config.Arch.INPUT_SIZE, dtype=torch.float, device=self.device)
        self.refresh_shared_dict()
        if self.has_finished:
            minutes = time // (1000 * 60)
            seconds = (time % (1000 * 60)) / 1000
            formatted_time = f"{int(minutes):02}.{seconds:05.2f}"

            if time < self.personal_best and self.spawn_point == 0:
                self.save()
                self.personal_best = time
                self.shared_dict["personal_best"] = self.personal_best
                print(f"New personal best: {formatted_time}")

            self.has_finished = False
            self.spawn_point = 0

            self.save_pb = True
            self.previous_finish_time = formatted_time
            launch_map(iface)
        else:
            self.spawn_point = 0
            iface.horn()
            iface.execute_command("press enter")

    def refresh_shared_dict(self) -> None:
        """
        Refresh the shared dictionary
        :return: None
        """
        self.eval = self.shared_dict["eval"]
        self.game_speed = self.shared_dict["game_speed"]
