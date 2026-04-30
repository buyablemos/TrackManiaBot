import signal
import multiprocessing
import time

from time import sleep
from tminterface.interface import TMInterface

from .ppo import PPOAgent
from .dqn import DQNAgent
from .game_interaction import launch_map
from ..utils.tm_launcher import TMLauncher
from ..config import Config
from .events import Events

class Worker(multiprocessing.Process):
    def __init__(self, algorithm, events: Events, shared_dict):
        super().__init__()
        self.server_id = 0
        self.algorithm = algorithm

        self.shared_dict = shared_dict

        self.events = events

        self.agent = None
        self.iface = None

    def start_game(self):
        TMLauncher.launch_game()
        sleep(5)
        TMLauncher.remove_fps_cap()
        self.events.embed_game_event.set()
        sleep(2)

    def connect_agent(self):
        if self.algorithm == "PPO":
            print("Using PPO")
            self.agent = PPOAgent(self.shared_dict)
        elif self.algorithm == "DQN":
            print("Using DQN")
            self.agent = DQNAgent(self.shared_dict)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        self.iface = TMInterface(f"TMInterface{self.server_id}")

        signal.signal(signal.SIGINT, self.close_signal_handler)
        signal.signal(signal.SIGTERM, self.close_signal_handler)
        self.iface.register(self.agent)

    def close_signal_handler(self, sig, frame):
        self.iface.execute_command("quit")
        self.iface.close()

    def run(self):
        while True:
            start_time = time.time()
            max_runtime = Config.Game.RESTART_INTERVAL_SECONDS

            try:
                self.start_game()
                self.connect_agent()
                sleep(5)
                if not self.iface.registered:
                    raise Exception("Agent not registered")
                self.agent.load_model()
                launch_map(self.iface)

                while self.iface.running:

                    if time.time() - start_time > max_runtime:
                        print(f"Session timed out after {max_runtime} seconds, restarting...")
                        break

                    if self.events.choose_map_event.is_set():
                        launch_map(self.iface)
                        self.events.choose_map_event.clear()

                    if self.events.print_state_event.is_set():
                        print(self.agent)
                        self.events.print_state_event.clear()

                    if self.events.load_model_event.is_set():
                        self.agent.load_model()
                        self.events.load_model_event.clear()

                    if self.events.save_model_event.is_set():
                        self.agent.save()
                        self.events.save_model_event.clear()

                    if self.events.quit_event.is_set():
                        self.close_signal_handler(None, None)
                        self.events.quit_event.clear()
                        return

                    sleep(0)

                if self.agent and not self.events.quit_event.is_set():
                    print("Saving model...")
                    self.agent.save()

                if self.iface.running:
                    self.iface.close()

                TMLauncher.kill_game_process()
            except Exception as e:
                print(f"Error: {e}")
                if hasattr(self, 'iface') and self.iface is not None:
                    try:
                        self.iface.close()
                    except:
                        pass
                TMLauncher.kill_game_process()

            self.server_id = (1 + self.server_id) % 2
            print("Worker process terminated, restarting...")
            sleep(20)


