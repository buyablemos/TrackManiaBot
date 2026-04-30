from tminterface.client import Client, run_client
from tminterface.interface import TMInterface
import sys
import numpy as np

class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()
        self.prev_state = None

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time >= 0:
            state = iface.get_simulation_state()
            iface.execute_command(f"press up; steer {np.sin(_time / 1000) * 65535}")
            self.prev_state = state

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        self.log(iface, f"Checkpoint {current}/{target}")

    def log(self, iface, msg):
        iface.execute_command(f"log {msg}")


server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
print(f"Connecting to {server_name}...")
client = MainClient()

run_client(client, server_name)

