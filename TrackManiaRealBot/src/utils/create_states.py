from tminterface.client import Client, run_client
from tminterface.interface import TMInterface

from src.config import Config

class SaveStateClient(Client):
    def __init__(self) -> None:
        super(SaveStateClient, self).__init__()
        self.state_dir = Config.Paths.MAP
        self.last_state_id = 0

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time >= 0 and _time % Config.Game.STATES_INTERVAL == 0:
            iface.execute_command(f"save_state {self.state_dir}/{self.last_state_id}.bin")
            self.last_state_id += 1
            iface.horn()

if __name__ == "__main__":
    server_name = f"TMInterface0"
    print(f"Connecting to {server_name}...")
    client = SaveStateClient()

    run_client(client, server_name)