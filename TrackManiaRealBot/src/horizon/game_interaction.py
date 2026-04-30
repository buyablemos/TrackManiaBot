
from tminterface.interface import TMInterface

from  ..config import Config
from ..utils.utils import get_default_map

def launch_map(iface: TMInterface) -> None:
    """
    Launch the map and set the speed
    :param iface: the TMInterface instance
    :return: None
    """
    iface.execute_command(f"map {get_default_map()}")
    iface.set_speed(Config.Game.GAME_SPEED)


def send_input(iface: TMInterface, move: int) -> None:
    """
    Send the input to the TMInterface
    :param iface: the TMInterface instance
    :param move: the move to send {0: stop, 1: accelerate, 2: right, 3: left, 4: accelerate + right, 5: accelerate + left,
    6: brake, 7: accelerate + brake, 8: right + brake, 9: left + brake, 10: accelerate + right + brake, 11: accelerate + left + brake}
    :return: None
    """
    match move:
        case 0:
            iface.set_input_state(accelerate=False, left=False, right=False, brake=False)
        case 1:
            iface.set_input_state(accelerate=True, left=False, right=False, brake=False)
        case 2:
            iface.set_input_state(accelerate=False, left=False, right=True, brake=False)
        case 3:
            iface.set_input_state(accelerate=False, left=True, right=False, brake=False)
        case 4:
            iface.set_input_state(accelerate=True, left=False, right=True, brake=False)
        case 5:
            iface.set_input_state(accelerate=True, left=True, right=False, brake=False)
        case 6:
            iface.set_input_state(accelerate=False, left=False, right=False, brake=True)
        case 7:
            iface.set_input_state(accelerate=True, left=False, right=False, brake=True)
        case 8:
            iface.set_input_state(accelerate=False, left=False, right=True, brake=True)
        case 9:
            iface.set_input_state(accelerate=False, left=True, right=False, brake=True)
        case 10:
            iface.set_input_state(accelerate=True, left=False, right=True, brake=True)
        case 11:
            iface.set_input_state(accelerate=True, left=True, right=False, brake=True)
        case _:
            iface.set_input_state(accelerate=False, left=False, right=False, brake=False)
