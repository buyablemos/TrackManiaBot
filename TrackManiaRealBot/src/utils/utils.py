import os
import torch
import time
import functools
import shutil
import math

from dotenv import load_dotenv
from time import sleep

from .tm_launcher import TMLauncher
from ..config import Config

load_dotenv()
profile_times = {}

def get_executable_path() -> tuple[str, str]:
    return os.getenv("EXECUTABLE"), os.getenv("EXECUTABLE_PATH")

def get_default_map() -> str:
    return os.path.join(os.getenv("DEFAULT_MAP_PATH"), os.getenv("DEFAULT_MAP_NAME"))

def get_states_path() -> str:
    return os.getenv("STATES_PATH")

def get_replays_path() -> str:
    return os.getenv("REPLAYS_PATH")

def get_random_states() -> list[str]:
    """
    Get the random states from the TMInterface directory
    :return: the random states
    """
    if not os.path.exists(os.path.join(get_states_path(), Config.Paths.MAP)):
        print(f"Map directory not found at {os.path.join(get_states_path(), Config.Paths.MAP)}")
        return []
    else:
        states = [state for state in os.listdir(os.path.join(get_states_path(), Config.Paths.MAP))
                  if state.endswith(".bin")]

        # Sort by the number in the filename
        def extract_number(filename):
            try:
                return int(filename.split('.')[0])
            except ValueError:
                return float('inf')

        states.sort(key=extract_number)

        return [os.path.join(Config.Paths.MAP, state) for state in states]

def trigger_map_event(event):
    event.set()
    sleep(2)
    TMLauncher.remove_fps_cap()
    TMLauncher.focus_windows()

def copy_model_to_latest(directory: str) -> None:
    """
    Copy the model to the latest directory
    :return: None
    """
    for file in os.listdir(directory):
        if file.endswith(".gbx"):
            continue
        shutil.copy(os.path.join(directory, file), Config.Paths.LATEST_MODEL_PATH)

def save_pb(directory: str, time: str, discard) -> bool:
    """
    Save the pb to the directory and rename it to the time
    :param directory: The directory to save the pb
    :param time: The duration of the pb
    :param discard: If True, don't save this pb, just delete it
    :return: True if the pb was saved, False otherwise
    """
    map_path = get_default_map()
    map_name =  os.path.basename(map_path).split(".")[0]
    replay_name_suffix = f"{map_name}.Replay.gbx"
    replays_path = get_replays_path()
    for file in os.listdir(replays_path):
        if file.endswith(replay_name_suffix):
            source_path = os.path.join(replays_path, file)
            target_path = os.path.join(directory, f"{time}.Replay.gbx")
            if not discard:
                shutil.copy2(source_path, target_path)
            os.remove(source_path)
            return True

    return False

def get_device_info(device: str):
    """
    Get the device information
    :param device: The device to get the information for 'cuda' or 'cpu'
    :return: The device name
    """
    device_name = "CPU"
    if device == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        import platform
        if platform.system() == "Windows":
            import subprocess
            result = subprocess.check_output("wmic cpu get name", shell=True).decode().strip().split("\n")[1]
            device_name = result
        else:
            import os
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            device_name = line.split(':')[1].strip()
                            break
    return device_name

def from_schedule(schedule: list[tuple[int, float]], time: int) -> float:
    run_steps = time / Config.Game.INTERVAL_BETWEEN_ACTIONS

    if not schedule:
        raise ValueError("Schedule must not be empty")
    # Sort schedule by step to ensure correct order
    schedule = sorted(schedule, key=lambda x: x[0])
    # Check if we have a value for step 0
    if schedule[0][0] != 0:
        raise ValueError("Schedule must contain a value for step 0")
    # If time is beyond the last scheduled step, return the last value
    if run_steps >= schedule[-1][0]:
        return schedule[-1][1]
    # Find the surrounding steps for interpolation
    for i in range(len(schedule) - 1):
        if schedule[i][0] <= run_steps < schedule[i + 1][0]:
            # Get the surrounding points
            begin_step, begin_value = schedule[i]
            end_step, end_value = schedule[i + 1]
            # Calculate the ratio between end and begin values
            ratio = begin_value / end_value
            # Calculate the annealing period
            annealing_period = end_step - begin_step
            # Calculate the current value using exponential decay
            return begin_value * math.exp(-math.log(ratio) * (run_steps - begin_step) / annealing_period)
    # This should never happen due to the earlier checks
    raise ValueError("Invalid schedule or time value")

def profile_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if func.__name__ not in profile_times:
            profile_times[func.__name__] = []
        profile_times[func.__name__].append(end_time - start_time)
        return result
    return wrapper

def print_profile_times():
    for func_name, times in profile_times.items():
        avg_time = sum(times) / len(times)
        print(f"{func_name}: {avg_time * 1000:.4f} ms (called {len(times)} times)")