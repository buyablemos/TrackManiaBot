import subprocess
import pygetwindow as gw
import pywinauto
import psutil

from time import sleep
from ReadWriteMemory import ReadWriteMemory

from ..config import Config


class TMLauncher:

    @staticmethod
    def launch_game() -> None:
        """
        Launch a single Trackmania client
        :return: None
        """
        from .utils import get_executable_path

        executable, path_to_executable = get_executable_path()
        subprocess.Popen([executable], cwd=path_to_executable, shell=True)

    @staticmethod
    def focus_windows() -> None:
        """
        Focus all Trackmania windows
        :return: None
        """
        windows = gw.getWindowsWithTitle(Config.Game.TMI_WINDOW_NAME)
        print(f"Focusing {len(windows)} windows")
        for window in windows:
            app = pywinauto.Application().connect(handle=window._hWnd)
            dlg = app.top_window()
            dlg.set_focus()
            sleep(0.3)

    @staticmethod
    def remove_fps_cap():
        """
        Remove the FPS cap in Trackmania. It used reverse engineering to find the memory addresses of the FPS cap.
        :return: None
        """
        process = filter(lambda p: p.name() == Config.Game.PROCESS_NAME, psutil.process_iter())
        rwm = ReadWriteMemory()
        for p in process:
            pid = int(p.pid)
            process = rwm.get_process_by_id(pid)
            process.open()
            process.write(0x005292F1, 4294919657)
            process.write(0x005292F1 + 4, 2425393407)
            process.write(0x005292F1 + 8, 2425393296)
            process.close()

    @staticmethod
    def kill_game_process():
        """
        Kills all Trackmania processes.
        :return: None
        """
        try:
            tm_processes = [p for p in psutil.process_iter() if p.name() == Config.Game.PROCESS_NAME]

            if not tm_processes:
                print("No Trackmania processes found.")
                return

            # Kill each Trackmania process
            for p in tm_processes:
                try:
                    pid = p.pid
                    p.kill()
                except Exception as e:
                    print(f"Error killing PID {p.pid}: {e}")

            gone, still_alive = psutil.wait_procs(tm_processes, timeout=3)

            # If some processes are still alive, force them to quit
            for p in still_alive:
                print(f"Force killing PID: {p.pid}...")
                try:
                    p.kill()
                except:
                    pass

            print("All Trackmania processes killed.")
        except Exception as e:
            print(f"Error killing Trackmania processes: {e}")

    @staticmethod
    def click_in_game_window():
        """
        Click in the Trackmania window to focus it.
        :return: None
        """
        windows = gw.getWindowsWithTitle(Config.Game.TMI_WINDOW_NAME)
        if windows:
            window = windows[0]
            window.activate()
            window.click_input()
            print(f"Clicked in {Config.Game.TMI_WINDOW_NAME} window")
        else:
            print(f"{Config.Game.TMI_WINDOW_NAME} window not found")