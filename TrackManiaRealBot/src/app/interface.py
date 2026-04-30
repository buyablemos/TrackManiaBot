import tkinter as tk
import re
import os
import pygetwindow as gw
import win32gui
import win32con

from time import sleep
from tkinter import messagebox
from tkinter import ttk

from src.config import Config
from src.app.plot import Plot
from .action_keys import ActionKeys
from ..utils.utils import trigger_map_event
from ..horizon.events import Events

class Interface:
    def __init__(self, events: Events, shared_dict) -> None:
        self.root = tk.Tk()
        self.root.title("Panel do trenowania bota - Trackmania")
        self.full_screen = False
        self.best_reward = 0

        self.game_frame = None
        self.graph_frame = None
        self.button_frame = None
        self.action_keys: ActionKeys = None
        self.best_reward_label = None
        self.game_speed_slider = None

        self.events: Events = events
        self.shared_dict = shared_dict

        self.evaluation_toggle_variable = tk.IntVar(value=0)

        self.game_geometry = (640, 480)
        self.graph_geometry = (640, 480)

        self.create_game_frame()
        self.create_graph_frame()
        self.create_button_frame()
        self.create_actions_squares()
        self.create_reward_label()
        self.create_game_speed_slider()

        self.root.bind("<F11>", self.toggle_fullscreen)

        self.graph = Plot(parent=self.graph_frame, plot_size=200, title="Reward", xlabel="Iteration", ylabel="Reward")

        self.after_id = None

        self.root.protocol("WM_DELETE_WINDOW", lambda: self.close_window)

    def run(self):
        """Run the main loop"""
        self.root.mainloop()

    def toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode"""
        self.full_screen = not self.full_screen
        self.root.attributes("-fullscreen", self.full_screen)
        return "break"

    def create_game_frame(self):
        """Create the game frame"""
        self.game_frame = tk.Canvas(self.root, width=self.game_geometry[0], height=self.game_geometry[1])
        self.game_frame.grid(row=0, column=0, padx=10, pady=10)

    def create_graph_frame(self):
        """Create the graph frame"""
        self.graph_frame = ttk.Frame(self.root, width=self.graph_geometry[0], height=self.graph_geometry[1])
        self.graph_frame.grid(row=0, column=1, padx=10, pady=10)

    def create_button_frame(self):
        """Create the button frame"""
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.grid(row=2, column=0, padx=10, sticky="n")
        self.load_map_button = ttk.Button(self.button_frame, text="Load the map", command=self.load_map)
        self.load_map_button.grid(row=0, column=0, padx=5, sticky="nsew")
    
        self.print_state_button = ttk.Button(self.button_frame, text="Print the state", command=self.events.print_state_event.set)
        self.print_state_button.grid(row=0, column=1, padx=5,sticky="nsew")

        self.load_model_button = ttk.Button(self.button_frame, text="Load a model", command=self.load_model)
        self.load_model_button.grid(row=0, column=2, padx=5, sticky="nsew")

        self.save_model_button = ttk.Button(self.button_frame, text="Save the model", command=self.save_model)
        self.save_model_button.grid(row=0, column=3, padx=5, sticky="nsew")

        self.quit_button = ttk.Button(self.button_frame, text="Quit", command=self.close_window)
        self.quit_button.grid(row=0, column=4, padx=5, sticky="nsew")

        self.evaluation_toggle = tk.Checkbutton(self.button_frame, text="Evaluation", command=self.toggle_evaluation,
                                                variable=self.evaluation_toggle_variable)
        self.evaluation_toggle.grid(row=0, column=5, padx=5, sticky="nsew")

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def save_model(self):
        """Ask for a new model name using a popup instead of askdirectory"""
        if self.shared_dict["model_path"].value:
            self.events.save_model_event.set()
            return

        def validate():
            name = entry.get().strip()
            if not name:
                messagebox.showerror("Invalid name", "Please enter a model name.")
                return

            if not re.match(r'^[\w\-]+$', name):
                messagebox.showerror("Invalid name", "Use only letters, numbers, underscores and dashes.")
                return

            path = os.path.join(Config.Paths.MODELS_PATH, name)
            if os.path.exists(path):
                messagebox.showerror("Already exists", f"The model '{name}' already exists.")
                return

            os.makedirs(path)
            self.shared_dict["model_path"].value = path
            top.destroy()
            self.events.save_model_event.set()

        top = tk.Toplevel(self.root)
        top.title("Create new model")
        top.geometry("300x150")
        top.transient(self.root)
        top.grab_set()

        label = ttk.Label(top, text="Enter a model name:")
        label.pack(pady=10)

        entry = ttk.Entry(top)
        entry.pack(pady=5)
        entry.focus()

        confirm_button = ttk.Button(top, text="Save", command=validate)
        confirm_button.pack(pady=10)

        top.protocol("WM_DELETE_WINDOW", top.destroy)


    def create_reward_label(self):
        """Create the reward label"""
        self.best_reward_label = ttk.Label(self.root, text=f"Best Reward: {self.best_reward}", font=("Arial", 20))
        self.best_reward_label.grid(row=1, column=1, padx=50, sticky="nsew")

    def create_actions_squares(self):
        self.action_keys = ActionKeys(self.root, 2, 1, key_size=20, padding=3, margin=5)

    def create_game_speed_slider(self):
        """Create the game speed slider"""
        self.game_speed_slider = tk.Scale(self.root, from_=1, to=50, tickinterval=5, label="Game Speed", length=400, resolution=1,
                                          orient="horizontal", command=self.update_game_speed)
        self.game_speed_slider.set(Config.Game.GAME_SPEED)
        self.game_speed_slider.grid(row=1, column=0, padx=10, sticky="nsew")

    def toggle_evaluation(self, new_epsilon_value=None):
        """Send the manual epsilon value to the client"""
        self.shared_dict["eval"] = self.evaluation_toggle_variable.get()

    def update_game_speed(self, value):
        """Update the game speed"""
        self.shared_dict["game_speed"] = int(float(value))

    def update_interface(self):
        if self.events.embed_game_event.is_set():
            self.events.embed_game_event.clear()
            self.embed_trackmania(self.game_frame)

        rewards = []
        # Collect all rewards before processing
        while not self.shared_dict["reward"].empty():
            reward = self.shared_dict["reward"].get()
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_reward_label["text"] = f"Best Reward: {self.best_reward:.2f}"

            # Only collect for graph if not in evaluation mode
            if not self.shared_dict["eval"]:
                rewards.append(reward)
        # Update graph only once with all collected points
        if rewards:
            self.graph.add_points(rewards)
        if self.evaluation_toggle_variable.get() == 1:
            self.action_keys.update_keys(self.shared_dict["q_values"])
        self.after_id = self.root.after(100, self.update_interface)

    def load_map(self):
        self.load_model_button["state"] = "disabled"
        trigger_map_event(self.events.choose_map_event)

    def load_model(self):
        models = [model for model in os.listdir(Config.Paths.MODELS_PATH) if os.path.isdir(os.path.join(Config.Paths.MODELS_PATH, model))]

        # Pop-up window
        top = tk.Toplevel()
        top.title("Load a model")
        top.geometry("200x250")
        top.resizable(False, False)
        top.transient(self.root)
        top.grab_set()

        frame = ttk.Frame(top)
        frame.pack(expand=True, fill="both")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        label = ttk.Label(frame, text="Choose a model")
        label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        listbox = tk.Listbox(frame, selectmode="single")
        for model in models:
            listbox.insert(tk.END, model)
        listbox.insert(tk.END, "New model")
        listbox.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Validate button
        button = tk.Button(frame, text="Load", command=lambda: self.load_model_from_listbox(listbox, top))
        button.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

        top.protocol("WM_DELETE_WINDOW", top.destroy)
        self.root.wait_window(top)

    def load_model_from_listbox(self, listbox, top):
        selected_model = listbox.get(listbox.curselection())
        if selected_model == "New model":
            self.events.load_model_event.set()
        else:
            self.shared_dict["model_path"].value = os.path.join(Config.Paths.MODELS_PATH, selected_model)
            self.events.load_model_event.set()
        top.destroy()

    def on_close(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        
        self.root.quit()
        self.root.destroy()

    def close_window(self):
        """Close the window"""
        response = messagebox.askyesnocancel("Save model", "Do you want to save the model before quitting?")
    
        if response is None:
            return 
    
        if response:
            self.events.save_model_event.set()
            sleep(2)
    
        self.events.quit_event.set()
        self.on_close()

    def embed_trackmania(self, frame):
        windows = gw.getWindowsWithTitle(Config.Game.TMI_WINDOW_NAME)

        if windows:
            self._move_window(frame, windows)
        else:
            print(f"No window with title {Config.Game.TMI_WINDOW_NAME} found, trying {Config.Game.WINDOW_NAME}")
            windows = gw.getWindowsWithTitle(Config.Game.WINDOW_NAME)
            if windows:
                self._move_window(frame, windows)
            else:
                print(f"No window with title {Config.Game.WINDOW_NAME} found")

    def _move_window(self, frame, windows):
        window = windows[0]
        hwnd = window._hWnd
        win32gui.SetParent(hwnd, frame.winfo_id())
        width, height = frame.winfo_width(), frame.winfo_height()
        win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, win32con.WS_VISIBLE)
        win32gui.MoveWindow(hwnd, 0, 0, width, height, True)
        if width != self.game_geometry[0] or height != self.game_geometry[1]:
            win32gui.MoveWindow(hwnd, 0, 0, self.game_geometry[0], self.game_geometry[1], True)
        win32gui.SetForegroundWindow(hwnd)
