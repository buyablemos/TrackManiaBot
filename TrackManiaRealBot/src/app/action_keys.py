import tkinter as tk
from tkinter import ttk
import numpy as np

from ..config import Config

class ActionKeys:
    def __init__(self, parent, row_pos, col_pos, key_size=40, padding=3, margin=10, ):
        self.parent = parent
        self.frame = None
        self.key_size = key_size
        self.padding = padding
        self.margin = margin
        self.frame_size = (3 * self.key_size + 2 * self.padding, 2 * self.key_size + self.padding)

        self.action_frame = ttk.Frame(self.parent, width=3 * self.frame_size[0] + 2 * margin, height=3 * self.frame_size[1] + margin)
        self.action_frame.grid(row=row_pos, column=col_pos, padx=10, pady=10)

        self.zqsd_keys = []

        self.create_actions()

    def create_actions(self):
        for i in range(Config.Arch.OUTPUT_SIZE):
            action = ZQSDKeys(self.action_frame, i//3, i%3, i, self.key_size, self.padding)
            self.zqsd_keys.append(action)

    def update_keys(self, q_values):
        is_random = q_values["is_random"]
        q_values = np.array([q_values[action] for action in Config.Arch.OUTPUTS_DESC])
        off_color = "gray"
        if is_random:
            on_color = "orange"
            for i, action in enumerate(self.zqsd_keys):
                if q_values[i] == 1:
                    action.update_keys(q_values[i], on_color, off_color)
                else:
                    action.update_keys(q_values[i], off_color, off_color)
        else:
            on_color = "red"
            best_color = "green"
            # Normalize the q_values
            for i, action in enumerate(self.zqsd_keys):
                action.update_q_value_label(q_values[i])
            q_values = self.normalize_q_values(q_values)
            for i, action in enumerate(self.zqsd_keys):
                if q_values[i] == 1:
                    action.update_keys(q_values[i], best_color, off_color)
                else:
                    action.update_keys(q_values[i], on_color, off_color)

    @staticmethod
    def normalize_q_values(q_values):
        q_values = np.array(q_values, dtype=float)
        q_min = np.min(q_values)
        q_max = np.max(q_values)

        if q_max == q_min:
            return np.zeros_like(q_values)

        return (q_values - q_min) / (q_max - q_min)


class ZQSDKeys:
    def __init__(self, parent, row, col, index, key_size=40, padding=3):
        self.parent = parent
        self.row = row
        self.col = col

        self.key_size = key_size
        self.padding = padding
        self.frame_size = (3 * self.key_size + 2 * self.padding, 2 * self.key_size + self.padding)

        self.activated_keys = Config.Arch.ACTIVATED_KEYS_PER_OUTPUT[index]
        self.key_colors = ["green" if key == 1 else "gray" for key in self.activated_keys]

        self.up = None
        self.keys = []
        self.q_label = None

        self.create_keys()

    def create_keys(self):
        self.frame = tk.Frame(self.parent, width=3 * self.frame_size[0], height=3 * self.frame_size[1])
        self.frame.grid(row=self.row, column=self.col, padx=10, pady=10)

        self.q_label = tk.Label(self.frame, text="Q: 0.00", font=("Arial", 10), fg="black", anchor="center", justify="center")
        self.q_label.grid(row=0, column=0, columnspan=3, sticky="ew")

        for i, (r, c) in enumerate([(1, 1), (2, 0), (2, 1), (2, 2)]):  # up, left, down, right
            base_color = self.key_colors[i]
            container = tk.Frame(self.frame, width=self.key_size, height=self.key_size, bg=base_color)
            container.grid(row=r, column=c, padx=self.padding, pady=self.padding, sticky="nsew")

            gauge = tk.Frame(container, bg="white")
            gauge.place(relx=0, rely=1.0, relwidth=1.0, anchor='sw', relheight=0.0)

            self.keys.append((container, gauge))

    def update_keys(self, q_value, color, bg_color="gray"):
        for i, (container, gauge) in enumerate(self.keys):
            if self.activated_keys[i] == 1:
                container.config(bg=bg_color)
                gauge.config(bg=color)
                gauge.place_configure(relheight=q_value)
            else:
                container.config(bg=bg_color)
                gauge.place_configure(relheight=0)

    def update_q_value_label(self, q_value):
        self.q_label.config(text=f"Q: {q_value:.2f}")