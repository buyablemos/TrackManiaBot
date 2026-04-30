import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')

class Plot:
    def __init__(self, parent, plot_size=100, title="Real-time Bar Chart", xlabel="Iteration", ylabel="Value"):
        self.parent = parent
        self.plot_size = plot_size
        self.avg_plot_size = plot_size
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        # Data containers
        self.x_data = np.array([], dtype=int)
        self.y_data = np.array([], dtype=float)
        self.avg_y_data = np.array([], dtype=float)
        self.avg_x_data = np.array([], dtype=int)

        # Set up the figure
        self.fig, self.ax = plt.subplots()
        self.bars = None  # BarContainer to store rectangles

        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(True)
        self.ax.set_axisbelow(True)  # Ensure grid is behind bars

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.iterations = 0

    def add_points(self, y_values):
        """Add multiple points at once to avoid multiple redraws"""
        if not y_values:  # If empty list, do nothing
            return

        for y in y_values:
            self.iterations += 1
            self.x_data = np.append(self.x_data, self.iterations)
            self.y_data = np.append(self.y_data, y)

        # Keep only the most recent plot_size points
        if len(self.x_data) > self.plot_size:
            self.x_data = self.x_data[-self.plot_size:]
            self.y_data = self.y_data[-self.plot_size:]

        if len(self.y_data) >= self.avg_plot_size:
            self.avg_y_data = np.append(self.avg_y_data, np.convolve(self.y_data[-self.avg_plot_size:],np.ones(self.avg_plot_size) / self.avg_plot_size, mode='valid'))
            self.avg_y_data = self.avg_y_data[-self.plot_size:]
            self.avg_x_data = self.x_data[-len(self.avg_y_data):]

        # Redraw everything once
        self.ax.clear()
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(True)
        self.ax.set_axisbelow(True)

        self.bars = self.ax.bar(self.x_data, self.y_data, width=1.0, align='center', color='#37504b', label="Reward")

        # Plot the average reward
        if len(self.avg_y_data) > 0:
            self.ax.plot(self.avg_x_data, self.avg_y_data, 'r--', label="Mean Reward (last %d)" % self.avg_plot_size)

            y_start = self.avg_y_data[0]
            baseline = np.linspace(y_start, y_start, len(self.avg_x_data))

            self.ax.fill_between(self.avg_x_data, self.avg_y_data, baseline, color='red', alpha=0.15)

        self.ax.legend(loc='upper left')
        # Set x-limits to follow data range
        self.ax.set_xlim(self.x_data[0] - 0.5, self.x_data[-1] + 0.5)

        # Adjust y-axis automatically
        self.ax.relim()
        self.ax.autoscale_view(scalex=False)

        if len(self.avg_y_data) > 0:
            y_start = self.avg_y_data[0]
            y_end = self.avg_y_data[-1]
            delta = y_end - y_start
            direction = 1 if delta > 0 else -1
            color = 'green' if direction == 1 else 'red'

            arrow_x = self.x_data[-1] + (self.plot_size * 0.02)
            self.ax.annotate(
                '',
                xy=(arrow_x, y_end),
                xytext=(arrow_x, y_start),
                arrowprops=dict(
                    facecolor=color,
                    edgecolor=color,
                    width=2,
                    headwidth=8
                ),
                annotation_clip=False
            )

        self.canvas.draw()

    def add_point(self, y):
        """Compatibility method for adding a single point"""
        self.add_points([y])

    def clear(self):
        self.x_data = np.array([], dtype=int)
        self.y_data = np.array([], dtype=float)
        self.ax.cla()
        self.bars = None
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(True)
        self.ax.set_axisbelow(True)
        self.canvas.draw()

    def close(self):
        plt.close(self.fig)

    def pause(self):
        plt.pause(0.1)
