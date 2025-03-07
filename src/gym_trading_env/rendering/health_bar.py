# src/gym_trading_env/rendering/health_bar.py

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class HealthBar:
    def __init__(self, initial_health=1000, unit_health=1000, 
                 color_A='red', color_B='orange', 
                 color_N='cyan', line_width=10):
        """
        Initialize the health bar object.
        
        Parameters:
        - initial_health: Initial health
        - unit_health: Health per unit
        - color_A: Color for a full unit of health
        - color_B: Color for the partial unit of health
        - color_N: Color for newly added partial health unit
        - line_width: Width of the health bar line
        """
        self.unit_health = unit_health
        self.health = initial_health
        self.line_width = line_width

        # Color configuration
        self.color_A = color_A  # Full unit health
        self.color_B = color_B  # Partial unit health
        self.color_N = color_N  # Newly added partial health

    def calculate_units(self, health=None):
        """
        Calculate the number of units and extra health corresponding to the health.
        """
        if health is None:
            health = self.health
        total_units = health // self.unit_health
        extra_health = health % self.unit_health
        return total_units, extra_health

    def draw_segment(self, ax, x_start, x_end, color, y_position, zorder=1):
        """
        Draw a segment of the health bar.
        """
        width = x_end - x_start
        if width <= 0:
            return
        segment = Rectangle((x_start, y_position), width, self.line_width, 
                            linewidth=0, edgecolor=None, 
                            facecolor=color, zorder=zorder)
        ax.add_patch(segment)
    
    def get_multiplier(self):
        multiplier = self.health // self.unit_health
        return multiplier

    def draw_on_ax(self, ax, max_health=None):
        """
        Draw the progress bar and time label on the specified Axes.
        
        Parameters:
        - ax: Matplotlib Axes object
        - max_health: Maximum health, used to determine the total length of the progress bar. 
                       If not provided, the initial_health will be used.
        """
        if max_health is None:
            max_health = self.health

        # Set up progress bar parameters
        bar_length = 200  # Can be adjusted as needed
        x_start = 10
        y_position = 5  # Fixed y position
        x_end = x_start + bar_length

        # Calculate the current health bar length
        health_percentage = self.health / max_health
        current_bar_length = bar_length * health_percentage

        # Draw empty health bar background (white)
        self.draw_segment(ax, x_start, x_end, 'white', y_position, zorder=0)

        # Draw black border
        border = Rectangle((x_start, y_position), bar_length, self.line_width, 
                           linewidth=2, edgecolor='black', 
                           facecolor='none', zorder=4)
        ax.add_patch(border)


        self.draw_segment(ax, x_start, x_start + current_bar_length, self.color_A, y_position, zorder=1)

        # Set axis limits and hide axes
        ax.set_xlim(0, x_start + bar_length + 50)  # Leave space for time label
        ax.set_ylim(0, y_position + self.line_width + 20)
        ax.axis('off')  # Turn off axis display
