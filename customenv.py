import pygame
import matplotlib.pyplot as plt
import numpy as np
import gymnasium
import math
import cv2

from gymnasium.spaces import Text, Box, Discrete
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Ball, Goal, Wall
from gymnasium import ObservationWrapper, spaces


def place_wall_segment(grid, x, y, length, orientation, color="grey"):
    """
    Places a wall segment on the grid.

    Parameters:
        grid: The MiniGrid grid object.
        x, y: Starting coordinates of the wall.
        length: Length of the wall.
        orientation: 'horizontal' or 'vertical'.
        color: Color of the wall (default is "grey").
    """
    for i in range(length):
        if orientation == "horizontal":
            grid.set(x + i, y, Wall(color=color))  # Place horizontally
        elif orientation == "vertical":
            grid.set(x, y + i, Wall(color=color))  # Place vertically

# Hexagon Drawing Helper Functions
def _draw_hexagon_border(img, center_x, center_y, hex_radius, color, thickness=2):
    """
    Draw a single hexagon on the img array.
    """
    height, width, _ = img.shape
    vertices = []
    for k in range(6):
        angle = 2 * math.pi * k / 6  # 60-degree angles
        vertex_x = int(center_x + hex_radius * math.cos(angle))
        vertex_y = int(center_y + hex_radius * math.sin(angle))
        vertices.append((vertex_x, vertex_y))

    # Draw lines between the vertices
    for v1, v2 in zip(vertices, vertices[1:] + [vertices[0]]):
        for t in range(-thickness // 2, thickness // 2 + 1):  # Loop for thickness
            _draw_line(img, (v1[0] + t, v1[1] + t), (v2[0] + t, v2[1] + t), color)

def _draw_filled_hexagon(img, center_x, center_y, hex_radius, fill_color):
    """
    Draw a filled hexagon on the img array.
    """
    # Define the vertices of the hexagon
    vertices = [
        (
            int(center_x + hex_radius * math.cos(2 * math.pi * k / 6)),
            int(center_y + hex_radius * math.sin(2 * math.pi * k / 6))
        )
        for k in range(6)
    ]

    # Create a polygon fill inside the hexagon using Bresenham's algorithm
    for y in range(center_y - hex_radius, center_y + hex_radius + 1):
        for x in range(center_x - hex_radius, center_x + hex_radius + 1):
            if _is_point_in_polygon(x, y, vertices):
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:  # Ensure it's inside the image bounds
                    img[y, x] = fill_color

def _is_point_in_polygon(x, y, vertices):
    """
    Check if a point is inside a polygon using the ray-casting algorithm.
    """
    n = len(vertices)
    inside = False

    px, py = x, y
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        if min(y1, y2) < py <= max(y1, y2) and px <= max(x1, x2):
            if y1 != y2:
                xinters = (py - y1) * (x2 - x1) / (y2 - y1) + x1
            if x1 == x2 or px <= xinters:
                inside = not inside

    return inside
                 
def _draw_line(img, start, end, color):
    """
    Draw a line between two points on the img array.
    """
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))
    for i in range(steps + 1):
        x = int(x1 + i * dx / steps)
        y = int(y1 + i * dy / steps)
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:  # Check bounds
            img[y, x] = color

class SoccerBall(Ball):
    def __init__(self):
        super().__init__(color='grey')  # Base color is grey

    def render(self, img):
        """
        Render the soccer ball as a pink circle with a simple crosshair pattern.
        """
        # Get the dimensions of the tile
        height, width, _ = img.shape
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 2

        # Draw the colored circle
        for y in range(height):
            for x in range(width):
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                    img[y, x] = [255, 255, 255]  # White RGB

        # Add hexagonal grid pattern
        hex_radius = radius // 2  # Adjust hexagon size
        hex_height = int(math.sqrt(3) * hex_radius)  # Height of each hexagon
        gap = hex_radius // 4 # Extra gap for spacing
        
        row_spacing = hex_height + (gap * 2)  # Add a gap for more space
        col_spacing = hex_height + (gap * 3)  # Extra gap for spacing
        
        row_index = 0
        for i in range(-radius + hex_radius - gap, radius, row_spacing): # Fine-tuned row spacing
            col_index = 0
            for j in range(-radius + hex_radius, radius, col_spacing): # Fine-tuned column spacing
                # Offset rows to create a grid pattern
                x_offset = hex_radius if (i // row_spacing) % 2 == 1 else 0 # Offset for odd rows
                hex_center_x = center_x + j + x_offset
                hex_center_y = center_y + i

                # Draw a hexagon if it falls within the circle
                if (hex_center_x - center_x) ** 2 + (hex_center_y - center_y) ** 2 <= radius ** 2:
                    # Check if the current hexagon should be filled
                    if (row_index + col_index) % 2 == 0:  # Condition for filled hexagons
                        _draw_filled_hexagon(img, hex_center_x, hex_center_y, hex_radius, [0, 0, 0])  # Black fill
                    else:
                        # Draw only the border for unfilled hexagons
                        _draw_hexagon_border(img, hex_center_x, hex_center_y, hex_radius, [0, 0, 0], thickness=2)  # Black border

            col_index += 1
        row_index += 1
               
        # Add a white circular outline
        cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), thickness=2)
        
        print("Debugging: Rendered soccer ball with a simple crosshair pattern.")


class CustomMazeSoccerBallEnv(MiniGridEnv):
    def __init__(self, size=23):
        mission_space = Text(max_length=100)  # Adjust max_length as needed
        super().__init__(
            grid_size=size,
            max_steps=4 * size ** 2,
            see_through_walls=False,
            mission_space=mission_space,  # Valid mission space
            render_mode='human',  # Enable human rendering
        )
        self.size = size
        self.action_space = Discrete(5)  # Add an extra action for 'kick'
        
    def place_agent(self, fixed_start=False):
        """
        Place the agent at a fixed or random starting position.
        
        Parameters:
        - fixed_start (bool): If True, place the agent in a fixed position.
        """
        if fixed_start:
            # Define the fixed starting position and direction
            self.agent_pos = (self.width // 2, self.height // 2)  # Example fixed position (1, 1)
            self.agent_dir = 0       # Example direction (0: facing right)
            self.grid.set(*self.agent_pos, None)  # Clear the starting cell
        else:
            # Call the parent class's place_agent method for random placement
            super().place_agent()

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Add maze walls
        vertical_wall_positions = [ # AI @ CityU
            (5, 2),
            (11, 2),
            (12, 2),
            (4, 3),
            (5, 3),
            (6, 3),
            (11, 4),
            (12, 4),
            (15, 4),
            (21, 4),
            (3, 5),
            (4, 5),
            (6, 5),
            (7, 5),
            (11, 6),
            (12, 6),
            (15, 6),
            (17, 6),
            (19, 6),
            (21, 6),
            (3, 7),
            (4, 7),
            (5, 7),
            (6, 7),
            (7, 7),
            (2, 8),
            (3, 8),
            (7, 8),
            (8, 8),
            (11, 8),
            (12, 8),
            (15, 8),
            (2, 10),
            (3, 10),
            (7, 10),
            (8, 10),
            (11, 10),
            (12, 10),
            (3, 14),
            (18, 14),
            (21, 14),
            (7, 15),
            (10, 15),
            (3, 16),
            (16, 16),
            (18, 16),
            (21, 16),
            (13, 17),
            (3, 18),
            (7, 18),
            (10, 18),
            (16, 18),
            (18, 18),          
            (21, 18),
            (13, 19),
            (3, 20),
            (4, 20),
            (5, 20),
            (7, 20),
            (10, 20),
            (16, 20),
            (18, 20),            
            (19, 20),
            (20, 20),
            (21, 20)
        ]
        horizontal_wall_positions = [
            (4, 14),
            (9, 17),
            (10, 17),
            (10, 21),
            (13, 22),
            (14, 20),
            (15, 22),
            (16, 3),
            (16, 10),
            (18, 3),
            (18, 5),
            (18, 8),
            (18, 10),
            (20, 3),
            (20, 8)         
        ]
        for x, y in vertical_wall_positions:
            place_wall_segment(self.grid, x, y, 2, "vertical", color="grey")
        for x, y in horizontal_wall_positions:
            place_wall_segment(self.grid, x, y, 2, "horizontal", color="grey")

        self.goal = Goal()
        self.put_obj(self.goal, width - 2, height // 2)

        # Place the agent
        self.place_agent(fixed_start=True)  # Start with a fixed position
        self.mission = "Find the ball and kick it into the goal!"

        ball_x = width // 4 *3
        ball_y = height // 2
        # self.ball = Ball(color='yellow')
        self.ball = SoccerBall()  # Use the SoccerBall class
        self.put_obj(self.ball, ball_x, ball_y)
        print(f"Ball placed at: ({ball_x}, {ball_y})")  # Add debug print
        print(f"Tile size: {self.tile_size}")
        print(f"Ball radius: {self.tile_size // 2}")
        
    def _distance(self, pos1, pos2):
        """
        Calculate the Manhattan distance between two positions.
        
        Parameters:
        pos1 (tuple): The (x, y) position of the first object.
        pos2 (tuple): The (x, y) position of the second object.
        
        Returns:
        int: The Manhattan distance between pos1 and pos2.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def step(self, action):
        print(f"Original action: {action}, type: {type(action)}")
        if isinstance(action, np.ndarray):
            action = action.item()
        print(f"Converted action: {action}, type: {type(action)}")
    
        ball_pos = self.ball.cur_pos
        agent_pos = self.agent_pos
        
        if action == 4:  # Kick action
            # Kick the ball towards the goal if adjacent
            if self._distance(agent_pos, ball_pos) == 1:
                self._move_ball_towards_goal()
       
        # Call the parent class's step method for other actions
        obs, reward, terminated, truncated, info = super().step(action)
        
        if np.array_equal(self.ball.cur_pos, self.goal.cur_pos):  # Ensure proper comparison
            reward = 1
            terminated = True
            
        # Ensure terminated and truncated are scalar boolean values
        if isinstance(terminated, np.ndarray):
            terminated = terminated.any()  # At least one environment is terminated
        if isinstance(truncated, np.ndarray):
            truncated = truncated.any()  # At least one environment is truncated

        if np.array_equal(agent_pos, ball_pos):
            print(f"Agent touched the ball at {agent_pos}. Grid state: {self.grid.get(*ball_pos)}")

        return obs, reward, terminated, truncated, info
    
    def _move_ball_towards_goal(self):
        # Move ball one step closer to the goal
        goal_pos = self.goal.cur_pos
        dx = np.sign(goal_pos[0] - self.ball.cur_pos[0])
        dy = np.sign(goal_pos[1] - self.ball.cur_pos[1])
        new_pos = (self.ball.cur_pos[0] + dx, self.ball.cur_pos[1] + dy)
        if self.grid.get(*new_pos) is None:  # Ensure no obstacles
            self.grid.set(*self.ball.cur_pos, None)  # Clear previous position
            self.grid.set(*new_pos, self.ball)
            self.ball.cur_pos = new_pos