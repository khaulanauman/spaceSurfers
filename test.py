import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.image as mpimg
from heapq import heappop, heappush

# Constants
WIDTH, HEIGHT = 15, 15
DIRECTIONAL_WEIGHTS = {'N': 1.2, 'S': 1.0, 'E': 1.5, 'W': 1.3}  # Different weights for each direction
DIRECTIONS = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}

# Load background image
background_image = mpimg.imread('b07.jpg')

# Initialize maze grid with walls (1 represents a wall, 0 represents a path)
maze = np.ones((2 * WIDTH + 1, 2 * HEIGHT + 1), dtype=int)


def carve(x, y):
    """Recursive function to carve out a maze."""
    maze[2 * x + 1, 2 * y + 1] = 0  # Mark current cell as path
    directions = list(DIRECTIONS.items())
    random.shuffle(directions)  # Randomize directions

    for _, (dx, dy) in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and maze[2 * nx + 1, 2 * ny + 1] == 1:
            # Carve wall between current cell and next cell
            maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
            carve(nx, ny)


# Start carving from the top-left corner of the grid
carve(0, 0)

# Define start and end points
start = (1, 1)
end = (2 * WIDTH - 1, 2 * HEIGHT - 1)
maze[start] = 0
maze[end] = 0


def heuristic(a, b):
    """Heuristic function for A* (Manhattan distance)"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(maze, start, end):
    """A* algorithm to find the shortest path in a weighted maze."""
    open_set = []
    heappush(open_set, (0, start))  # Priority queue of (cost, position)

    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        current_cost, current = heappop(open_set)

        if current == end:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for direction, (dx, dy) in DIRECTIONS.items():
            # New cell position
            nx, ny = current[0] + 2 * dx, current[1] + 2 * dy  # Move two cells to cross walls
            mx, my = current[0] + dx, current[1] + dy  # Middle cell (potential wall position)
            new_position = (nx, ny)

            if (0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and
                    maze[nx, ny] == 0 and maze[mx, my] == 0):

                # Calculate new cost with directional weight
                new_cost = cost_so_far[current] + DIRECTIONAL_WEIGHTS[direction]

                if new_position not in cost_so_far or new_cost < cost_so_far[new_position]:
                    cost_so_far[new_position] = new_cost
                    priority = new_cost + heuristic(new_position, end)
                    heappush(open_set, (priority, new_position))
                    came_from[new_position] = current

    return None  # Return None if no path is found


# Find the shortest path
path = a_star(maze, start, end)

# Define a colormap for the maze
cmap = ListedColormap(['black', 'darkgrey'])

# Create and display the figure with the background and the maze
fig, ax = plt.subplots(figsize=(8, 6))
fig.set_facecolor("black")
ax.imshow(background_image, extent=[0, maze.shape[1], 0, maze.shape[0]])  # Background image
ax.imshow(maze, cmap=cmap, alpha=0.5)  # Maze overlay with transparency

# Draw the shortest path if it exists
if path:
    path_x, path_y = zip(*path)
    ax.plot(path_y, path_x, color="cyan", linewidth=2, label="Drone Path")  # Draw path

# Mark start and end points
ax.plot(start[1], start[0], "go", label="Start")  # Start point in green
ax.plot(end[1], end[0], "ro", label="End")  # End point in red

ax.axis('off')  # Hide axes
plt.legend(loc="upper left")
plt.show()
