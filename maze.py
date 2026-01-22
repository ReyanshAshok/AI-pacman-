import pygame
import random
from collections import deque

# Configuration
GRID_SIZE = 20
CELL_SIZE = 30
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)

class Maze:
    def __init__(self, complexity_factor=0.3):
        # Initialize grid with all walls
        self.grid = [[{"top": True, "right": True, "bottom": True, "left": True, "visited": False} 
                     for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]
        
        # Start and goal positions
        self.start = (0, 0)
        self.goal = (GRID_SIZE - 1, GRID_SIZE - 1)
        
        # Generate maze using recursive backtracking with modifications
        self.generate_complex_maze(complexity_factor)
        
        # Ensure there's always a path from start to goal
        self.ensure_solution_path()
        
        # Add additional false paths and dead ends
        self.add_false_paths()
        
        # Add some random openings to create alternative routes
        self.add_alternative_routes()

    def get_neighbors(self, x, y):
        """Get valid neighboring cells"""
        neighbors = []
        directions = [("top", 0, -1), ("right", 1, 0), ("bottom", 0, 1), ("left", -1, 0)]
        
        for direction, dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                neighbors.append((nx, ny, direction))
        return neighbors

    def remove_wall(self, x, y, direction):
        """Remove wall between two cells"""
        self.grid[y][x][direction] = False
        
        # Remove corresponding wall in adjacent cell
        if direction == "top" and y > 0:
            self.grid[y-1][x]["bottom"] = False
        elif direction == "bottom" and y < GRID_SIZE - 1:
            self.grid[y+1][x]["top"] = False
        elif direction == "left" and x > 0:
            self.grid[y][x-1]["right"] = False
        elif direction == "right" and x < GRID_SIZE - 1:
            self.grid[y][x+1]["left"] = False

    def add_wall(self, x, y, direction):
        """Add wall between two cells"""
        self.grid[y][x][direction] = True
        
        # Add corresponding wall in adjacent cell
        if direction == "top" and y > 0:
            self.grid[y-1][x]["bottom"] = True
        elif direction == "bottom" and y < GRID_SIZE - 1:
            self.grid[y+1][x]["top"] = True
        elif direction == "left" and x > 0:
            self.grid[y][x-1]["right"] = True
        elif direction == "right" and x < GRID_SIZE - 1:
            self.grid[y][x+1]["left"] = True

    def generate_complex_maze(self, complexity_factor):
        """Generate maze using modified recursive backtracking"""
        # Reset visited flags
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                self.grid[y][x]["visited"] = False
        
        stack = [(0, 0)]
        self.grid[0][0]["visited"] = True
        
        while stack:
            current_x, current_y = stack[-1]
            
            # Get unvisited neighbors
            unvisited_neighbors = []
            for nx, ny, direction in self.get_neighbors(current_x, current_y):
                if not self.grid[ny][nx]["visited"]:
                    unvisited_neighbors.append((nx, ny, direction))
            
            if unvisited_neighbors:
                # Choose random neighbor
                next_x, next_y, direction = random.choice(unvisited_neighbors)
                
                # Remove wall and mark as visited
                self.remove_wall(current_x, current_y, direction)
                self.grid[next_y][next_x]["visited"] = True
                
                stack.append((next_x, next_y))
                
                # Sometimes create loops by removing additional walls (complexity factor)
                if random.random() < complexity_factor:
                    additional_neighbors = self.get_neighbors(next_x, next_y)
                    for ax, ay, a_direction in additional_neighbors:
                        if self.grid[ay][ax]["visited"] and random.random() < 0.3:
                            self.remove_wall(next_x, next_y, a_direction)
            else:
                stack.pop()

    def ensure_solution_path(self):
        """Ensure there's always a path from start to goal using A* pathfinding"""
        if not self.has_path(self.start, self.goal):
            # Create a direct path if none exists
            path = self.create_direct_path(self.start, self.goal)
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                
                # Determine direction and remove wall
                if x2 > x1:
                    self.remove_wall(x1, y1, "right")
                elif x2 < x1:
                    self.remove_wall(x1, y1, "left")
                elif y2 > y1:
                    self.remove_wall(x1, y1, "bottom")
                elif y2 < y1:
                    self.remove_wall(x1, y1, "top")

    def has_path(self, start, goal):
        """Check if path exists using BFS"""
        queue = deque([start])
        visited = set([start])
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) == goal:
                return True
            
            # Check all four directions
            directions = [("right", 1, 0), ("left", -1, 0), ("bottom", 0, 1), ("top", 0, -1)]
            for direction, dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check if we can move in this direction
                if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 
                    (nx, ny) not in visited and not self.grid[y][x][direction]):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return False

    def create_direct_path(self, start, goal):
        """Create a simple path from start to goal"""
        path = [start]
        x, y = start
        gx, gy = goal
        
        # Move horizontally first
        while x != gx:
            if x < gx:
                x += 1
            else:
                x -= 1
            path.append((x, y))
        
        # Then move vertically
        while y != gy:
            if y < gy:
                y += 1
            else:
                y -= 1
            path.append((x, y))
        
        return path

    def add_false_paths(self):
        """Add dead ends and false paths throughout the maze"""
        num_false_paths = GRID_SIZE // 2
        
        for _ in range(num_false_paths):
            # Pick random starting point
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            
            # Create a short dead end path
            path_length = random.randint(2, 5)
            current_x, current_y = x, y
            
            for _ in range(path_length):
                neighbors = self.get_neighbors(current_x, current_y)
                valid_neighbors = []
                
                for nx, ny, direction in neighbors:
                    # Only add wall removal if it creates a dead end
                    if self.count_open_walls(nx, ny) <= 1:
                        valid_neighbors.append((nx, ny, direction))
                
                if valid_neighbors:
                    next_x, next_y, direction = random.choice(valid_neighbors)
                    self.remove_wall(current_x, current_y, direction)
                    current_x, current_y = next_x, next_y
                else:
                    break

    def count_open_walls(self, x, y):
        """Count how many walls are open for a cell"""
        count = 0
        for direction in ["top", "right", "bottom", "left"]:
            if not self.grid[y][x][direction]:
                count += 1
        return count

    def add_alternative_routes(self):
        """Add some alternative routes to make maze more complex"""
        num_alternatives = GRID_SIZE // 3
        
        for _ in range(num_alternatives):
            x = random.randint(1, GRID_SIZE - 2)
            y = random.randint(1, GRID_SIZE - 2)
            
            # Randomly remove one wall
            directions = ["top", "right", "bottom", "left"]
            direction = random.choice(directions)
            
            # Only remove wall if it doesn't make the maze too easy
            if random.random() < 0.4:
                self.remove_wall(x, y, direction)

    def get_maze_data(self):
        """Return maze data for ML processing"""
        # Convert maze to numerical format for ML
        maze_array = []
        for y in range(GRID_SIZE):
            row = []
            for x in range(GRID_SIZE):
                cell_value = 0
                if self.grid[y][x]["top"]: cell_value += 1
                if self.grid[y][x]["right"]: cell_value += 2
                if self.grid[y][x]["bottom"]: cell_value += 4
                if self.grid[y][x]["left"]: cell_value += 8
                row.append(cell_value)
            maze_array.append(row)
        return maze_array

    def update_maze_from_data(self, maze_array):
        """Update maze from ML-generated data"""
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                cell_value = maze_array[y][x]
                self.grid[y][x]["top"] = bool(cell_value & 1)
                self.grid[y][x]["right"] = bool(cell_value & 2)
                self.grid[y][x]["bottom"] = bool(cell_value & 4)
                self.grid[y][x]["left"] = bool(cell_value & 8)

    def draw(self, screen):
        # Fill background
        screen.fill(WHITE)
        
        # Draw maze walls
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                cell = self.grid[y][x]
                px = x * CELL_SIZE
                py = y * CELL_SIZE

                # Draw walls
                if cell["top"]:
                    pygame.draw.line(screen, BLACK, (px, py), (px + CELL_SIZE, py), 2)
                if cell["right"]:
                    pygame.draw.line(screen, BLACK, (px + CELL_SIZE, py), (px + CELL_SIZE, py + CELL_SIZE), 2)
                if cell["bottom"]:
                    pygame.draw.line(screen, BLACK, (px, py + CELL_SIZE), (px + CELL_SIZE, py + CELL_SIZE), 2)
                if cell["left"]:
                    pygame.draw.line(screen, BLACK, (px, py), (px, py + CELL_SIZE), 2)

        # Draw start position (blue)
        sx, sy = self.start
        pygame.draw.rect(screen, BLUE, (sx * CELL_SIZE + 4, sy * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8))

        # Draw goal (green)
        gx, gy = self.goal
        pygame.draw.rect(screen, GREEN, (gx * CELL_SIZE + 4, gy * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8))