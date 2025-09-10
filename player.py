import pygame
from maze import CELL_SIZE, GRID_SIZE

# Colors
BLUE = (0, 0, 200)
RED = (200, 0, 0)

class Player:
    def __init__(self, start=(0, 0)):
        self.x, self.y = start
        self.start_pos = start

    def can_move(self, direction, maze):
        """Check if player can move in given direction"""
        current_cell = maze.grid[self.y][self.x]
        
        # Check if there's a wall in the direction we want to move
        if current_cell[direction]:
            return False
        
        # Calculate new position
        new_x, new_y = self.x, self.y
        if direction == "right":
            new_x += 1
        elif direction == "left":
            new_x -= 1
        elif direction == "bottom":
            new_y += 1
        elif direction == "top":
            new_y -= 1
        
        # Check bounds
        if not (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE):
            return False
            
        return True

    def move(self, dx, dy, maze):
        """Move player based on direction deltas"""
        direction = None
        
        # Convert dx, dy to direction string
        if dx == 1 and dy == 0:
            direction = "right"
        elif dx == -1 and dy == 0:
            direction = "left"
        elif dx == 0 and dy == 1:
            direction = "bottom"
        elif dx == 0 and dy == -1:
            direction = "top"
        
        if direction and self.can_move(direction, maze):
            self.x += dx
            self.y += dy
            return True
        return False

    def move_up(self, maze):
        """Move player up"""
        return self.move(0, -1, maze)
    
    def move_down(self, maze):
        """Move player down"""
        return self.move(0, 1, maze)
    
    def move_left(self, maze):
        """Move player left"""
        return self.move(-1, 0, maze)
    
    def move_right(self, maze):
        """Move player right"""
        return self.move(1, 0, maze)

    def reset_position(self):
        """Reset player to starting position"""
        self.x, self.y = self.start_pos

    def is_at_goal(self, maze):
        """Check if player reached the goal"""
        return (self.x, self.y) == maze.goal

    def get_position(self):
        """Get current position as tuple"""
        return (self.x, self.y)

    def set_position(self, x, y):
        """Set player position directly"""
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            self.x, self.y = x, y

    def draw(self, screen):
        """Draw player on screen"""
        # Draw player as a circle instead of rectangle for better visibility
        center_x = self.x * CELL_SIZE + CELL_SIZE // 2
        center_y = self.y * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 3
        
        pygame.draw.circle(screen, BLUE, (center_x, center_y), radius)
        
        # Add a small border for better visibility
        pygame.draw.circle(screen, (0, 0, 150), (center_x, center_y), radius, 2)

    def get_possible_moves(self, maze):
        """Get list of possible moves from current position (useful for AI)"""
        possible_moves = []
        directions = [("up", 0, -1), ("down", 0, 1), ("left", -1, 0), ("right", 1, 0)]
        
        for direction_name, dx, dy in directions:
            if self.can_move(direction_name.replace("up", "top").replace("down", "bottom"), maze):
                possible_moves.append((direction_name, dx, dy))
        
        return possible_moves