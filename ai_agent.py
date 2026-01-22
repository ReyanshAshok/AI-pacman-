import numpy as np
import random
from collections import deque, defaultdict
from maze import GRID_SIZE
import time
import heapq
import math

class PredictiveGhost:
    def __init__(self, start_position, ghost_id=0):
        self.x, self.y = start_position
        self.ghost_id = ghost_id
        self.last_positions = deque(maxlen=5)
        self.stuck_counter = 0
        self.chase_mode = "hunt"  # hunt, ambush, patrol
        self.mode_timer = 0
        self.predicted_intercept = None
        
    def get_position(self):
        return (self.x, self.y)
    
    def move_to(self, x, y):
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            self.last_positions.append((self.x, self.y))
            self.x, self.y = x, y
            return True
        return False
    
    def is_stuck(self):
        if len(self.last_positions) >= 3:
            return len(set(self.last_positions)) <= 2
        return False

class GreedyPredictiveAgent:
    def __init__(self, prediction_steps=5):
        # Prediction parameters
        self.prediction_steps = prediction_steps
        self.player_history = deque(maxlen=20)
        self.player_preferences = defaultdict(int)
        
        # Ghost management
        self.ghosts = []
        self.reinforcement_trigger_time = None  # When player first reached midpoint
        self.first_reinforcement_spawned = False  # First reinforcement ghost
        self.second_reinforcement_spawned = False  # Second reinforcement ghost
        self.initial_ghost_count = 2  # Remember initial number of ghosts
        self.catch_radius = 1.0
        self.player_caught = False
        
        # A* pathfinding cache
        self.path_cache = {}
        self.cache_timeout = {}
        
        # Player behavior analysis
        self.directional_bias = defaultdict(int)
        self.position_choices = defaultdict(lambda: defaultdict(int))
        self.last_update_time = time.time()
        
        # Maze progression tracking
        self.maze_midpoint = GRID_SIZE // 2

    def initialize_ghosts(self, maze, num_ghosts=2):
        """Initialize ghosts at strategic positions"""
        self.ghosts = []
        self.initial_ghost_count = num_ghosts
        self.reinforcement_trigger_time = None
        self.first_reinforcement_spawned = False
        self.second_reinforcement_spawned = False
        
        # Find good spawn positions away from player start and goal
        spawn_candidates = []
        
        # Add positions that are roughly 1/3 and 2/3 through the maze
        third_x = GRID_SIZE // 3
        two_third_x = 2 * GRID_SIZE // 3
        mid_y = GRID_SIZE // 2
        
        potential_spawns = [
            (third_x, mid_y), (two_third_x, mid_y),
            (mid_y, third_x), (mid_y, two_third_x),
            (GRID_SIZE-3, 2), (2, GRID_SIZE-3),
            (GRID_SIZE//2, 2), (2, GRID_SIZE//2)
        ]
        
        # Filter valid spawn positions
        for pos in potential_spawns:
            x, y = pos
            if (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and 
                pos != maze.start and pos != maze.goal and
                self.is_position_accessible(x, y, maze)):
                spawn_candidates.append(pos)
        
        # Add some random positions if not enough candidates
        while len(spawn_candidates) < num_ghosts * 2:
            x = random.randint(1, GRID_SIZE-2)
            y = random.randint(1, GRID_SIZE-2)
            if self.is_position_accessible(x, y, maze):
                spawn_candidates.append((x, y))
        
        # Create initial ghosts
        selected_spawns = random.sample(spawn_candidates, min(num_ghosts, len(spawn_candidates)))
        for i, spawn_pos in enumerate(selected_spawns):
            ghost = PredictiveGhost(spawn_pos, i)
            self.ghosts.append(ghost)

    def check_for_reinforcement_spawn(self, player_pos, maze):
        """Check player progress and manage timed reinforcement spawning"""
        px, py = player_pos
        current_time = time.time()
        
        # Check if player has crossed the midpoint for the first time
        progress_x = px >= self.maze_midpoint
        progress_y = py >= self.maze_midpoint
        
        if ((progress_x and py >= GRID_SIZE // 3) or (progress_y and px >= GRID_SIZE // 3)):
            # Player has reached the trigger zone
            if self.reinforcement_trigger_time is None:
                self.reinforcement_trigger_time = current_time
                print("🎯 Player entered the danger zone! Reinforcements incoming...")
                return False
        
        # If trigger time is set, check for timed spawns
        if self.reinforcement_trigger_time is not None:
            time_since_trigger = current_time - self.reinforcement_trigger_time
            
            # Spawn first reinforcement ghost immediately
            if not self.first_reinforcement_spawned:
                self.spawn_single_reinforcement_ghost(maze, 1)
                self.first_reinforcement_spawned = True
                return True
            
            # Spawn second reinforcement ghost after 30 seconds
            elif time_since_trigger >= 10.0 and not self.second_reinforcement_spawned:
                self.spawn_single_reinforcement_ghost(maze, 2)
                self.second_reinforcement_spawned = True
                return True
        
        return False

    def spawn_single_reinforcement_ghost(self, maze, ghost_number):
        """Spawn a single reinforcement ghost from the goal area"""
        print(f"🚨 REINFORCEMENT GHOST #{ghost_number} SPAWNED! 🚨")
        
        # Start from goal position and find nearby accessible positions
        goal_x, goal_y = maze.goal
        spawn_candidates = []
        
        # Search in expanding rings around the goal
        for radius in range(1, 5):  # Search up to 4 cells away
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        x, y = goal_x + dx, goal_y + dy
                        
                        if (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and 
                            (x, y) != maze.goal and (x, y) != maze.start and
                            self.is_position_accessible(x, y, maze)):
                            
                            # Check if position is not occupied by existing ghosts
                            position_free = True
                            for existing_ghost in self.ghosts:
                                if existing_ghost.get_position() == (x, y):
                                    position_free = False
                                    break
                            
                            if position_free:
                                spawn_candidates.append((x, y))
            
            # If we found candidates at this radius, use them
            if spawn_candidates:
                break
        
        # If no good positions found near goal, fall back to random positions
        if not spawn_candidates:
            for _ in range(20):  # Try 20 random positions
                x = random.randint(self.maze_midpoint, GRID_SIZE - 1)
                y = random.randint(self.maze_midpoint, GRID_SIZE - 1)
                
                if (self.is_position_accessible(x, y, maze) and 
                    (x, y) != maze.goal and (x, y) not in [g.get_position() for g in self.ghosts]):
                    spawn_candidates.append((x, y))
        
        # Spawn the ghost
        if spawn_candidates:
            spawn_pos = random.choice(spawn_candidates)
            ghost_id = len(self.ghosts)
            reinforcement_ghost = PredictiveGhost(spawn_pos, ghost_id)
            reinforcement_ghost.chase_mode = "hunt"  # Start in aggressive hunt mode
            self.ghosts.append(reinforcement_ghost)
            
            print(f"✅ Reinforcement ghost #{ghost_number} deployed at: {spawn_pos}")
        else:
            print("⚠️ Could not find suitable spawn position for reinforcement ghost!")

    def is_position_accessible(self, x, y, maze):
        """Check if position has at least one open direction"""
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return False
        
        directions = ["top", "right", "bottom", "left"]
        for direction in directions:
            if not maze.grid[y][x][direction]:
                return True
        return False
        """Check if position has at least one open direction"""
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return False
        
        directions = ["top", "right", "bottom", "left"]
        for direction in directions:
            if not maze.grid[y][x][direction]:
                return True
        return False

    def analyze_player_movement(self, player_pos, move_direction):
        """Analyze player movement for prediction"""
        self.player_history.append((player_pos, move_direction, time.time()))
        
        # Track directional preferences
        if move_direction:
            self.player_preferences[move_direction] += 1
            self.directional_bias[move_direction] += 1
        
        # Track position-based choices
        if len(self.player_history) >= 2:
            prev_pos, prev_dir, _ = self.player_history[-2]
            if move_direction:
                self.position_choices[prev_pos][move_direction] += 1

    def a_star_pathfind(self, start, goal, maze):
        """A* pathfinding algorithm"""
        # Check cache first
        cache_key = (start, goal)
        current_time = time.time()
        
        if (cache_key in self.path_cache and 
            current_time - self.cache_timeout.get(cache_key, 0) < 2.0):
            return self.path_cache[cache_key]
        
        def heuristic(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            directions = [(0, -1, "top"), (1, 0, "right"), (0, 1, "bottom"), (-1, 0, "left")]
            
            for dx, dy, wall_dir in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 
                    not maze.grid[y][x][wall_dir]):
                    neighbors.append((nx, ny))
            return neighbors
        
        # A* algorithm
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                
                # Cache the result
                self.path_cache[cache_key] = path
                self.cache_timeout[cache_key] = current_time
                return path
            
            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return [start]  # No path found

    def predict_player_path(self, player_pos, goal_pos, maze):
        """Predict player's optimal path using A* with behavioral bias"""
        # Get optimal path using A*
        optimal_path = self.a_star_pathfind(player_pos, goal_pos, maze)
        
        if len(optimal_path) <= 1:
            return [player_pos]
        
        # Apply behavioral bias to prediction
        predicted_path = [player_pos]
        current_pos = player_pos
        
        # Look ahead up to prediction_steps
        for step in range(min(self.prediction_steps, len(optimal_path) - 1)):
            next_optimal = optimal_path[step + 1]
            
            # Get available moves from current position
            available_moves = self.get_available_moves(current_pos, maze)
            
            if not available_moves:
                break
            
            # Find the move that leads to optimal next position
            optimal_move = None
            for move in available_moves:
                dx, dy = self.direction_to_delta(move)
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if next_pos == next_optimal:
                    optimal_move = move
                    break
            
            # Apply behavioral bias
            if optimal_move:
                # Player likely to take optimal move, but consider preferences
                if len(self.player_preferences) > 0:
                    # Check if player has strong directional bias
                    total_moves = sum(self.player_preferences.values())
                    bias_strength = max(self.player_preferences.values()) / total_moves
                    
                    # If player has strong bias and there's an alternative
                    if bias_strength > 0.6:
                        preferred_dir = max(self.player_preferences, key=self.player_preferences.get)
                        if preferred_dir in available_moves and random.random() < 0.3:
                            # Sometimes player follows bias instead of optimal
                            dx, dy = self.direction_to_delta(preferred_dir)
                            biased_pos = (current_pos[0] + dx, current_pos[1] + dy)
                            predicted_path.append(biased_pos)
                            current_pos = biased_pos
                            continue
                
                predicted_path.append(next_optimal)
                current_pos = next_optimal
            else:
                # If no optimal move available, use greedy approach
                best_move = self.get_greedy_move(current_pos, goal_pos, available_moves)
                if best_move:
                    dx, dy = self.direction_to_delta(best_move)
                    next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                    predicted_path.append(next_pos)
                    current_pos = next_pos
                else:
                    break
        
        return predicted_path

    def get_greedy_move(self, current_pos, goal_pos, available_moves):
        """Get move that reduces distance to goal most"""
        best_move = None
        min_distance = float('inf')
        
        for move in available_moves:
            dx, dy = self.direction_to_delta(move)
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            distance = abs(next_pos[0] - goal_pos[0]) + abs(next_pos[1] - goal_pos[1])
            
            if distance < min_distance:
                min_distance = distance
                best_move = move
        
        return best_move

    def get_available_moves(self, position, maze):
        """Get available moves from a position"""
        x, y = position
        available_moves = []
        
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            directions = [("UP", "top"), ("DOWN", "bottom"), ("LEFT", "left"), ("RIGHT", "right")]
            
            for move, wall_dir in directions:
                if not maze.grid[y][x][wall_dir]:
                    available_moves.append(move)
        
        return available_moves

    def direction_to_delta(self, direction):
        """Convert direction to coordinate delta"""
        direction_map = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }
        return direction_map.get(direction, (0, 0))

    def calculate_intercept_positions(self, player_pos, goal_pos, maze):
        """Calculate where ghosts should move to intercept player"""
        predicted_path = self.predict_player_path(player_pos, goal_pos, maze)
        
        intercept_positions = []
        
        for i, ghost in enumerate(self.ghosts):
            ghost_pos = ghost.get_position()
            best_intercept = None
            min_intercept_time = float('inf')
            
            # Check each predicted position as potential intercept point
            for step, predicted_pos in enumerate(predicted_path):
                # Calculate time for ghost to reach this position
                ghost_path = self.a_star_pathfind(ghost_pos, predicted_pos, maze)
                ghost_time = len(ghost_path) - 1
                
                # Player will be at this position at time 'step'
                player_time = step
                
                # Ghost should arrive slightly before or at the same time
                if ghost_time <= player_time + 1:  # +1 for tolerance
                    intercept_time = max(ghost_time, player_time)
                    if intercept_time < min_intercept_time:
                        min_intercept_time = intercept_time
                        best_intercept = predicted_pos
            
            # If no good intercept found, go for closest predicted position
            if best_intercept is None and predicted_path:
                min_dist = float('inf')
                for pred_pos in predicted_path[:3]:  # Look at first 3 predicted positions
                    dist = abs(ghost_pos[0] - pred_pos[0]) + abs(ghost_pos[1] - pred_pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_intercept = pred_pos
            
            intercept_positions.append(best_intercept or predicted_path[0] if predicted_path else player_pos)
            ghost.predicted_intercept = best_intercept
        
        return intercept_positions

    def update_ghosts(self, player_pos, maze):
        """Update ghost positions using intercept strategy"""
        current_time = time.time()
        
        if current_time - self.last_update_time < 0.4:  # Update every 0.4 seconds
            return
        
        # Check for reinforcement spawning (non-blocking)
        self.check_for_reinforcement_spawn(player_pos, maze)
        
        # Continue with normal ghost updates
        goal_pos = maze.goal
        
        # Only calculate intercepts if we have ghosts
        if self.ghosts:
            try:
                intercept_positions = self.calculate_intercept_positions(player_pos, goal_pos, maze)
                
                for i, ghost in enumerate(self.ghosts):
                    target_pos = intercept_positions[i] if i < len(intercept_positions) else player_pos
                    self.move_ghost_toward_target(ghost, target_pos, maze)
            except Exception as e:
                # If intercept calculation fails, fall back to simple chase
                print(f"Ghost update error: {e}")
                for ghost in self.ghosts:
                    self.move_ghost_toward_target(ghost, player_pos, maze)
        
        self.last_update_time = current_time

    def move_ghost_toward_target(self, ghost, target_pos, maze):
        """Move ghost one step toward target using A*"""
        ghost_pos = ghost.get_position()
        
        if ghost_pos == target_pos:
            return
        
        # Get path to target
        path = self.a_star_pathfind(ghost_pos, target_pos, maze)
        
        if len(path) > 1:
            next_pos = path[1]  # First step in path
            ghost.move_to(next_pos[0], next_pos[1])
        else:
            # If no path, try random movement
            available_moves = self.get_available_moves(ghost_pos, maze)
            if available_moves:
                move = random.choice(available_moves)
                dx, dy = self.direction_to_delta(move)
                new_pos = (ghost_pos[0] + dx, ghost_pos[1] + dy)
                if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE:
                    ghost.move_to(new_pos[0], new_pos[1])

    def check_player_caught(self, player_pos):
        """Check if any ghost caught the player"""
        for ghost in self.ghosts:
            ghost_pos = ghost.get_position()
            distance = math.sqrt((player_pos[0] - ghost_pos[0])**2 + 
                               (player_pos[1] - ghost_pos[1])**2)
            
            if distance < self.catch_radius:
                self.player_caught = True
                return True
        
        return False

    def reset_game_state(self):
        """Reset for new game"""
        self.player_caught = False
        self.ghosts = []
        self.path_cache.clear()
        self.cache_timeout.clear()
        # Reset all reinforcement tracking
        self.reinforcement_trigger_time = None
        self.first_reinforcement_spawned = False
        self.second_reinforcement_spawned = False

    def get_prediction_info(self, player_pos, maze):
        """Get current prediction information for debugging/display"""
        if not hasattr(maze, 'goal'):
            return {
                "predicted_path": [player_pos],
                "ghost_targets": [],
                "path_length_to_goal": 0,
                "player_preferences": dict(self.player_preferences)
            }
        
        predicted_path = self.predict_player_path(player_pos, maze.goal, maze)
        
        info = {
            "predicted_path": predicted_path,
            "ghost_targets": [getattr(ghost, 'predicted_intercept', None) for ghost in self.ghosts],
            "path_length_to_goal": len(self.a_star_pathfind(player_pos, maze.goal, maze)) - 1,
            "player_preferences": dict(self.player_preferences)
        }
        
        return info

    def get_learning_stats(self):
        """Get statistics about the AI's performance"""
        reinforcement_status = "None"
        if self.reinforcement_trigger_time is not None:
            if self.second_reinforcement_spawned:
                reinforcement_status = "Both Spawned"
            elif self.first_reinforcement_spawned:
                current_time = time.time()
                time_left = max(0, 30.0 - (current_time - self.reinforcement_trigger_time))
                reinforcement_status = f"Second in {time_left:.1f}s"
            else:
                reinforcement_status = "Triggered"
        
        return {
            "active_ghosts": len(self.ghosts),
            "initial_ghosts": self.initial_ghost_count,
            "reinforcement_status": reinforcement_status,
            "patterns_learned": len(self.position_choices),
            "directional_preferences": dict(self.directional_bias),
            "cache_size": len(self.path_cache),
            "player_caught": self.player_caught
        }