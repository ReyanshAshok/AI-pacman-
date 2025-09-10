import pygame
import sys
import time
from collections import deque

# Import your modules
from maze import Maze, WIDTH, HEIGHT, WHITE, BLACK, GREEN, RED, BLUE, CELL_SIZE,GRID_SIZE
from player import Player
from ai_agent import GreedyPredictiveAgent

# Ghost colors
GHOST_COLORS = [(255, 100, 100), (255, 165, 0), (255, 192, 203), (128, 0, 128)]
GHOST_TRAIL_ALPHA = 100

class PredictiveMazeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH + 350, HEIGHT))
        pygame.display.set_caption("Greedy Predictive Ghost Maze - A* + Behavioral Prediction")
        self.clock = pygame.time.Clock()
        
        # Initialize fonts
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 28)
        
        # Initialize game components
        self.maze = Maze(complexity_factor=0.3)
        self.player = Player(start=self.maze.start)
        self.ai_agent = GreedyPredictiveAgent(prediction_steps=5)
        
        # Initialize ghosts
        self.ai_agent.initialize_ghosts(self.maze, num_ghosts=2)
        
        # Game state
        self.game_won = False
        self.game_over = False
        self.moves_count = 0
        self.player_path = []
        self.ghost_trails = {i: deque(maxlen=8) for i in range(len(self.ai_agent.ghosts))}
        
        # Timing
        self.start_time = time.time()
        self.game_time = 0.0
        
        # UI state
        self.show_ai_info = True
        self.show_predictions = True
        self.show_trails = True
        
        # Statistics
        self.games_completed = 0
        self.total_catches = 0
        self.best_time = float('inf')

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if not self.game_over and not self.game_won:
                    # Player movement
                    moved = False
                    move_direction = None
                    
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        moved = self.player.move_up(self.maze)
                        move_direction = "UP"
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        moved = self.player.move_down(self.maze)
                        move_direction = "DOWN"
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        moved = self.player.move_left(self.maze)
                        move_direction = "LEFT"
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        moved = self.player.move_right(self.maze)
                        move_direction = "RIGHT"
                    
                    if moved:
                        self.moves_count += 1
                        current_pos = self.player.get_position()
                        self.player_path.append(current_pos)
                        
                        # Analyze player movement for prediction
                        self.ai_agent.analyze_player_movement(current_pos, move_direction)
                        
                        # Check if player reached goal
                        if self.player.is_at_goal(self.maze):
                            self.game_won = True
                            self.games_completed += 1
                            self.game_time = time.time() - self.start_time
                            if self.game_time < self.best_time:
                                self.best_time = self.game_time
                        
                        # Check if player was caught
                        if self.ai_agent.check_player_caught(current_pos):
                            self.game_over = True
                            self.total_catches += 1
                
                # Game controls (work anytime)
                if event.key == pygame.K_SPACE:
                    self.reset_game(new_maze=True)
                elif event.key == pygame.K_r:
                    self.reset_game(new_maze=False)
                elif event.key == pygame.K_i:
                    self.show_ai_info = not self.show_ai_info
                elif event.key == pygame.K_p:
                    self.show_predictions = not self.show_predictions
                elif event.key == pygame.K_t:
                    self.show_trails = not self.show_trails
                elif event.key == pygame.K_ESCAPE:
                    return False
        
        return True

    def update_game_state(self):
        """Update game state including ghost AI"""
        if not self.game_over and not self.game_won:
            # Update ghost positions using the greedy predictive algorithm
            self.ai_agent.update_ghosts(self.player.get_position(), self.maze)
            
            # Update ghost trails
            for i, ghost in enumerate(self.ai_agent.ghosts):
                if i < len(self.ghost_trails):
                    self.ghost_trails[i].append(ghost.get_position())
            
            # Check for catches after ghost movement
            if self.ai_agent.check_player_caught(self.player.get_position()):
                self.game_over = True
                self.total_catches += 1

    def reset_game(self, new_maze=True):
        """Reset game state"""
        if new_maze:
            # Gradually increase difficulty
            complexity = min(0.2 + (self.games_completed * 0.03), 0.5)
            self.maze = Maze(complexity_factor=complexity)
            
            # Add more ghosts as player gets better (max 3)
            num_ghosts = min(2 + (self.games_completed // 4), 3)
            self.ai_agent = GreedyPredictiveAgent(prediction_steps=5 + (self.games_completed // 2))
            self.ai_agent.initialize_ghosts(self.maze, num_ghosts)
        else:
            # Just reset positions, keep same maze and AI learning
            self.ai_agent.reset_game_state()
            self.ai_agent.initialize_ghosts(self.maze, len(self.ai_agent.ghosts) or 2)
        
        self.player = Player(start=self.maze.start)
        self.game_won = False
        self.game_over = False
        self.moves_count = 0
        self.player_path = []
        self.start_time = time.time()
        
        # Reset ghost trails
        self.ghost_trails = {i: deque(maxlen=8) for i in range(len(self.ai_agent.ghosts))}

    def draw_ghost_trails(self):
        """Draw trails behind ghosts"""
        if not self.show_trails:
            return
        
        for ghost_id, trail in self.ghost_trails.items():
            if len(trail) > 1:
                color = GHOST_COLORS[ghost_id % len(GHOST_COLORS)]
                
                for i, pos in enumerate(trail):
                    if i < len(trail) - 1:  # Don't draw trail on current position
                        x, y = pos
                        alpha = int((i + 1) / len(trail) * GHOST_TRAIL_ALPHA)
                        
                        # Create a surface for alpha blending
                        trail_surface = pygame.Surface((CELL_SIZE//3, CELL_SIZE//3), pygame.SRCALPHA)
                        trail_surface.fill((*color, alpha))
                        
                        px = x * CELL_SIZE + CELL_SIZE//3
                        py = y * CELL_SIZE + CELL_SIZE//3
                        self.screen.blit(trail_surface, (px, py))

    def draw_ghosts(self):
        """Draw all ghosts with different colors"""
        for i, ghost in enumerate(self.ai_agent.ghosts):
            x, y = ghost.get_position()
            color = GHOST_COLORS[i % len(GHOST_COLORS)]
            
            # Draw ghost as a circle
            center_x = x * CELL_SIZE + CELL_SIZE // 2
            center_y = y * CELL_SIZE + CELL_SIZE // 2
            radius = CELL_SIZE // 3
            
            # Draw main ghost body
            pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
            pygame.draw.circle(self.screen, BLACK, (center_x, center_y), radius, 2)
            
            # Draw ghost ID number
            id_text = self.small_font.render(str(i+1), True, WHITE)
            text_rect = id_text.get_rect(center=(center_x, center_y))
            self.screen.blit(id_text, text_rect)
            
            # Draw eyes
            eye_offset = radius // 3
            pygame.draw.circle(self.screen, WHITE, 
                             (center_x - eye_offset//2, center_y - eye_offset), 3)
            pygame.draw.circle(self.screen, WHITE, 
                             (center_x + eye_offset//2, center_y - eye_offset), 3)
            pygame.draw.circle(self.screen, BLACK, 
                             (center_x - eye_offset//2, center_y - eye_offset), 1)
            pygame.draw.circle(self.screen, BLACK, 
                             (center_x + eye_offset//2, center_y - eye_offset), 1)

    def draw_predictions(self):
        """Draw predicted player path and ghost intercept points"""
        if not self.show_predictions:
            return
        
        try:
            player_pos = self.player.get_position()
            prediction_info = self.ai_agent.get_prediction_info(player_pos, self.maze)
            
            # Draw predicted player path
            predicted_path = prediction_info.get("predicted_path", [])
            if len(predicted_path) > 1:
                for i, pos in enumerate(predicted_path[1:], 1):  # Skip current position
                    if (isinstance(pos, tuple) and len(pos) == 3 and 
                        0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE):
                        x, y = pos
                        center_x = x * CELL_SIZE + CELL_SIZE // 2
                        center_y = y * CELL_SIZE + CELL_SIZE // 2
                        
                        # Draw prediction with decreasing opacity
                        alpha = max(50, 200 - (i * 25))
                        prediction_surface = pygame.Surface((CELL_SIZE//2, CELL_SIZE//2), pygame.SRCALPHA)
                        prediction_surface.fill((255, 255, 0, alpha))  # Yellow with alpha
                        
                        px = x * CELL_SIZE + CELL_SIZE//4
                        py = y * CELL_SIZE + CELL_SIZE//4
                        self.screen.blit(prediction_surface, (px, py))
                        
                        # Draw step number
                        if i <= 5:  # Only show numbers for first 5 steps
                            step_text = self.small_font.render(str(i), True, BLACK)
                            self.screen.blit(step_text, (center_x-4, center_y-6))
            
            # Draw ghost intercept targets
            ghost_targets = prediction_info.get("ghost_targets", [])
            for i, target in enumerate(ghost_targets):
                if (target and isinstance(target, tuple) and len(target) == 2 and
                    0 <= target[0] < GRID_SIZE and 0 <= target[1] < GRID_SIZE):
                    x, y = target
                    center_x = x * CELL_SIZE + CELL_SIZE // 2
                    center_y = y * CELL_SIZE + CELL_SIZE // 2
                    
                    # Draw intercept target
                    color = GHOST_COLORS[i % len(GHOST_COLORS)]
                    # pygame.draw.circle(self.screen, color, (center_x, center_y), 10, 3)
                    
                    # Draw target symbol
                    # pygame.draw.line(self.screen, color, 
                                #    (center_x-6, center_y), (center_x+6, center_y), 2)
                    # pygame.draw.line(self.screen, color, 
                                #    (center_x, center_y-6), (center_x, center_y+6), 2)
                    
        except Exception as e:
            print(f"Prediction drawing error: {e}")

    def draw_ui(self):
        """Draw game UI and AI information"""
        ui_x = WIDTH + 10
        y_offset = 10
        
        # Game title
        title = self.title_font.render("Predictive Ghost Maze", True, BLACK)
        self.screen.blit(title, (ui_x, y_offset))
        y_offset += 35
        
        # Game stats
        current_time = time.time() - self.start_time if not (self.game_won or self.game_over) else self.game_time
        
        texts = [
            f"Time: {current_time:.1f}s",
            f"Moves: {self.moves_count}",
            f"Games Won: {self.games_completed}",
            f"Times Caught: {self.total_catches}",
            f"Best Time: {self.best_time:.1f}s" if self.best_time != float('inf') else "Best Time: --",
            f"Position: {self.player.get_position()}",
            ""
        ]
        
        if self.game_won:
            texts.insert(0, "🎉 ESCAPED! 🎉")
            texts.insert(1, f"⏱️ Time: {self.game_time:.1f}s")
        elif self.game_over:
            texts.insert(0, "👻 CAUGHT! 👻")
        
        for text in texts:
            if text:
                color = GREEN if "ESCAPED" in text else RED if "CAUGHT" in text else BLACK
                rendered = self.font.render(text, True, color)
                self.screen.blit(rendered, (ui_x, y_offset))
            y_offset += 25
        
        # Ghost information
        if self.ai_agent.ghosts:
            ghost_title = self.font.render("Ghost Status:", True, BLACK)
            self.screen.blit(ghost_title, (ui_x, y_offset))
            y_offset += 25
            
            for i, ghost in enumerate(self.ai_agent.ghosts):
                ghost_pos = ghost.get_position()
                player_pos = self.player.get_position()
                distance = abs(player_pos[0] - ghost_pos[0]) + abs(player_pos[1] - ghost_pos[1])
                
                target = getattr(ghost, 'predicted_intercept', None)
                target_str = f"{target}" if target else "None"
                
                ghost_info = [
                    f"Ghost {i+1}: {ghost_pos}",
                    f"  Distance: {distance}",
                    f"  Target: {target_str}",
                    ""
                ]
                
                color = GHOST_COLORS[i % len(GHOST_COLORS)]
                for info in ghost_info:
                    if info:
                        text_color = color if "Ghost" in info else BLACK
                        rendered = self.small_font.render(info, True, text_color)
                        self.screen.blit(rendered, (ui_x, y_offset))
                    y_offset += 18
        
        # AI Prediction Information
        if self.show_ai_info:
            y_offset += 10
            ai_title = self.font.render("AI Algorithm Info:", True, BLACK)
            self.screen.blit(ai_title, (ui_x, y_offset))
            y_offset += 25
            
            try:
                prediction_info = self.ai_agent.get_prediction_info(self.player.get_position(), self.maze)
                stats = self.ai_agent.get_learning_stats()
                
                ai_info = [
                    f"Algorithm: A* + Greedy",
                    f"Prediction Steps: {len(prediction_info.get('predicted_path', []))}",
                    f"Steps to Goal: {prediction_info.get('path_length_to_goal', 0)}",
                    f"Active Ghosts: {stats['active_ghosts']}",
                    f"Behavior Patterns: {stats['patterns_learned']}",
                    ""
                ]
                
                # Show player preferences if available
                preferences = prediction_info.get("player_preferences", {})
                if preferences:
                    most_used = max(preferences, key=preferences.get)
                    total = sum(preferences.values())
                    percentage = (preferences[most_used] / total) * 100 if total > 0 else 0
                    ai_info.append(f"Your Preference: {most_used} ({percentage:.1f}%)")
                else:
                    ai_info.append("Learning your patterns...")
                
                ai_info.extend(["", "How it works:", "• A* finds optimal path", "• Predicts your behavior", "• Ghosts intercept ahead"])
                
                for info in ai_info:
                    if info:
                        color = BLUE if info.startswith("•") else BLACK
                        rendered = self.small_font.render(info, True, color)
                        self.screen.blit(rendered, (ui_x, y_offset))
                    y_offset += 18
                    
            except Exception as e:
                error_text = self.small_font.render(f"AI Error: {str(e)[:25]}...", True, RED)
                self.screen.blit(error_text, (ui_x, y_offset))
                y_offset += 20
        
        # Controls
        y_offset = HEIGHT - 140
        controls_title = self.font.render("Controls:", True, BLUE)
        self.screen.blit(controls_title, (ui_x, y_offset))
        y_offset += 25
        
        controls = [
            "WASD/Arrows: Move",
            "SPACE: New Maze",
            "R: Reset Position",
            "P: Toggle Predictions",
            "T: Toggle Trails", 
            "I: Toggle AI Info",
            "ESC: Exit"
        ]
        
        for control in controls:
            rendered = self.small_font.render(control, True, BLACK)
            self.screen.blit(rendered, (ui_x, y_offset))
            y_offset += 16

    def draw_game_elements(self):
        """Draw all game elements in correct order"""
        # Clear screen
        self.screen.fill(WHITE)
        
        # Draw maze
        self.maze.draw(self.screen)
        
        # Draw ghost trails (behind everything)
        self.draw_ghost_trails()
        
        # Draw predictions (behind player and ghosts)
        self.draw_predictions()
        
        # Draw player
        self.player.draw(self.screen)
        
        # Draw ghosts (on top)
        self.draw_ghosts()
        
        # Draw UI
        self.draw_ui()

    def run(self):
        """Main game loop"""
        running = True
        
        print("🎮 Greedy Predictive Ghost Maze Started!")
        print("📋 How it works:")
        print("   • AI uses A* to find your optimal path to goal")
        print("   • Learns your movement preferences over time")
        print("   • Ghosts predict where you'll be and intercept")
        print("   • Press P to see predictions, I for AI info")
        print()
        
        while running:
            running = self.handle_events()
            
            # Update game state
            self.update_game_state()
            
            # Draw everything
            self.draw_game_elements()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
        
        # Print final statistics
        print("\n🏁 Final Game Statistics:")
        print(f"   Games Completed: {self.games_completed}")
        print(f"   Times Caught: {self.total_catches}")
        print(f"   Best Escape Time: {self.best_time:.1f}s" if self.best_time != float('inf') else "   Best Time: Never escaped!")
        
        if self.games_completed > 0:
            stats = self.ai_agent.get_learning_stats()
            print(f"   AI Learned {stats['patterns_learned']} behavior patterns")
            print(f"   Directional preferences: {stats.get('directional_preferences', {})}")
        
        pygame.quit()
        sys.exit()

def main():
    game = PredictiveMazeGame()
    game.run()

if __name__ == "__main__":
    main()