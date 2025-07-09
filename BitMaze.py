import pygame 
import math
import random
import os
import time
import copy

import numpy as np
pygame.init()



#pygame-definitions:
creature_set = []
food_set = []
obstacle_set = []
clock = pygame.time.Clock()
white = (255,255,255)
red = (255,0,0)
black = (000,000,000)
green = (50, 205, 50)
blue = (0, 0, 255)
yellow = (255, 255, 0)
purple = (128, 0, 128)
orange = (255, 165, 0)
pink = (255, 192, 203)
cyan = (0, 255, 255)
gray = (128, 128, 128)
colors = [white, blue, yellow, purple, orange, pink, cyan, gray]
moves = [1,-1]
touchdist = 15
vision_range = 100
Background = pygame.image.load(os.path.join("./Assets", "Background.jpg"))
Font = pygame.font.Font('freesansbold.ttf', 16)
Small_Font = pygame.font.Font('freesansbold.ttf', 12)
game_display = pygame.display.set_mode((1080,720))
pygame.display.set_caption("BitMaze - Evolutionary Neural Network")
game_display.blit(Background,(0,0))
pygame.display.update()

# Evolution parameters
POPULATION_SIZE = 50
GENERATION_TIME = 10  # seconds per generation
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
ELITE_SIZE = 5  # top performers to keep

# Neural network parameters
INPUT_SIZE = 8  # distance to food x,y, nearest obstacle x,y, wall distances (4 directions)
HIDDEN_SIZE = 12
OUTPUT_SIZE = 2  # movement x,y

# Speed control
SPEED_LEVELS = [15, 30, 60, 120, 240, 480]  # FPS levels for different speeds
SPEED_NAMES = ["Slow", "Normal", "Fast", "Very Fast", "Ultra Fast", "Maximum"]
current_speed_index = 2  # Start at "Fast" (60 FPS)


    
    
#classes/methods:  
class Location():
    def draw(self):
        pygame.draw.rect(game_display,self.color, [self.x, self.y, self.width,self.height]  )
    def dist(self, entity):
        return math.dist((self.x, self.y),(entity.x, entity.y))


    def check_close(self, creatures):
        for creature in creatures:
            if (isinstance(creature, Food)):
                self.distance_to_food = self.dist(creature)
            if (self.dist(creature) < touchdist) and (((self.x, self.y) != (creature.x, creature.y))):
                if isinstance(creature, Creature):
                    self.touching_creature.add(creature)
                    creature.touching_creature.add(self)
                elif isinstance(creature, Food):
                    self.touching_food.add(creature)
                    creature.touching_food.add(self)
                elif isinstance(creature, Obstacle):
                    self.touching_obstacle.add(creature)
                    creature.touching_obstacle.add(self)
            if  (self.dist(creature) >= touchdist) and ((creature in self.touching_food) or (creature in self.touching_obstacle) or (creature in self.touching_creature)):
                if isinstance(creature, Creature):
                    self.touching_creature.remove(creature)
                    creature.touching_creature.remove(self)
                elif isinstance(creature, Food):
                    self.touching_food.remove(creature)
                    creature.touching_food.remove(self)
                elif isinstance(creature, Obstacle):
                    self.touching_obstacle.remove(creature)
                    creature.touching_obstacle.remove(self)
        if len(self.touching_food)>= 1:
            self.close_to_food = True
        else : 
            self.close_to_food = False
        if len(self.touching_obstacle)>=1:
            self.close_to_obstacle = True
        else : 
            self.close_to_obstacle = False
        if len(self.touching_creature)>=1:
            self.close_to_creature = True
        else :
            self.close_to_creature = False

    def move(self):
        self.x += random.choice(moves)
        self.y += random.choice(moves)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(np.clip(x, -500, 500))

class NeuralNet:
    def __init__(self, weights=None):
        if weights is None:
            # Initialize random weights
            self.weights1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.5
            self.bias1 = np.random.randn(HIDDEN_SIZE) * 0.5
            self.weights2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.5
            self.bias2 = np.random.randn(OUTPUT_SIZE) * 0.5
        else:
            self.weights1, self.bias1, self.weights2, self.bias2 = weights
    
    def forward(self, inputs):
        # Forward pass through neural network
        inputs = np.array(inputs)
        hidden = sigmoid(np.dot(inputs, self.weights1) + self.bias1)
        output = tanh(np.dot(hidden, self.weights2) + self.bias2)
        return output
    
    def get_weights(self):
        return (self.weights1.copy(), self.bias1.copy(), self.weights2.copy(), self.bias2.copy())
    
    def mutate(self, rate=MUTATION_RATE):
        # Mutate weights with given probability
        if random.random() < rate:
            self.weights1 += np.random.randn(*self.weights1.shape) * 0.1
        if random.random() < rate:
            self.bias1 += np.random.randn(*self.bias1.shape) * 0.1
        if random.random() < rate:
            self.weights2 += np.random.randn(*self.weights2.shape) * 0.1
        if random.random() < rate:
            self.bias2 += np.random.randn(*self.bias2.shape) * 0.1

class Genome:
    def __init__(self, neural_net=None):
        self.neural_net = neural_net if neural_net else NeuralNet()
        self.fitness = 0
        self.age = 0
    
    def crossover(self, other):
        # Create offspring by crossing over weights
        w1_1, b1_1, w2_1, b2_1 = self.neural_net.get_weights()
        w1_2, b1_2, w2_2, b2_2 = other.neural_net.get_weights()
        
        # Random crossover point
        mask1 = np.random.random(w1_1.shape) < 0.5
        mask2 = np.random.random(w2_1.shape) < 0.5
        
        new_w1 = np.where(mask1, w1_1, w1_2)
        new_w2 = np.where(mask2, w2_1, w2_2)
        
        new_b1 = np.where(np.random.random(b1_1.shape) < 0.5, b1_1, b1_2)
        new_b2 = np.where(np.random.random(b2_1.shape) < 0.5, b2_1, b2_2)
        
        child_net = NeuralNet((new_w1, new_b1, new_w2, new_b2))
        child_net.mutate()
        
        return Genome(child_net)

class Obstacle(Location):
    def __init__(self, x, y):
        self.x = x 
        self.y = y
        self.color = red
        self.close_to_food = False
        self.close_to_obstacle = False
        self.height = 5
        self.width = 5
        self.touching_creature = set()
        self.touching_food = set()
        self.touching_obstacle = set()
class Food(Location):
    def __init__(self, x, y):
        self.x = x 
        self.y = y
        self.color = green
        self.close_to_food = False
        self.close_to_obstacle = False
        self.touching_creature = set()
        self.touching_food = set()
        self.touching_obstacle = set()
        self.value = random.randint(20,30)
        self.height = 8
        self.width = 8


class Creature(Location):
    def __init__(self, x, y, genome=None):
        self.width = 8
        self.height = 8
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.distance_to_food = 0
        self.initial_distance_to_food = 0
        self.dead = False
        self.reached_food = False
        self.close_to_obstacle = False
        self.close_to_food = False
        self.steps_taken = 0
        self.time_alive = 0
        self.fitness = 0
        self.color = random.choice(colors)
        self.touching_creature = set()
        self.touching_food = set()
        self.touching_obstacle = set()
        self.genome = genome if genome else Genome()
        
    def get_sensor_data(self):
        # Get input data for neural network
        inputs = []
        
        # Distance to food (normalized)
        if food_set:
            food = food_set[0]
            dx = food.x - self.x
            dy = food.y - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            self.distance_to_food = dist
            inputs.extend([dx/500, dy/500])  # Normalize to screen size
        else:
            inputs.extend([0, 0])
        
        # Distance to nearest obstacle
        min_obstacle_dist = float('inf')
        nearest_obstacle_dx, nearest_obstacle_dy = 0, 0
        
        for obstacle in obstacle_set:
            dx = obstacle.x - self.x
            dy = obstacle.y - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_obstacle_dist:
                min_obstacle_dist = dist
                nearest_obstacle_dx, nearest_obstacle_dy = dx, dy
        
        if min_obstacle_dist != float('inf'):
            inputs.extend([nearest_obstacle_dx/100, nearest_obstacle_dy/100])
        else:
            inputs.extend([0, 0])
        
        # Distance to walls (4 directions)
        inputs.extend([
            self.x / 1080,  # distance to left wall
            (1080 - self.x) / 1080,  # distance to right wall
            self.y / 720,   # distance to top wall
            (720 - self.y) / 720    # distance to bottom wall
        ])
        
        return inputs
    
    def move_with_nn(self):
        if self.dead:
            return
            
        # Get sensor inputs
        inputs = self.get_sensor_data()
        
        # Get neural network output
        output = self.genome.neural_net.forward(inputs)
        
        # Convert output to movement
        self.vx = output[0] * 3  # max speed of 3 pixels
        self.vy = output[1] * 3
        
        # Update position
        new_x = self.x + self.vx
        new_y = self.y + self.vy
        
        # Keep within bounds
        self.x = max(5, min(1075, new_x))
        self.y = max(5, min(715, new_y))
        
        self.steps_taken += 1
        self.time_alive += 1
        
    def check_collisions(self):
        if self.dead:
            return
            
        # Check food collision
        for food in food_set:
            if self.dist(food) < touchdist:
                self.reached_food = True
                self.close_to_food = True
                self.color = green
                return
        
        # Check obstacle collision
        for obstacle in obstacle_set:
            if self.dist(obstacle) < touchdist:
                self.dead = True
                self.color = red
                return
        
        # Check wall collision
        if self.x <= 5 or self.x >= 1075 or self.y <= 5 or self.y >= 715:
            self.dead = True
            self.color = red
            return
    
    def calculate_fitness(self):
        # Fitness function rewards getting close to food and reaching it
        fitness = 0
        
        # Reward for reaching food
        if self.reached_food:
            fitness += 1000
        
        # Reward for getting closer to food
        if self.initial_distance_to_food > 0:
            improvement = (self.initial_distance_to_food - self.distance_to_food)
            fitness += improvement * 2
        
        # Penalty for dying
        if self.dead:
            fitness -= 200
        
        # Small reward for staying alive
        fitness += self.time_alive * 0.5
        
        # Penalty for taking too many steps without progress
        if self.steps_taken > 0:
            efficiency = (self.initial_distance_to_food - self.distance_to_food) / self.steps_taken
            fitness += efficiency * 50
        
        self.fitness = fitness
        self.genome.fitness = fitness
        return fitness
    
    def reset(self, x, y):
        # Reset creature for new generation
        self.x = x
        self.y = y
        self.start_x = x
        self.start_y = y
        self.vx = 0
        self.vy = 0
        self.dead = False
        self.reached_food = False
        self.close_to_obstacle = False
        self.close_to_food = False
        self.steps_taken = 0
        self.time_alive = 0
        self.fitness = 0
        self.color = random.choice(colors)
        self.touching_creature = set()
        self.touching_food = set()
        self.touching_obstacle = set()
        
        if food_set:
            self.initial_distance_to_food = self.dist(food_set[0])
            self.distance_to_food = self.initial_distance_to_food

class EvolutionManager:
    def __init__(self):
        self.generation = 1
        self.generation_start_time = time.time()
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.spawn_x = 50
        self.spawn_y = 50
        
    def create_population(self, x, y):
        self.spawn_x = x
        self.spawn_y = y
        creature_set.clear()
        for i in range(POPULATION_SIZE):
            creature = Creature(x, y)
            if food_set:
                creature.initial_distance_to_food = creature.dist(food_set[0])
                creature.distance_to_food = creature.initial_distance_to_food
            creature_set.append(creature)
    
    def evolve_population(self):
        # Calculate fitness for all creatures
        for creature in creature_set:
            creature.calculate_fitness()
        
        # Sort by fitness (descending)
        creature_set.sort(key=lambda c: c.fitness, reverse=True)
        
        # Track statistics
        fitnesses = [c.fitness for c in creature_set]
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        print(f"Generation {self.generation}: Best={best_fitness:.1f}, Avg={avg_fitness:.1f}")
        
        # Keep elite
        new_population = []
        elite = creature_set[:ELITE_SIZE]
        
        for creature in elite:
            new_creature = Creature(self.spawn_x, self.spawn_y, copy.deepcopy(creature.genome))
            new_creature.reset(self.spawn_x, self.spawn_y)
            new_population.append(new_creature)
        
        # Create offspring
        while len(new_population) < POPULATION_SIZE:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            if random.random() < CROSSOVER_RATE:
                child_genome = parent1.genome.crossover(parent2.genome)
            else:
                child_genome = copy.deepcopy(parent1.genome)
                child_genome.neural_net.mutate()
            
            child = Creature(self.spawn_x, self.spawn_y, child_genome)
            child.reset(self.spawn_x, self.spawn_y)
            new_population.append(child)
        
        creature_set.clear()
        creature_set.extend(new_population)
        
        self.generation += 1
        self.generation_start_time = time.time()
    
    def tournament_selection(self, tournament_size=3):
        # Select random individuals for tournament
        tournament = random.sample(creature_set[:len(creature_set)//2], 
                                 min(tournament_size, len(creature_set)//2))
        return max(tournament, key=lambda c: c.fitness)
    
    def should_evolve(self):
        # Evolve when generation time is up or most creatures are dead/reached food
        time_up = time.time() - self.generation_start_time > GENERATION_TIME
        most_dead = sum(1 for c in creature_set if c.dead or c.reached_food) > POPULATION_SIZE * 0.8
        return time_up or most_dead

# Initialize evolution manager
evolution_manager = EvolutionManager()

def initialize_food(x, y):
    food_set.clear()
    food_set.append(Food(x, y))

def update_creatures():
    for creature in creature_set:
        if not creature.dead and not creature.reached_food:
            creature.move_with_nn()
            creature.check_collisions()
        creature.draw()

def update_food():
    for food in food_set:
        food.draw()

def update_obstacles():
    for obstacle in obstacle_set:
        obstacle.draw()

def draw_obstacles():
    cur = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if click[2] == True:  # Middle click (scroll wheel)
        # Create multiple small obstacles in a pattern
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                obstacle = Obstacle(cur[0] + dx * 8, cur[1] + dy * 8)
                obstacle_set.append(obstacle)

def draw_ui():
    # Draw generation info
    gen_text = Font.render(f"Generation: {evolution_manager.generation}", True, black)
    game_display.blit(gen_text, (10, 10))
    
    # Draw population stats
    alive = sum(1 for c in creature_set if not c.dead and not c.reached_food)
    dead = sum(1 for c in creature_set if c.dead)
    reached = sum(1 for c in creature_set if c.reached_food)
    
    stats_text = Small_Font.render(f"Alive: {alive} | Dead: {dead} | Reached: {reached}", True, black)
    game_display.blit(stats_text, (10, 40))
    
    # Draw time remaining
    time_elapsed = time.time() - evolution_manager.generation_start_time
    time_remaining = max(0, GENERATION_TIME - time_elapsed)
    time_text = Small_Font.render(f"Time: {time_remaining:.1f}s", True, black)
    game_display.blit(time_text, (10, 60))
    
    # Draw best fitness
    if evolution_manager.best_fitness_history:
        best_text = Small_Font.render(f"Best Fitness: {evolution_manager.best_fitness_history[-1]:.1f}", True, black)
        game_display.blit(best_text, (10, 80))
    
    # Draw speed control
    speed_text = Small_Font.render(f"Speed: {SPEED_NAMES[current_speed_index]} ({SPEED_LEVELS[current_speed_index]} FPS)", True, black)
    game_display.blit(speed_text, (10, 100))
    
    # Draw instructions
    instructions = [
        "Left Click: Place spawn point",
        "Right Click: Place food", 
        "Middle Click: Draw obstacles",
        "SPACE: Start/Pause",
        "UP/DOWN: Change speed",
        "R: Reset simulation",
        "ESC: Quit"
    ]
    
    for i, instruction in enumerate(instructions):
        text = Small_Font.render(instruction, True, black)
        game_display.blit(text, (10, 130 + i * 20))
#MAIN GAME LOOP
def game_loop():
    SETUP, RUNNING, PAUSED = 0, 1, 2
    state = SETUP
    game_exit = False
    spawn_set = False
    food_set_placed = False
    
    while not game_exit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                if state == SETUP:
                    if event.button == 1 and not spawn_set:  # Left click - set spawn
                        evolution_manager.spawn_x = mouse_x
                        evolution_manager.spawn_y = mouse_y
                        spawn_set = True
                        print(f"Spawn point set at ({mouse_x}, {mouse_y})")
                        
                    elif event.button == 3 and spawn_set and not food_set_placed:  # Right click - set food
                        initialize_food(mouse_x, mouse_y)
                        food_set_placed = True
                        print(f"Food placed at ({mouse_x}, {mouse_y})")
                        
                    elif event.button == 2 and spawn_set and food_set_placed:  # Middle click - draw obstacles
                        draw_obstacles()
                        
                elif state == RUNNING:
                    if event.button == 2:  # Middle click - draw obstacles during simulation
                        draw_obstacles()
                        
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if state == SETUP and spawn_set and food_set_placed:
                        # Start simulation
                        evolution_manager.create_population(evolution_manager.spawn_x, evolution_manager.spawn_y)
                        state = RUNNING
                        print("Simulation started!")
                    elif state == RUNNING:
                        state = PAUSED
                        print("Simulation paused")
                    elif state == PAUSED:
                        state = RUNNING
                        print("Simulation resumed")
                        
                elif event.key == pygame.K_UP:
                    # Increase speed
                    global current_speed_index
                    if current_speed_index < len(SPEED_LEVELS) - 1:
                        current_speed_index += 1
                        print(f"Speed increased to: {SPEED_NAMES[current_speed_index]} ({SPEED_LEVELS[current_speed_index]} FPS)")
                        
                elif event.key == pygame.K_DOWN:
                    # Decrease speed
                    if current_speed_index > 0:
                        current_speed_index -= 1
                        print(f"Speed decreased to: {SPEED_NAMES[current_speed_index]} ({SPEED_LEVELS[current_speed_index]} FPS)")
                        
                elif event.key == pygame.K_r:
                    # Reset simulation
                    creature_set.clear()
                    food_set.clear()
                    obstacle_set.clear()
                    evolution_manager.__init__()
                    state = SETUP
                    spawn_set = False
                    food_set_placed = False
                    print("Simulation reset")
                    
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()
        
        # Update simulation
        if state == RUNNING:
            # Check if evolution should occur
            if evolution_manager.should_evolve():
                evolution_manager.evolve_population()
            
            # Update creatures
            update_creatures()
            
            # Use current speed setting
            clock.tick(SPEED_LEVELS[current_speed_index])
        
        # Render everything
        game_display.blit(Background, (0, 0))
        
        if state == SETUP:
            # Draw setup instructions
            if not spawn_set:
                text = Font.render("Left click to set creature spawn point", True, black)
                game_display.blit(text, (300, 300))
            elif not food_set_placed:
                text = Font.render("Right click to place food", True, black)
                game_display.blit(text, (300, 300))
                # Draw spawn point indicator
                pygame.draw.circle(game_display, blue, (evolution_manager.spawn_x, evolution_manager.spawn_y), 15, 3)
            else:
                text = Font.render("Middle click to draw obstacles, SPACE to start", True, black)
                game_display.blit(text, (250, 300))
                # Draw spawn and food indicators
                pygame.draw.circle(game_display, blue, (evolution_manager.spawn_x, evolution_manager.spawn_y), 15, 3)
        
        elif state == PAUSED:
            text = Font.render("PAUSED - Press SPACE to resume", True, red)
            game_display.blit(text, (350, 350))
        
        # Always draw existing elements
        update_obstacles()
        update_food()
        
        # Draw creatures if they exist
        if creature_set:
            for creature in creature_set:
                creature.draw()
        
        # Draw UI
        draw_ui()
        
        pygame.display.update()

if __name__ == "__main__":
    game_loop()
