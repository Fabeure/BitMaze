import pygame 
import math
import random
import os
import time

import numpy as np
pygame.init()



#pygame-definitions:
creature_set = set()
food_set = set()
obstacle_set = set()
clock = pygame.time.Clock()
white = (255,255,255)
red = (255,0,0)
black = (000,000,000)
green = (50, 205, 50)
moves = [1,-1]
touchdist = 10
Background = pygame.image.load(os.path.join("./Assets", "Background.jpg"))
Font = pygame.font.Font('freesansbold.ttf', 10)
game_display = pygame.display.set_mode((1080,720))
game_display.blit(Background,(0,0))
pygame.display.update()


    
    
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

class Genome():
    pass
class NeuralNet():
    pass

class Value:
    pass

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
class Food(Location, Value):
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
        self.height = 5 #?? to be made
        self.width = 5 #?? to be made


class Creature(NeuralNet, Location, Genome):
    def __init__(self,x,y):
        self.width = 10
        self.height = 10
        self.x = x
        self.y = y
        self.distance_to_food = 0
        self.dead = False
        self.close_to_obstacle = False
        self.close_to_food = False
        self.color = white
        self.touching_creature = set()
        self.touching_food = set()
        self.touching_obstacle = set()
        self.distance_to_food = 0
    def check_color(self):
        if self.close_to_food:
            self.color = green
            print("Winning")
        elif self.close_to_creature:
        	self.color = green

        elif self.close_to_obstacle:
            self.color = red
            print("Losing")
            creature_set.remove(self)
        
        else : 
            self.color = white
    def score(self, score):
        text = Font.render(f'Score : {str(score)}', True, (0,0,0))
        game_display.blit(text, (self.x + 5, self.y + 5))

#initialising creature set       
def initialise_creatures(n, creature_set, x,y):
    creature_set.clear()
    for i in range(n):
        creature_set.add(Creature(x, y))

def food_spawn():
    done = False
    cur = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if click[2] == True:
        initialise_food(1, food_set, cur[0], cur[1])
        pygame.draw.rect(game_display,green, [cur[0], cur[1], 20,20]  )
        pygame.display.update()
        print("will spawn food here")
        done = True
    return done
#choosing creature spawn point:
def creature_spawn():
    done = False
    cur = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if click[0] == True:
        initialise_creatures(1, creature_set, cur[0], cur[1])
        pygame.draw.rect(game_display,white, [cur[0], cur[1], 20,20]  )
        
        pygame.display.update()
        print("will spawn creature here")
        done = True
    return done

#initialising food set 
def initialise_food(n, food_set, x,y):
    for i in range(n):
        food_set.add(Food(x,y))
    


#updating creatures in creature set (position, state)
def update_creatures(creature_list, auto, input, inititial_distance):
    for creature in creature_list.copy() :
        if auto == True:
            creature.move()
        elif auto == False:
            if input == 1:
                creature.x += 1
            if input == -1:
                creature.x -= 1
            if input == 2:
                creature.y -= 1
            if input == -2:
                creature.y += 1
            if input == 3:
                creature.x += 1
                creature.y -= 1
            if input == -3:
                creature.x -= 1
                creature.y -= 1
            if input == 4:
                creature.x += 1
                creature.y += 1
            if input == -4:
                creature.x -= 1
                creature.y += 1
            else: pass
        score = round(inititial_distance-creature.distance_to_food)
        creature.check_close(creature_set)
        creature.check_close(obstacle_set)
        creature.check_close(food_set)
        creature.check_color()
        creature.score(score)
        creature.draw()
    input = 0

#drawing obscatles :
def draw_obstacles():
    cur = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if (click[0] == True):
        Obstacle(cur[0], cur[1]).draw()
        pygame.display.update()
        obstacle_set.add(Obstacle(cur[0], cur[1]))
#updating food in food set :
def update_food(food_list):
    for food in food_list.copy() :
        food.draw()
#update obsatcle in obstacle set:
def update_obstacle(obstacle_list):
    for obstacle in obstacle_list.copy() :
        obstacle.draw()



def get_movement():
    keypress = 0
    keys = pygame.key.get_pressed()
    if keys[pygame.K_d]:
        keypress = 1
    if keys[pygame.K_q]:
        keypress = -1
    if keys[pygame.K_z]:
        keypress = 2
    if keys[pygame.K_s]:
        keypress = -2
    if keys[pygame.K_z] and keys[pygame.K_d]:
        keypress = 3
    if keys[pygame.K_z] and keys[pygame.K_q]:
        keypress = -3
    if keys[pygame.K_s] and keys[pygame.K_d]:
        keypress = 4
    if keys[pygame.K_s] and keys[pygame.K_q]:
        keypress = -4
    return keypress
#GAME LOOP OPTIMISATION NEEDED 
def game_loop():
    RUNNING, PAUSED, INITIALISING= 1,0,-1
    state = INITIALISING
    game_exit = False
    spawned1 = False
    spawned2 = False
    Auto = True
    keypress = 0
    while not game_exit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if (event.type == pygame.MOUSEBUTTONDOWN) and (state == RUNNING):
                mousex, mousey = pygame.mouse.get_pos() 
                creature_set.add(Creature(mousex,mousey))   
            if event.type == pygame.KEYDOWN:            
                if event.key == pygame.K_SPACE:
                    if (state == PAUSED) or (state == INITIALISING):
                        print("resumed")
                        state = RUNNING
                    else :
                        print("paused")
                        state = PAUSED
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:
            	    pygame.quit()
                quit()
        if (state == INITIALISING) and (spawned1 == False):
                if creature_spawn():
                    global cx, cy
                    cx, cy =pygame.mouse.get_pos() 
                    spawned1 = True
        if (state == INITIALISING) and (spawned1 == True):
                if food_spawn():
                    global fx, fy
                    fx, fy =pygame.mouse.get_pos() 
                    spawned2 = True
        if (state == INITIALISING) and (spawned2 == True): 
                draw_obstacles()
        if state == RUNNING:
            keypress = 0
            keypress = get_movement()
            abs_dist = math.dist((cx,cy),(fx,fy))
            clock.tick(60)
            #print(clock.get_fps())       
            game_display.blit(Background, (0,0))
            update_obstacle(obstacle_set)
            update_food(food_set)
            update_creatures(creature_set, Auto, keypress, abs_dist)
            pygame.display.update()

#initialise_food(10, food_set)
game_loop()

    
            
