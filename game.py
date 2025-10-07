from queue import Queue
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np


pygame.init()
font = pygame.font.SysFont('arial', 25)
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE = (0, 0, 255)
BLACK = (0,0,0)
BLOCK_SIZE = 10
SPEED = 60

class SnakeGameAI:

    def __init__(self, w=640, h=480, render=True):
        self.w = w
        self.h = h
        self.plot_width = 400
        self.render = render
        if self.render:
            self.display = pygame.display.set_mode((self.w + self.plot_width, self.h))
            pygame.display.set_caption('Змейка')
        else:
            self.display = None
        self.clock = pygame.time.Clock()
        self.grid_w = self.w // BLOCK_SIZE
        self.grid_h = self.h // BLOCK_SIZE
        self.plot_surf = None
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.foods = []
        while len(self.foods) < 10:
            self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        food = Point(x, y)
        if food == self.head:
            self._place_food()
        else:
            self.foods.append(food)

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.render:
                    pygame.quit()
                    quit()
                else:
                    pygame.quit()
                    quit()

        self._move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        ate_food = False
        for food in self.foods[:]:
            if self.head == food:
                self.foods.remove(food)
                self.score += 1
                reward = 10
                ate_food = True
                self._place_food()
        if not ate_food:
            self.snake.pop()
        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
    
    def _update_ui(self):
        if not self.render or self.display is None:
            return
        self.display.fill(BLACK)
        for food in self.foods:
            pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Счёт: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        if self.plot_surf:
            self.display.blit(self.plot_surf, (self.w, 0))
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def _nearest_food(self):
        if not self.foods:
                return None
        closest_food = self.foods[0]
        min_distance = abs(self.head.x - closest_food.x) + abs(self.head.y - closest_food.y)
        for food in self.foods[1:]:
            distance = abs(self.head.x - food.x) + abs(self.head.y - food.y)
            if distance < min_distance:
                min_distance = distance
                closest_food = food
        return closest_food
    
    def _furthest_tail(self):
        if not self.snake:
                return None
        tail_end = self.snake[-1]
        return tail_end
    
    def _simulate_move(self, action):
        virt_snake = self.snake.copy()
        virt_direction = self.direction
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(virt_direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        virt_direction = new_dir
        x = virt_snake[0].x
        y = virt_snake[0].y
        if virt_direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif virt_direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif virt_direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif virt_direction == Direction.UP:
            y -= BLOCK_SIZE
        virt_head = Point(x, y)
        ate_food = any(food == virt_head for food in self.foods)
        virt_snake.insert(0, virt_head)
        if not ate_food:
            virt_snake.pop()
        return virt_head, virt_snake, ate_food
    
    def _get_occupied_cells(self, virt_snake):
        occupied = set()
        for point in virt_snake:
            gx = point.x // BLOCK_SIZE
            gy = point.y // BLOCK_SIZE
            occupied.add((gx, gy))
        return occupied
    
    def _bfs_reachable_count(self, virt_head, virt_snake, ate_food):
        gx = virt_head.x // BLOCK_SIZE
        gy = virt_head.y // BLOCK_SIZE
        start = (gx, gy)
        occupied = self._get_occupied_cells(virt_snake)
        visited = {start}
        queue = Queue()
        queue.put(start)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while not queue.empty():
            current = queue.get()
            for dx, dy in directions:
                next_cell = (current[0] + dx, current[1] + dy)
                if (0 <= next_cell[0] < self.grid_w and 0 <= next_cell[1] < self.grid_h and next_cell not in occupied and next_cell not in visited):
                    visited.add(next_cell)
                    queue.put(next_cell)
        tail_reachable = False
        if ate_food:
            tail_gx = virt_snake[-1].x // BLOCK_SIZE
            tail_gy = virt_snake[-1].y // BLOCK_SIZE
            tail_reachable = (tail_gx, tail_gy) in visited
        return len(visited), tail_reachable
