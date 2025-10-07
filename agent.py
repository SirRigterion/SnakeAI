import random
from collections import deque
import torch
import numpy as np

import helper
from model import Linear_QNet, QTrainer
from game import SnakeGameAI, Direction, Point


MAX_MEMORY = 100_000
BASIC_SIZE = 1_000
LR = 0.001

class Agent():
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 80
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(16, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def get_state(self, game):
        step = game.w // game.grid_w
        head = game.snake[0]
        point_l = Point(head.x - step, head.y)
        point_r = Point(head.x + step, head.y)
        point_u = Point(head.x, head.y - step)
        point_d = Point(head.x, head.y + step)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        nearest_food = game._nearest_food()
        furthest_tail = game._furthest_tail()
        # Вычисляем reachable_count для каждого действия
        actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]    # straight, right, left
        reachable_counts = []
        for action in actions:
            virt_head, virt_snake, ate_food = game._simulate_move(action)
            reachable_count, _ = game._bfs_reachable_count(virt_head, virt_snake, ate_food)
            reachable_counts.append(reachable_count / (game.grid_w * game.grid_h))    # Нормируем

        state = [
            # Опасности
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Направления
            dir_l, dir_r, dir_u, dir_d,

            # Еда
            nearest_food.x < game.head.x,
            nearest_food.x > game.head.x,
            nearest_food.y < game.head.y,
            nearest_food.y > game.head.y,

            # Хвост
            furthest_tail.x < game.head.x,
            furthest_tail.y < game.head.y,

            # Reachable counts для каждого действия
            reachable_counts[0],
            reachable_counts[1],
            reachable_counts[2]
        ]
        return np.array(state, dtype=float)
    
    def get_action(self, game, state):
        self.epsilon = max(5, 80 - self.n_games * 2)
        final_move = [0, 0, 0]
        actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
            return final_move
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.model(state0)
            move = torch.argmax(prediction[0]).item()
            final_move[move] = 1
        safe_moves = []
        max_reachable = -1
        best_move = final_move
        for i, action in enumerate(actions):
            virt_head, virt_snake, ate_food = game._simulate_move(action)
            reachable_count, tail_reachable = game._bfs_reachable_count(virt_head, virt_snake, ate_food)
            is_safe = reachable_count >= len(virt_snake) * 1.5
            if ate_food:
                is_safe = is_safe and tail_reachable
            if is_safe:
                safe_moves.append((action, reachable_count))
            if reachable_count > max_reachable:
                max_reachable = reachable_count
                best_move = action
        if any(np.array_equal(final_move, move[0]) for move in safe_moves):
            return final_move
        if safe_moves:
            best_safe_move = max(safe_moves, key=lambda x: x[1])[0]
            return best_safe_move
        tail_action = self._follow_tail(game)
        return tail_action if tail_action is not None else best_move
    
    def _follow_tail(self, game):
        tail = game._furthest_tail()
        actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        min_distance = float('inf')
        best_action = None
        for action in actions:
            virt_head, _, _ = game._simulate_move(action)
            distance = abs(virt_head.x - tail.x) + abs(virt_head.y - tail.y)
            if distance < min_distance and not game.is_collision(virt_head):
                min_distance = distance
                best_action = action
        return best_action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BASIC_SIZE:
            mini_sample = random.sample(self.memory, BASIC_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


def train():
    plot_scores = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(render=True)
    while True:
        nearest_food = game._nearest_food()
        prev_dist = abs(game.head.x - nearest_food.x) + abs(game.head.y - nearest_food.y)
        state_old = agent.get_state(game)
        final_move = agent.get_action(game, state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        if reward < 10:
            nearest_food_after = game._nearest_food()
            new_dist = abs(game.head.x - nearest_food_after.x) + abs(game.head.y - nearest_food_after.y)
            if new_dist < prev_dist:
                reward += 1.0
            else:
                reward -= 0.3
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
            print(f"Эпизод: {agent.n_games}  Счёт: {score}  Рекорд: {record}")
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            game.plot_surf = helper.plot(plot_scores, plot_mean_score)


if __name__ == '__main__':
    train()