import os  # Make sure this is at the top
import torch
import random
import numpy as np
import pygame
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 20_000
BATCH_SIZE = 128
LR = 0.001

class Agent:
    def __init__(self, model_path="best_model.pth"):
        self.n_games = 0
        self.epsilon = 0 
        self.gamma = 0.9  
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


        if os.path.exists(model_path):
            self.model.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"No model found at {model_path}. Starting fresh.")

    def load(self, model_path="best_model.pth"):
        if os.path.exists(model_path):
            self.model.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"No model found at {model_path}. Starting fresh.")

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)), 

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            

            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move



def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    model_path = "best_model.pth"
    
 
    agent = Agent(model_path=model_path)
    game = SnakeGameAI()

    try:
       while True:
          
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  
                    pygame.quit()
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q: 
                    pygame.quit()
       
            state_old = agent.get_state(game)

     
            final_move = agent.get_action(state_old)

          
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

  
            agent.train_short_memory(state_old, final_move, reward, state_new, done)


            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
     
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save("best_model.pth") 
                print(f'Game {agent.n_games} Score {score} Record: {record}')

                plot_scores.append(score)
                total_score += score

               
                mean_score = total_score / agent.n_games if agent.n_games > 0 else 0
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        
        print("Saving the model...")
        agent.model.save("best_model.pth")
        pygame.quit() 
        print("Game Over and resources cleaned up.")

if __name__ == '__main__':
    train()