# Import os for file nav
import os 
# Import callback class from sb3
from stable_baselines3.common.callbacks import BaseCallback
# import ppo for training
from stable_baselines3 import PPO
from environment import VizDoomGym
# Import eval policy to test agent
from stable_baselines3.common.evaluation import evaluate_policy
import time

if __name__ == '__main__':
    CHECKPOINT_DIR = './train/train_corridor'
    LOG_DIR = './logs/log_corridor'
    CONFIG ='VizDoom/scenarios/deadly_corridor.cfg'
    env = VizDoomGym(render=True, config=CONFIG)
    # Reload model from disc
    model = PPO.load('train/train_corridor/best_model_560000')
    # Evaluate mean reward for 10 games
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    for episode in range(20): 
        obs = env.reset()
        done = False
        total_reward = 0
        while not done: 
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            time.sleep(0.02)
            total_reward += reward
        print('Total Reward for episode {} is {}'.format(total_reward, episode))
        time.sleep(2)