# import ppo for training
from stable_baselines3 import PPO
from environment import VizDoomGym
import time

if __name__ == '__main__':
    CHECKPOINT_DIR = './train/train_corridor'
    LOG_DIR = './logs/log_corridor'
    CONFIG ='VizDoom/scenarios/deadly_corridor.cfg'
    env = VizDoomGym(render=True, config=CONFIG)
    # Reload model from disc
    model = PPO.load('train/train_corridor/best_model_560000')
    # Evaluate mean reward for 10 games
    final_reward = 0
    min_rew = 1e10
    max_rew = -1e10
    for episode in range(100): 
        obs = env.reset()
        done = False
        total_reward = 0
        st = 0
        while not done:
            st += 1
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            time.sleep(0.02)
            total_reward += reward
            print(f'Reward on step {st} in episode {episode} : {reward}')
        final_reward += total_reward
        min_rew = min(min_rew, total_reward)
        max_rew = max(max_rew, total_reward)
        print('Total Reward for episode {} is {}'.format(total_reward, episode))
        time.sleep(2)
    print(f'Maximal Reward : {max_rew}')
    print(f'Minimal Reward : {min_rew}')
    print(f'ep_rew_mean : {final_reward / 100}')