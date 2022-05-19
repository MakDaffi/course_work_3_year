# Import os for file nav
import os 
# Import callback class from sb3
from stable_baselines3.common.callbacks import BaseCallback
# import ppo for training
from stable_baselines3 import PPO
from environment import VizDoomGym

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

if __name__ == '__main__':
    CHECKPOINT_DIR = './train/train_corridor'
    LOG_DIR = './logs/log_corridor'
    CONFIG ='VizDoom/scenarios/deadly_corridor.cfg'
    env = VizDoomGym(render=True, config=CONFIG)
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=8192, clip_range=.1, gamma=.95, gae_lambda=.9)
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    model.learn(total_timesteps=1000000, callback=callback)