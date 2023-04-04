from omegaconf import DictConfig, OmegaConf
import hydra
import time
import gym
import numpy as np
import os 
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from mbrl.third_party.pytorch_sac_pranz24.sac_reg_norm import SAC_REG
from mbrl.third_party.pytorch_sac_pranz24.sac import SAC
from mbrl.util.logger import Logger
from mbrl.util.replay_buffer import ReplayBuffer
import mbrl.constants
import torch
import itertools

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    sac_cfg = cfg.train_config.algorithm
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    env_train = gym.make('gym_custom:' + cfg.sim_config.name, cfg = cfg.sim_config)
    agent = SAC_REG(env_train.observation_space.shape[0], env_train.action_space, sac_cfg)
    
    memory = ReplayBuffer(
        cfg.train_config.replay_size,
        env_train.observation_space.shape,
        env_train.action_space.shape,
        rng=np.random.default_rng(seed=seed),
    )

    # Training Loop
    total_numsteps = 0
    updates = 0
    last_improve = 0
    last_eval = 0
    env_steps = 0
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env_train.reset()

        while not done:
            if cfg.train_config.start_steps > total_numsteps:
                action = env_train.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)
            if len(memory) > cfg.train_config.batch_size:
                # Number of updates per step in environment
                for i in range(cfg.train_config.updates_per_step):
                    # Update parameters of all the networks
                    (
                        critic_1_loss,
                        critic_2_loss,
                        policy_loss,
                        ent_loss,
                        alpha,
                    ) = agent.update_parameters(
                        memory, cfg.train_config.batch_size, updates
                    )

                    updates += 1
            # print(action)
            # print(state, action)
            next_state, reward, done, _ = env_train.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            last_improve += 1
            last_eval += 1
            episode_reward += reward
            env_steps += 1

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = True if episode_steps == env_train.max_episode_steps else not done
           
            memory.add(state, action, next_state, reward, mask)
    
            state = next_state
        # print(state)
        if total_numsteps > cfg.train_config.total_train_steps:
            break

        print(
            "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
                i_episode, total_numsteps, episode_steps, round(episode_reward, 2)
            )
        )
            
    

        

    env_train.close()
if __name__ == "__main__":
    main()