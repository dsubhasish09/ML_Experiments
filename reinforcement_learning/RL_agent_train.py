from omegaconf import DictConfig, OmegaConf
import hydra
import time
import gym
import numpy as np
import os 
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from mbrl.third_party.pytorch_sac_pranz24.sac_reg_norm2 import SAC_REG
from mbrl.third_party.pytorch_sac_pranz24.sac import SAC
from mbrl.util.logger import Logger
from mbrl.util.replay_buffer import ReplayBuffer
import mbrl.constants
import torch
import itertools

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    run_identifier = cfg.run_config.run_identifier
    log_dir = cfg.train_config.log_dir
    sac_cfg = cfg.train_config.algorithm
    seed = cfg.train_config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    env_train = gym.make('gym_custom:' + cfg.sim_config.name, cfg = cfg.sim_config)
    env_train.seed(cfg.train_config.seed)
    env_train.action_space.seed(seed)

    env_eval = gym.make('gym_custom:' + cfg.sim_config.name, cfg = cfg.sim_config)
    env_eval.seed(cfg.train_config.seed)
    agent = SAC_REG(env_train.observation_space.shape[0], env_train.action_space, sac_cfg)
    agent.load_checkpoint("logs/Under_Act_Cartpole_Agent.pt", evaluate=False)
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
    eval_freq = cfg.train_config.eval_freq
    best_eval_reward = -np.inf
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

        print(
            "[TRAIN] Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
                i_episode, total_numsteps, episode_steps, round(episode_reward, 2)
            )
        )
        
        if last_eval > eval_freq:
            last_eval = 0
            eval_state = env_eval.reset(init_state=np.array([[0.75, np.pi, 0, 0]]).T)
            eval_done = False
            eval_ep_reward = 0
            eval_steps = 0
            while not eval_done:
                eval_steps += 1
                u = agent.select_action(eval_state, evaluate = True)
                next_eval_state, eval_reward, eval_done, _ = env_eval.step(u)
                eval_ep_reward += eval_reward
                eval_state = next_eval_state
            print(
            "[EVAL] Episode: {}, total numsteps: {}, eval episode steps: {}, eval reward: {}".format(
                i_episode, total_numsteps, eval_steps, round(eval_ep_reward, 2)))
            
            if eval_ep_reward > best_eval_reward:
                print("Better model found!!!")
                best_eval_reward = eval_ep_reward
                last_improve = 0
                agent.save_checkpoint(ckpt_path = log_dir + run_identifier + ".pt")

        if total_numsteps > cfg.train_config.total_train_steps:
            break
         
    

        

    env_train.close()
if __name__ == "__main__":
    main()