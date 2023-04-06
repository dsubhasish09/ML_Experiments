from gym.envs.registration import register

register(id='cartpole-v0', entry_point='gym_custom.envs:CartPole_Base')
register(id='cartpole-v1', entry_point='gym_custom.envs:CartPole_UnderAct')