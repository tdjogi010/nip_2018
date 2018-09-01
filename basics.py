from osim.env import ProstheticsEnv

env = ProstheticsEnv()
print("obs space: ",env.observation_space)
observation = env.reset(project=False)
print(env.reward_range)
print("obs space: ",env.observation_space,"obs:",observation)
observation = env.reset()
print("obs space: ",env.observation_space,"obs:",len(observation))

for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    # print("obs space: ",env.observation_space)