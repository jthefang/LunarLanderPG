"""
Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. 

Reward for moving from the top of the screen to landing pad and (at?) zero speed is about 100..140 points. 
If lander moves away from landing pad it loses reward back. 
Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. 
Each leg ground contact is +10. Firing main engine is -0.3 points each frame. 
Solved is 200 points. Landing outside landing pad is possible. 
Fuel is infinite, so an agent can learn to fly and then land on its first attempt. 

Four discrete actions available: 
    do nothing, 
    fire left orientation engine, 
    fire main engine, 
    fire right orientation engine.
"""
import sys
sys.path.append('/Users/jfang/Research/gym')
import gym
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import time

env = gym.make('LunarLander-v2')
env = env.unwrapped

# Policy gradient has high variance, seed for reproducability
env.seed(1)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)

EPISODES = 5000
TIME_LIMIT = 30 #sec
rewards = []
RENDER_REWARD_MIN = 5000
if __name__ == "__main__":
    """
        1st arg = load directory #<---------------------oscillate between 0 and 1
        2nd arg = render environment (default = False)
        3rd arg = render every nth episode (default = 100)
        4th arg = num episodes (default = 5000)
    """
    RENDER_ENV = bool(sys.argv[2]) if len(sys.argv) > 2 else False
    RENDER_EVERY_N = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    # Load checkpoint
    load_version = int(sys.argv[1]) if len(sys.argv) > 1 else 0 #default to 0
    save_version = (load_version + 1) % 2
    load_path = "output/weights/LunarLander/{}/LunarLander-v2.ckpt".format(load_version)
    save_path = "output/weights/LunarLander/{}/LunarLander-v2.ckpt".format(save_version)

    PG = PolicyGradient(
        n_x = env.observation_space.shape[0],
        n_y = env.action_space.n,
        learning_rate=0.02,
        reward_decay=0.99,
        num_episodes=EPISODES,
        load_path=load_path,
        save_path=save_path
    )

    for episode in range(EPISODES):
        observation = env.reset()
        episode_reward = 0

        tic = time.clock()
        while True:
            #if RENDER_ENV or (episode % RENDER_EVERY_N == 0): env.render()
            # 1. Choose an action based on observation
            action = PG.choose_action(observation)
            # 2. Take action in the environment
            observation_, reward, done, info = env.step(action)
            # 3. Store transition for training
            PG.store_transition(observation, action, reward)

            toc = time.clock()
            elapsed_sec = toc - tic
            if elapsed_sec > TIME_LIMIT: #episode shouldn't last longer than this in sec
                done = True

            episode_rewards_sum = sum(PG.episode_rewards)
            if episode_rewards_sum < -250:
                done = True

            if done: 
                episode_rewards_sum = sum(PG.episode_rewards)
                rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)

                print("==========================================")
                print("Episode: ", episode)
                print("Seconds: ", elapsed_sec)
                print("Reward: ", episode_rewards_sum)
                print("Average reward so far:", np.sum(rewards) / len(rewards))
                print("Max reward so far: ", max_reward_so_far)

                # 4. Train neural network
                discounted_episode_rewards_norm = PG.learn(episode)

                if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True
                break #done with episode, then end game loop

            # Save new observation
            observation = observation_
