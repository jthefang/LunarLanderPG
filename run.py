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

def isWithinFlags(x):
    if abs(x) <= .10:
        return 1
    return 0

def isOver45deg(theta):
    if abs(theta) > np.pi / 4: #45 degree cone
        return 1
    return 0

env = gym.make('LunarLander-v2')
env = env.unwrapped

# Policy gradient has high variance, seed for reproducability
env.seed(1)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)

RENDER_ENV = True
EPISODES = 2501
rewards = []
stock_rewards = []
RENDER_REWARD_MIN = 5000

if __name__ == "__main__":

    # Load checkpoint
    load_path = "output/weights/LunarLander/reward-l-cf-45cone/LunarLander-v2.ckpt-2500"
    save_path = "output/weights/LunarLander/save/LunarLander-v2.ckpt"

    PG = PolicyGradient(
        n_x = env.observation_space.shape[0],
        n_y = env.action_space.n,
        learning_rate=0.002,
        reward_decay=0.99,
        num_episodes=EPISODES
    )


    for episode in range(EPISODES):

        observation = env.reset()
        episode_reward = 0

        tic = time.clock()

        numStepsInEpisode = 0
        while True:
            if RENDER_ENV: env.render()

            # 1. Choose an action based on observation
            action = PG.choose_action(observation)

            # 2. Take action in the environment
            toc = time.clock()
            elapsed_sec = toc - tic
            observation_, reward, done, info = env.step(action)

            # 4. Store transition for training
            #print(observation_[0])
            """
            state = [
            0: (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            1: (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
            2: vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            3: vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            4: self.lander.angle,
            5: 20.0*self.lander.angularVelocity/FPS,
            6: 1.0 if self.legs[0].ground_contact else 0.0,
            7: 1.0 if self.legs[1].ground_contact else 0.0
            ]
            """
            obs_meters = 100 * observation_
            xPos, yPos, xVel, yVel =obs_meters[0],obs_meters[1],obs_meters[2],obs_meters[3]
            vel = np.sqrt(xVel**2 + yVel**2)
            dist = np.sqrt(xPos**2 + yPos**2)
            print("xPos: {}, yPos: {}, xVel: {}, yVel: {}".format(np.round(xPos, 3), np.round(yPos, 3), np.round(xVel, 3), np.round(yVel, 3)))
            #print("Distance: {}".format(dist))
            PG.store_transition(observation, action, reward)
            numStepsInEpisode += 1

            toc = time.clock()
            elapsed_sec = toc - tic
            if elapsed_sec > 30: #episode shouldn't last longer than 30s
                done = True

            episode_rewards_sum = sum(PG.episode_rewards)
            if episode_rewards_sum < -300:
                done = True

            if done:
                rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)

                print("==========================================")
                print("Episode: ", episode)
                print("Seconds: ", elapsed_sec)
                print("Reward: ", episode_rewards_sum)
                print("Average reward: ", np.sum(rewards) / len(rewards))
                print("Max reward so far: ", max_reward_so_far)
                print("Num steps taken this episode:", numStepsInEpisode)

                # 5. Train neural network
                discounted_episode_rewards_norm = PG.learn(episode)

                if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True
                break

            # Save new observation
            observation = observation_
