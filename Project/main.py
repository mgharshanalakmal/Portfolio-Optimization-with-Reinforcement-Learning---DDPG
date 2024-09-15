import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve
from gym_environment import StockPortfolioEnv

if __name__ == "__main__":
    env = StockPortfolioEnv()
    agent = Agent(
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
    )
    n_games = 250

    figure_file = "plots/test.png"

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        n_step = 0
        while n_step <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_step += 1

        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            # if not load_checkpoint:
            agent.save_models()

        print("episode", i, "score %.1f" % score, "avg score %.1f" % avg_score)

        # if i % 5 == 0:
        #     agent.load_models()

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
