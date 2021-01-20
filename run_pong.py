import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from gym.envs.classic_control import rendering
from replayMemory import replayMemory
from myDQN import DQN

task = "train"
ModelPath="Model4"
TOTAL_EPISODES = 550
viewer = rendering.SimpleImageViewer()

game = 'PongNoFrameskip-v4'
env = gym.make(game)

name = "DQN_" + game

episodes_reward = []
episodes_qvalue = []
last_100_avgs = []
global_step = 0
episode = 0
bestResult = -1e5
bestEpisodeReward = -1e5
lastResult = 0

state_size = env.observation_space.shape
num_actions = env.action_space.n

syncTarget = 1000

env = wrappers.Monitor(env, "recording/" + name, force=True)

initializeReplayBuffer = 10000
repBufferSize = 50000
sampleSize = 64
startingEpsilon = 1.0
endEpsilon = 0.001
epsilonDecay = 10 ** 4
learn_rate = 0.0001

factor = (endEpsilon - startingEpsilon) / epsilonDecay
frameskip = 2

width = 84
height = 84

def startTraining():
    global episode
    global global_step
    global lastResult
    global bestResult
    global bestEpisodeReward
    print("\n\n FILLING REPLAY BUFFER... \n\n")
    initializationExperiences = 0
    while (initializationExperiences < initializeReplayBuffer):
        env.reset()
        f, _, _, lives = env.step(1)

        state = DQN.inputPreprocess(f)
        d = False

        while not (d):
            a = np.random.randint(0, num_actions)

            r = 0
            for i in range(frameskip):
                f1, rew, d, lives = env.step(a)
                r += rew

                if d:
                    break

            if r > 0:
                r = 1
            elif r < 0:
                r = -1

            newState = DQN.inputPreprocess(f1)
            memory.addExperience(state, a, r, d, newState)

            initializationExperiences += 1
            state = newState

    print("\n\n STARTING TRAINING.. \n\n")
    for i in range(TOTAL_EPISODES):
        episode_reward = 0
        episode_qvalues = []
        env.reset()
        f, _, _, lives = env.step(1)

        DQN.resetObservationState()
        state = DQN.inputPreprocess(f)
        d = False

        while not (d):
            a, qvalue = DQN.actionSelection(state)

            r = 0
            for i in range(frameskip):
                f1, rew, d, lives = env.step(a)
                r += rew

                if d:
                    break

            if r > 0:
                r = 1
            elif r < 0:
                r = -1

            newState = DQN.inputPreprocess(f1)

            memory.addExperience(state, a, r, d, newState)

            if (global_step % 4 == 0):
                DQN.training(memory.sampleExperience())

            if global_step <= epsilonDecay:
                DQN.epsilon = (factor * global_step) + startingEpsilon
            else:
                DQN.epsilon = endEpsilon

            if (global_step % (4 * syncTarget) == 0 and global_step != 0):
                print("\n\nGlobal step", global_step, "Updating target network..\n\n")
                DQN.updateTargetNetwork()

            if global_step % 50000 == 0:
                summ = DQN.sess.run(DQN.mergeFilters)
                DQN.writeOps.add_summary(summ, global_step=global_step)

            state = newState
            global_step += 1
            episode_reward += r
            episode_qvalues.append(qvalue)

        avgQVal = np.mean(episode_qvalues)
        episodes_reward.append(episode_reward)
        lastResult = np.mean(episodes_reward[-100:])
        last_100_avgs.append(lastResult)

        summ = DQN.sess.run(DQN.mergeEpisodeData, feed_dict={DQN.averagedReward: lastResult,
                                                             DQN.PHEpsilon: DQN.epsilon,
                                                             DQN.avgQValue: avgQVal})
        DQN.writeOps.add_summary(summ, global_step=episode)

        if episode_reward > bestEpisodeReward:
            bestEpisodeReward = episode_reward

        if lastResult > bestResult:
            if lastResult >= 17 or bestResult == -1e5:
                print("Saving model..")
                DQN.save_restore_Model(restore=False, globa_step=global_step, episode=episode,
                                       rewards=episodes_reward[-100:], episodes_reward=episodes_reward,
                                       last_100_avgs=last_100_avgs)
            bestResult = lastResult

        print("\nEnded episode:", episode, "Global step:", global_step, "curEpsilon:", DQN.epsilon,
              "EpisodeReward:", episode_reward, "bestEpisodeReward:", bestEpisodeReward,
              "bestAvgReward:", "%.3f" % bestResult, "Avg Reward:", "%.3f" % lastResult, "\n")

        episode += 1

    print("Saving model..")
    DQN.save_restore_Model(restore=False, globa_step=global_step, episode=episode, rewards=episodes_reward[-100:],
                               episodes_reward=episodes_reward, last_100_avgs=last_100_avgs)


def test():
    DQN.save_restore_Model(restore=True)
    episode = DQN.episode.eval()
    episodes_reward = DQN.episodes_reward.eval().tolist()[:episode]
    last_100_avgs = DQN.last_100_avgs.eval().tolist()[:episode]
    global_step = DQN.global_step.eval()

    DQN.epsilon = DQN.lastEpsilon.eval()
    print(len(episodes_reward), episodes_reward)
    print(len(last_100_avgs), last_100_avgs)
    print("globel_step:", global_step)
    print("episode:", episode)
    print("epsilon:", DQN.epsilon)
    plt.plot(last_100_avgs)
    plt.xlabel('episodes')
    plt.ylabel('Last 100 Episodes Average Reward')
    plt.title("mean_100_episodes_reward Curve")
    plt.legend()
    plt.show()
    plt.plot(episodes_reward)
    plt.xlabel('episodes')
    plt.ylabel('Episodes Reward')
    plt.title("every_episode_reward Curve")
    plt.legend()
    plt.show()

    env.reset()
    reward = 0
    f, _, _, lives = env.step(1)

    state = DQN.inputPreprocess(f)
    DQN.resetObservationState()
    d = False
    while not d:
        env.render()
        a, qvalue = DQN.actionSelection(state)

        r = 0
        for i in range(frameskip):
            f1, rew, d, lives = env.step(a)
            r += rew

            if d:
                break

        # Reward clipping
        if r > 0:
            r = 1
        elif r < 0:
            r = -1

        newState = DQN.inputPreprocess(f1)
        reward += r
        state = newState
    env.close()
    print("reward:", reward)


if __name__ == '__main__':
    with tf.Session() as sess:
        try:
            if task == "test":
                DQN = DQN(sess, num_actions=num_actions, num_frames=4, width=width, height=height, lr=learn_rate,
                          startEpsilon=0.01, folderName=name, ModelPath=ModelPath)
                writer = tf.summary.FileWriter("logs/", sess.graph)
                test()
            elif task == "train":
                DQN = DQN(sess, num_actions=num_actions, num_frames=4, width=width, height=height, lr=learn_rate,
                          startEpsilon=startingEpsilon, folderName=name, ModelPath=ModelPath)
                DQN.buildwriteOps()
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter("logs/", sess.graph)
                memory = replayMemory(sizeMemory=repBufferSize, sampleSize=sampleSize, image_height=height,
                                      image_width=width, num_frames=4)
                res = input("Do you want to load the model? [y/n]")
                if res.lower() == "y":
                    DQN.save_restore_Model(restore=True)
                    episode = DQN.episode.eval()
                    episodes_reward = DQN.episodes_reward.eval().tolist()[:episode]
                    last_100_avgs = DQN.last_100_avgs.eval().tolist()[:episode]
                    global_step = DQN.global_step.eval()

                    DQN.epsilon = DQN.lastEpsilon.eval()

                startTraining()
        except (KeyboardInterrupt, SystemExit):
            print("Program shut down, saving the model..")
            DQN.save_restore_Model(restore=False, globa_step=global_step, episode=episode,
                                   rewards=episodes_reward[-100:])
            print("\n\nModel saved!\n\n")
            raise




