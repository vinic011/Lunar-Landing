
import numpy as np

def reward_engineering(state, action, reward, next_state, done, time):
    """
    Makes reward engineering to allow faster training in problem environment.

    :param state: state.
    :type state: NumPy array with dimension (1, 2).
    :param action: action.
    :type action: int.l
    :param reward: original reward.
    :type reward: float.
    :param next_state: next state.
    :type next_state: NumPy array with dimension (1, 2).
    :param done: if the simulation is over after this experience.
    :type done: bool.
    :param time: used in some tested rewards.
    :type time: int.
    :return: modified reward for faster training.
    :rtype: float.
    """
    dt = state[0] ** 2 + state[1] ** 2
    dt = np.sqrt(dt)
    dt1 = next_state[0] ** 2 + next_state[1] ** 2
    dt1 = np.sqrt(dt1)
    vt = state[2] ** 2 + state[3] ** 2
    vt = np.sqrt(vt)
    vt1 = next_state[2] ** 2 + next_state[3] ** 2
    vt1 = np.sqrt(vt1)
    omegat = state[5]
    omegat1 = next_state[5]

    ## Comente esta linha para rodar o dqn_evaluation.
    reward += 80 * (vt - vt1) - 60 * (np.abs(omegat - omegat1)) + 100*(dt-dt1) + 5*(state[6] + state[7]) -0.05*np.abs(state[0])

    return reward

