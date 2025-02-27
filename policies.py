import gymnasium as gym
import numpy as np

from tetris_gymnasium.envs.tetris import Tetris


def random_move(env,obs):
    """Basic move functionality that returns a random move
    
    Arguments:
        env: Gym Environment
        obs: Dictionary, with three things
            'board': List of Lists with x, ., and o for closed, open, and piece
            'queue': List of four numbers, the upcoming pieces
            'holder': One Integer or '.', for what's in the hold currently
    
    Returns: Action, one of the following:
        env.unwrapped.actions.move_left
        env.unwrapped.actions.move_right
        env.unwrapped.actions.move_down
        env.unwrapped.actions.rotate_counterclockwise
        env.unwrapped.actions.rotate_clockwise
        env.unwrapped.actions.swap
        env.unwrapped.actions.hard_drop
    """

    action = env.action_space.sample()
    return action 

