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

def score_grid(grid):
    """Give a heuristic score to a grid based on lines, etc.
    
    Arguments:
        grid: List of lists with x where there is a stone
    
    Returns: Float, heuristic score"""

    height,width = len(grid),len(grid[0])
    stones_by_row = [i.count("x") for i in grid]
    num_lines = len([i for i in stones_by_row if i == width])

    heights = [0 for i in range(width)]
    holes = 0
    for c in range(width):  # Iterate from bottom to top
        has_seen_x = False
        for r in range(height):
            if grid[r][c] == 'x':
                heights[c] = max(heights[c],height - r)
                has_seen_x = True
            elif grid[r][c] == '.' and has_seen_x:
                holes += 1 
            
    bumpiness = 0
    for i in range(len(heights)-1):
        bumpiness+=abs(heights[i+1]-heights[i])

    a = -0.510066
    b = 0.760666
    c = -0.35663
    d = -0.184483

    return a*sum(heights) + b*num_lines + c*holes + d*bumpiness

def play_max_score(env,obs):
    """Play the move that maximizes score_grid
    
    Arguments:
        env: Gym Environment
        obs: Dictionary, with three things
            'board': List of Lists with x, ., and o for closed, open, and piece
            'queue': List of four numbers, the upcoming pieces
            'holder': One Integer or '.', for what's in the hold currently
    
    Returns: Action, whichever maximizes the score_grid function"""

    curr_board = obs['board']

    # TODO: Implement a way to compute the heuristic for each up/down/left/right/rotate CW/rotate CCW