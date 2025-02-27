import gymnasium as gym
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris
from policies import *

def get_formatted_queue(queue):
    """Turn a list of pieces into a list of integers
    
    e.g., a 4x16 board into 4 integers for the next 4 pieces
    
    Arguments:
        queue: 4x16 numpy array

    Returns: List of integers of size 4"""

    formatted_queue = []
    for i in range(0,len(queue[0]),4):
        formatted_queue.append(np.max(queue[:,i:i+4]))
    return formatted_queue

def get_formatted_holder(holder):
    """Turn a 4x4 numpy array into a single integer for 
        the current piece in the hold
        
    Arguments:
        holder: 4x4 numpy array
        
    Returns: Either a piece from 2-9 or . for an empty holder"""

    formatted_holder = "."
    if np.max(holder)>1:
        formatted_holder = np.max(holder)
    return formatted_holder

def format_obs(obs,curr_piece):
    """Turn an observation and an integer representing the current piece
        into a simplified state space
    
    NOTE: It's not clear if the current piece works perfectly well yet
        The way they represent it might lead to some bugs

    Arguments:
        obs: Observation from OpenGym
        curr_piece: Integer, what the number of the current piece is
    
    Returns: Dictionary with three keys
        'board': array with x marking filled, o marking current piece, and . marking empty
        'queue': 4 integers (list) with the next four pieces
        'holder': Single integer or '.' for what's in the holder currently"""

    formatted_queue = get_formatted_queue(obs["queue"])
    formatted_holder = get_formatted_holder(obs["holder"])

    board =  obs["board"][:-4,4:-4]
    active_mask = obs["active_tetromino_mask"][:-4,4:-4]
    char_board = np.full(board.shape, ".", dtype=str) 
    char_board[board > 0] = "x"  
    char_board[((active_mask*board) > 0) & (board == curr_piece)] = "o"  
    return {'board': char_board, 'queue': formatted_queue, 'holder': formatted_holder}

def run_loop(action_function):
    """Main loop that plays tetris given an action function
    
    Arguments:
        action_function: Maps the environment and formatted observation (dictionary, see above)
    
    Returns: Total reward for playing """

    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    obs, _ = env.reset(seed=seed)
    curr_piece = np.max(obs["board"])
    curr_queue = get_formatted_queue(obs["queue"])
    curr_holder = get_formatted_holder(obs["holder"])

    terminated = False
    total_reward = 0
    while not terminated:
        action = action_function(env,format_obs(obs,curr_piece))
        obs, reward, terminated, _, _ = env.step(action)
        new_queue = get_formatted_queue(obs["queue"])
        new_holder = get_formatted_holder(obs["holder"])
        total_reward += reward

        if new_queue != curr_queue or new_holder != curr_holder:
            if new_holder != curr_holder and new_queue == curr_queue:
                curr_piece = curr_holder 
            else:
                curr_piece = curr_queue[0]

        curr_holder = new_holder
        curr_queue = new_queue

    print("Game Over! Score {}".format(total_reward))
    return total_reward

seed = 43
run_loop(random_move)
