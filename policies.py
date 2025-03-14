import gymnasium as gym
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris
# from tetris_gymnasium.components.tetromino import Tetromino
from copy import deepcopy 

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
    stones_by_row = [list(i).count("x")+list(i).count("o") for i in grid]
    num_lines = len([i for i in stones_by_row if i == width])

    heights = [0 for i in range(width)]
    holes = 0
    for c in range(width):  # Iterate from bottom to top
        has_seen_x = False
        for r in range(height):
            if grid[r][c] == 'x' or grid[r][c] == 'o':
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

def move_down(grid):
    """Move the box down and return the new grid
    
    Arguments:
        grid: List of lists with x, o
    
    Returns: New grid, another list of lists with x and o"""

    new_grid = deepcopy(grid)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if new_grid[i][j] == "o":
                new_grid[i][j] = "."
    
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == "o":
                if i+1<len(grid) and new_grid[i+1][j] == ".":
                    new_grid[i+1][j] = "o"
                else:
                    return None  
    return new_grid 

def move_down_max(grid):
    """Move the box down and return the new grid
    
    Arguments:
        grid: List of lists with x, o
    
    Returns: New grid, another list of lists with x and o"""
    new_grid = deepcopy(grid)
    max_moves = 1000
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if new_grid[i][j] == "o":
                for i_prime in range(i+1,len(grid)):
                    if new_grid[i_prime][j] == "x":
                        max_moves = min(max_moves,(i_prime-i)-1)
                        break 
                else:
                    max_moves = min(max_moves,(len(grid)-i)-1)
                new_grid[i][j] = "."
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == "o":
                new_grid[i+max_moves][j] = "o"
    return new_grid, max_moves

def move_left(grid):
    """Move the box to the left and return the new grid
    
    Arguments:
        grid: List of lists with x, o
    
    Returns: New grid, another list of lists with x and o"""

    new_grid = deepcopy(grid)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if new_grid[i][j] == "o":
                new_grid[i][j] = "."
    
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == "o":
                if i+1<len(grid) and j-1>=0 and new_grid[i+1][j-1] == ".":
                    new_grid[i+1][j-1] = "o"
                else:
                    return None 
    return new_grid 

def move_right(grid):
    """Move the box to the right and return the new grid
    
    Arguments:
        grid: List of lists with x, o
    
    Returns: New grid, another list of lists with x and o"""

    if grid is None:
        return grid 

    new_grid = deepcopy(grid)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if new_grid[i][j] == "o":
                new_grid[i][j] = "."
    
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == "o":
                if i+1<len(grid) and j+1<len(grid[0]) and new_grid[i+1][j+1] == ".":
                    new_grid[i+1][j+1] = "o"
                else:
                    return None 
    return new_grid 
    
def rotate_clockwise(grid,active_mask):
    """Rotate the box clockwise and return the new grid
    
    Arguments:
        grid: List of lists with x, o
        active_mask: 0-1 matrix with 1s where the rotation box is
    
    Returns: New grid, another list of lists with x and o"""

    if grid is None:
        return grid 

    new_grid = deepcopy(grid)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if new_grid[i][j] == "o":
                new_grid[i][j] = "."

    total_active_mask = round(np.sum(active_mask)**.5)
    if np.sum(active_mask) != total_active_mask**2:
        return None, None 
    curr_active_mask = grid[active_mask == 1].reshape((total_active_mask,total_active_mask))
    rotated_mask = np.rot90(curr_active_mask,k=1)
    new_active_mask = [['.' for j in range(len(grid[0]))] for i in range(len(grid))]
    smallest_i = 1000
    smallest_j = 1000

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if active_mask[i][j] == 1:
                smallest_i = min(i,smallest_i)
                smallest_j = min(j,smallest_j)
                if i == len(grid)-1:
                    return None, None
                else:
                    new_active_mask[i+1][j] = rotated_mask[i-smallest_i][j-smallest_j]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if new_active_mask[i][j] == 'o':
                new_active_mask[i][j] = 1
                if grid[i][j] == '.' or grid[i][j] == 'o':
                    new_grid[i][j] = 'o'
                else:
                    return None, None 
    new_active_mask = np.concatenate(([[0 for i in range(len(active_mask[0]))]], active_mask[:-1]))
    return new_grid, new_active_mask

def rotate_counterclockwise(grid,active_mask):
    """Rotate the box clockwise and return the new grid
    
    Arguments:
        grid: List of lists with x, o
        active_mask: 0-1 matrix with 1s where the rotation box is
    
    Returns: New grid, another list of lists with x and o"""

    if grid is None:
        return grid 

    new_grid = deepcopy(grid)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if new_grid[i][j] == "o":
                new_grid[i][j] = "."

    total_active_mask = round(np.sum(active_mask)**.5)
    if np.sum(active_mask) != total_active_mask**2:
        return None, None 

    curr_active_mask = grid[active_mask == 1].reshape((total_active_mask,total_active_mask))
    rotated_mask = np.rot90(curr_active_mask,k=-1)
    new_active_mask = [['.' for j in range(len(grid[0]))] for i in range(len(grid))]
    smallest_i = 1000
    smallest_j = 1000

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if active_mask[i][j] == 1:
                smallest_i = min(i,smallest_i)
                smallest_j = min(j,smallest_j)
                if i == len(grid)-1:
                    return None, None 
                else:
                    new_active_mask[i+1][j] = rotated_mask[i-smallest_i][j-smallest_j]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if new_active_mask[i][j] == 'o':
                new_active_mask[i][j] = 1
                if grid[i][j] == '.' or grid[i][j] == 'o':
                    new_grid[i][j] = 'o'
                else:
                    return None, None 
    new_active_mask = np.concatenate(([[0 for i in range(len(active_mask[0]))]], active_mask[:-1]))
    return new_grid, new_active_mask

def swap(grid, holder, queue):
    """Switch the current piece with the held piece
    
    Arguments:
        grid: List of lists with x, o
        holder: One Integer or '.', for what's in the hold currently
        queue: List of four numbers, the upcoming pieces
    
    Returns: New grid, another list of lists with x and o"""
    def switch(grid, piece_id):
        new_grid = deepcopy(grid)
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if new_grid[i][j] == "o":
                    new_grid[i][j] = "."

        piece_matrix = Tetris.TETROMINOES[piece_id-2].matrix
        piece = [['.' if piece_matrix[j][i]==0 else 'o' for i  in range(piece_matrix.shape[1])] for j in range(piece_matrix.shape[0])]
        position = len(piece)

        new_grid[1:position+1, (grid.shape[1]-position+1)//2:(grid.shape[1]-position+1)//2 + position] = deepcopy(piece)
        return new_grid

    if holder == '.':
        return switch(grid, queue[0])
    else:
        return switch(grid, holder)


def play_max_score(env,obs):
    """Play the move that maximizes score_grid
    
    Arguments:
        env: Gym Environment
        obs: Dictionary, with three things
            'board': List of Lists with x, ., and o for closed, open, and piece
            'queue': List of four numbers, the upcoming pieces
            'holder': One Integer or '.', for what's in the hold currently
    
    Returns: Action, whichever maximizes the score_grid function"""

    grid = obs['board']
    active_mask = obs['active_mask']
    holder = obs['holder']
    queue = obs['queue']

    _,width = len(grid),len(grid[0])

    curr_move_overall = ''
    scores = []
    moves = []
    for swap_ in ['', 's']:
        for rotation in ['','c','w','ww']:
            if swap_ == 's':
                new_grid = swap(grid,holder,queue)
            else:
                new_grid = grid
            if rotation == 'c':
                new_grid,_ = rotate_clockwise(grid,active_mask)
                if new_grid is None:
                    continue 
            if rotation == 'w':
                new_grid,_ = rotate_counterclockwise(grid,active_mask)
                if new_grid is None:
                    continue 
            elif rotation == 'ww':
                new_grid,new_active_mask = rotate_counterclockwise(grid,active_mask)
                if new_grid is None:
                    continue 
                new_grid,_ = rotate_counterclockwise(new_grid,new_active_mask)
                if new_grid is None:
                    continue 

            grid_left = deepcopy(new_grid)
            for num_left in range(width):
                if grid_left is None:
                    break 
                else:
                    moved_down_grid,num_move_down = move_down_max(grid_left)
                    moves.append(swap_ + rotation + "l"*num_left+"d"*num_move_down)
                    scores.append(score_grid(moved_down_grid))
                    grid_left = move_left(grid_left)

            grid_right = deepcopy(new_grid)
            for num_right in range(width):
                if grid_right is None:
                    break 
                else:
                    moved_down_grid,num_move_down = move_down_max(grid_right)
                    moves.append(swap_ + rotation + "r"*num_right+"d"*num_move_down)
                    scores.append(score_grid(moved_down_grid))
                    grid_right = move_right(grid_right)

    best_move = np.argmax(scores)
    curr_move_overall = moves[best_move]

    if len(curr_move_overall) == 0:
        return [env.unwrapped.actions.hard_drop]
    actions = []
    for action in curr_move_overall:
        if action=='s':
            actions.append(env.unwrapped.actions.swap)
        elif action=='l':
            actions.append(env.unwrapped.actions.move_left)
        elif action=='r':
            actions.append(env.unwrapped.actions.move_right)
        elif action=='c':
            actions.append(env.unwrapped.actions.rotate_clockwise)
        elif action=='w':
            actions.append(env.unwrapped.actions.rotate_counterclockwise)
        elif action=='d':
            actions.append(env.unwrapped.actions.hard_drop)
            return(actions)

    return actions
    

# print(Tetris.TETROMINOES[0])