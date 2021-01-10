import math
from variable import *
def direction_rack_command(key, current_rack, MAX_i, MAX_j, i, j, c):
    if key == pygame.K_RIGHT:
        pre_c = c
        if j < MAX_j:
            j = j + 1
        c = math.floor(j / 4)
        if c > pre_c:
            current_rack.container[i][pre_c].unselect()
    elif key == pygame.K_LEFT:
        pre_c = c
        if j > 0:
            j = j - 1
        c = math.floor(j / 4)
        if c < pre_c:
            current_rack.container[i][pre_c].unselect()
    elif key == pygame.K_UP:
        current_rack.container[i][c].unselect()
        if i > 0:
            i = i - 1
    else:
        current_rack.container[i][c].unselect()
        if i < MAX_i:
            i = i + 1
    return i, j, c

def skip_rack_command(key, current_rack):
    if key == pygame.K_w:
        if current_rack.top != None:
            current_rack.unselect()
            return current_rack.top
    elif key == pygame.K_a:
        if current_rack.left != None:
            current_rack.unselect()
            return current_rack.left
    elif key == pygame.K_s:
        if current_rack.down != None:
            current_rack.unselect()
            return current_rack.down
    elif key == pygame.K_d:
        if current_rack.right != None:
            current_rack.unselect()
            return current_rack.right
    return current_rack

def direction_jobqueue_command(key, job_i, slot, current_page):
    MAX = (len(slot) - current_page * 8) - 1
    if MAX > 7:
        MAX = 7

    if key == pygame.K_RIGHT:
        if current_page < 3:
            current_page = current_page + 1
            MAX = (len(slot) - current_page * 8) - 1
            if MAX > 7:
                MAX = 7
            if job_i > MAX:
                job_i = MAX
            elif job_i < 0:
                job_i = 0
    elif key == pygame.K_LEFT:
        if current_page > 0:
            current_page = current_page - 1
            MAX = (len(slot) - current_page * 8) - 1
            if MAX > 7:
                MAX = 7
            if job_i > MAX:
                job_i = MAX
            elif job_i < 0:
                job_i = 0
    elif key == pygame.K_UP:
        if job_i > 0 and job_i <= MAX:
            job_i = job_i - 1
    else:
        if job_i < MAX:
            job_i = job_i + 1
    return job_i, current_page
