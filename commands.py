import math
from variable import *
def direction_cluster_command(key, current_cluster, MAX_i, MAX_j, i, j, c):
    if key == pygame.K_RIGHT:
        pre_c = c
        if j < MAX_j:
            j = j + 1
        c = math.floor(j / 4)
        if c > pre_c:
            current_cluster.container[i][pre_c].unselect()
    elif key == pygame.K_LEFT:
        pre_c = c
        if j > 0:
            j = j - 1
        c = math.floor(j / 4)
        if c < pre_c:
            current_cluster.container[i][pre_c].unselect()
    elif key == pygame.K_UP:
        current_cluster.container[i][c].unselect()
        if i > 0:
            i = i - 1
    else:
        current_cluster.container[i][c].unselect()
        if i < MAX_i:
            i = i + 1
    return i, j, c

def skip_cluster_command(key, current_cluster):
    if key == pygame.K_w:
        if current_cluster.top != None:
            current_cluster.unselect()
            return current_cluster.top
    elif key == pygame.K_a:
        if current_cluster.left != None:
            current_cluster.unselect()
            return current_cluster.left
    elif key == pygame.K_s:
        if current_cluster.down != None:
            current_cluster.unselect()
            return current_cluster.down
    elif key == pygame.K_d:
        if current_cluster.right != None:
            current_cluster.unselect()
            return current_cluster.right
    return current_cluster

def direction_jobqueue_command(key, job_i, pages, current_page):
    MAX = len(pages[current_page]) - 1
    if key == pygame.K_RIGHT:
        if current_page < 3:
            current_page = current_page + 1
            MAX = len(pages[current_page]) - 1
            if job_i > MAX and MAX >= 0:
                job_i = MAX
    elif key == pygame.K_LEFT:
        if current_page > 0:
            current_page = current_page - 1
            MAX = len(pages[current_page]) - 1
            if job_i > MAX and MAX >= 0:
                job_i = MAX
    elif key == pygame.K_UP:
        if job_i > 0:
            job_i = job_i - 1
    else:
        if job_i < MAX:
            job_i = job_i + 1
    return job_i, current_page