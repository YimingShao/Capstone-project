import time
import random
from Job_queue import Job_queue
from Cluster import Cluster
from side_view import Side_view
from commands import *
from variable import *

# Initialize the game windowD
pygame.init()
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
pygame.display.set_caption("Scheduler")
screen.fill(BACKGROUND_COLOR)

cluster1 = Cluster(CLUSTER_SPACE, CLUSTER_SPACE)
cluster2 = Cluster(CLUSTER_SPACE * 2 + CLUSTER_WIDTH, CLUSTER_SPACE)
cluster3 = Cluster(CLUSTER_SPACE, CLUSTER_SPACE * 2 + CLUSTER_HEIGHT)
cluster4 = Cluster(CLUSTER_SPACE * 2 + CLUSTER_WIDTH, CLUSTER_SPACE * 2 + CLUSTER_HEIGHT)

jobqueue = Job_queue(cluster2, QUEUE_X, CLUSTER_SPACE)
current_cluster = cluster1

cluster1.set_up(None, cluster3, None, cluster2)
cluster2.set_up(None, cluster4, cluster1, jobqueue)
cluster3.set_up(cluster1, None, None, cluster4)
cluster4.set_up(cluster2, None, cluster3, jobqueue)
side_view = Side_view(SIDE_VIEW_X, SIDE_VIEW_Y, jobqueue)

running = True

# Main loop
i = 0
j = 0
c = 0
current_page = 0
job_i = 0
MAX_i = len(cluster1.container) - 1
MAX_j = len(cluster1.container[0]) * 4 - 1
initial_time = time.perf_counter()
current_job = None
while running:
    current_cluster.selected()
    cluster1.update()
    cluster2.update()
    cluster3.update()
    cluster4.update()
    jobqueue.update(current_page)
    side_view.update(current_page)
    if (time.perf_counter() - initial_time > 1 and jobqueue.calculate_total_job() <= 31):
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = color_gen(n)
        jobqueue.insert(color, side_view)
        # color_array.pop(0)
        initial_time = time.perf_counter()
    if isinstance(current_cluster, Cluster):
        current_cluster.container[i][c].selected()
        # if not current_cluster.get_gpu(i, j).picked_up:
        current_cluster.select_gpu(i, j, c)
    else:
        current_cluster.select_job(job_i, current_page)
    screen.blit(cluster1.surf, (cluster1.x, cluster1.y))
    screen.blit(cluster3.surf, (cluster3.x, cluster3.y))
    screen.blit(cluster2.surf, (cluster2.x, cluster2.y))
    screen.blit(cluster4.surf, (cluster4.x, cluster4.y))
    screen.blit(side_view.surf, (side_view.x, side_view.y))
    screen.blit(jobqueue.surf, (jobqueue.x, jobqueue.y))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if isinstance(current_cluster, Cluster):
                current_cluster.unselect_gpu(i, j, c)
            if event.key in [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]:
                if isinstance(current_cluster, Cluster):
                    current_cluster.container[i][c].unselect()
                i = 0
                j = 0
                c = 0
                current_cluster = skip_cluster_command(event.key, current_cluster)
                if isinstance(current_cluster, Cluster):
                    current_cluster.container[i][c].selected()
                    current_cluster.select_gpu(i, j, c)
            elif event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                if isinstance(current_cluster, Cluster):
                    i,j,c = direction_cluster_command(event.key, current_cluster, MAX_i, MAX_j, i, j, c)
                else:
                    current_cluster.unselect_job(job_i, current_page)
                    job_i, current_page = direction_jobqueue_command(event.key, job_i, current_cluster.container, current_page)
            elif event.key == pygame.K_SPACE:
                if isinstance(current_cluster, Cluster):
                    if current_job != None:
                        # if current_job.is_avaliable():
                        current_job.pick_gpu(current_cluster.gpu_all_list[i][j])
                else:
                    current_job = current_cluster.pick_up_job(job_i, current_page)
    pygame.display.update()

