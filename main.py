import time
from Job_queue import Job_queue
from Rack import Rack
from side_view import Side_view
from commands import *
from variable import *


# Initialize the game windowD
pygame.init()
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
pygame.display.set_caption("Scheduler")
screen.fill(BACKGROUND_COLOR)

rack1 = Rack(CLUSTER_SPACE, CLUSTER_SPACE)
rack2 = Rack(CLUSTER_SPACE * 2 + CLUSTER_WIDTH, CLUSTER_SPACE)
rack3 = Rack(CLUSTER_SPACE, CLUSTER_SPACE * 2 + CLUSTER_HEIGHT)
rack4 = Rack(CLUSTER_SPACE * 2 + CLUSTER_WIDTH, CLUSTER_SPACE * 2 + CLUSTER_HEIGHT)

jobqueue = Job_queue(rack2, QUEUE_X, CLUSTER_SPACE)
create_env(jobqueue)
current_rack = rack1

rack1.set_up(None, rack3, None, rack2)
rack2.set_up(None, rack4, rack1, jobqueue)
rack3.set_up(rack1, None, None, rack4)
rack4.set_up(rack2, None, rack3, jobqueue)
side_view = Side_view(SIDE_VIEW_X, SIDE_VIEW_Y, jobqueue.slot)

running = True

# Main loop
i = 0
j = 0
c = 0
current_page = 0
job_i = 0
MAX_i = len(rack1.container) - 1
MAX_j = len(rack1.container[0]) * 4 - 1
initial_time = time.perf_counter()
current_job = None

iteration = 0
while running:

    print(get_env().get_reward_fullpreempt())

    current_rack.selected()
    # Todo, updates
    '''
    v(below)-cbb. in scope of env.update()
        for r in range(len(self.rack).racks)):
            r.update()
    '''
    rack1.update()
    rack2.update()
    rack3.update()
    rack4.update()
    jobqueue.update(current_page, job_i)
    side_view.update(current_page)

    if  not current_job in jobqueue.slot:
        current_job = None

    if (time.perf_counter() - initial_time > 1 and len(jobqueue.slot) < 32):

        jobqueue.insert()
        initial_time = time.perf_counter()

    if isinstance(current_rack, Rack):
        current_rack.container[i][c].selected()
        current_rack.select_gpu(i, j, c)

    screen.blit(rack1.surf, (rack1.x, rack1.y))
    screen.blit(rack3.surf, (rack3.x,  rack3.y))
    screen.blit(rack2.surf, (rack2.x, rack2.y))
    screen.blit(rack4.surf, (rack4.x, rack4.y))
    screen.blit(side_view.surf, (side_view.x, side_view.y))
    screen.blit(jobqueue.surf, (jobqueue.x, jobqueue.y))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if isinstance(current_rack, Rack):
                current_rack.unselect_gpu(i, j, c)
            if event.key in [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]:
                if isinstance(current_rack, Rack):
                    current_rack.container[i][c].unselect()
                i = 0
                j = 0
                c = 0
                current_rack = skip_rack_command(event.key, current_rack)
                if isinstance(current_rack, Rack):
                    current_rack.container[i][c].selected()
                    current_rack.select_gpu(i, j, c)
            elif event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                if isinstance(current_rack, Rack):
                    i,j,c = direction_rack_command(event.key, current_rack, MAX_i, MAX_j, i, j, c)
                else:
                    job_i, current_page = direction_jobqueue_command(event.key, job_i, current_rack.slot, current_page)
            elif event.key == pygame.K_SPACE:
                if isinstance(current_rack, Rack):
                    if current_job != None:
                        current_job.pick_gpu(current_rack.gpu_all_list[i][j], jobqueue)
                else:
                    current_job = current_rack.pick_up_job(job_i, current_page)
    iteration += 1
    if iteration > 10:
        iteration = 0
        get_env().advance_runningjobs_onestep()
    pygame.display.update()


