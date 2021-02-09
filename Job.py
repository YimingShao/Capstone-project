from variable import *
import pygame
import random
import numpy as np

class Job(pygame.sprite.Sprite):
    def __init__(self, x_coor, y_coor):
        self.is_selected = False
        pygame.sprite.Sprite.__init__(self)
        self.id = id_gen()
        self.color_represent = color_gen()
        self.surf = pygame.Surface((JOB_WDITH, JOB_HEIGHT))
        self.surf.fill(BACKGROUND_COLOR)
        self.rect = self.surf.get_rect()
        self.x = x_coor
        self.y = y_coor
        self.is_running = False
        self.iteration = 0
        self.iteration_left = 2000
        self.speed = random.randint(1, 3)
        self.job_size = random.randint(int(JOB_SIZE_HEIGHT / 10), JOB_SIZE_HEIGHT)

        self.res_vec = [random.randint(1, 4),0]
        self.len = 0
        self.numcells = self.res_vec * self.len

        self.enter_time = 0
        self.start_time = -1
        self.finish_time = -1
        self.stepcounter = 0

        self.d_ex = 0.1
        self.m = 0.1
        self.d_m = self.d_ex * self.m
        self.gradsize = 0.5
        self.modelname = 0
        self.d_f = 0
        self.d_done = 0
        self.fraction_done = 0
        self.tt_m = 0.1
        self.rt_m = 0.1
        self.v_m = 0
        self.color = None
        self.ts_togo = 0
        self.ts_done = 0
        self.singlejoblimbw = np.Inf
        self.multijoblimbw = np.Inf
        self.scale = 1
        self.gpusassigned_set = set()

        
        pygame.draw.rect(self.surf, (0,0,0), self.rect, JOB_BORDER)
        color_rect = pygame.Rect(ID_RECT_X, ID_RECT_Y, ID_RECT_WIDTH, ID_RECT_WIDTH)

        job_size_outer = pygame.Rect(JOB_SIZE_X, JOB_SIZE_Y, JOB_SIZE_WIDTH, JOB_SIZE_HEIGHT)
        job_size_inner = pygame.Rect(JOB_SIZE_X, JOB_SIZE_MAX-self.job_size, JOB_SIZE_WIDTH, self.job_size)

        pygame.draw.rect(self.surf, self.color_represent, color_rect)
        pygame.draw.rect(self.surf, (0,0,0), color_rect, JOB_SIZE_BORDER)
        pygame.draw.rect(self.surf, JOB_SIZE_COLOR, job_size_outer, JOB_SIZE_BORDER)
        pygame.draw.rect(self.surf, JOB_SIZE_COLOR, job_size_inner)

        for i in range(1, self.speed+1):
            triangle = [[SPEED_ONE_X+(i-1)*SPEED_SPACE, SPEED_ONE_Y], [SPEED_THREE_X+(i-1)*SPEED_SPACE, SPEED_THREE_Y], [SPEED_TWO_X+(i-1)*SPEED_SPACE,SPEED_TWO_Y]]
            pygame.draw.polygon(self.surf, SPEED_COLOR, triangle)
            pygame.draw.polygon(self.surf, (0,0,0), triangle, SPEED_BORDER)
        iteration_bar_outer = pygame.Rect(ITERATION_BAR_X, ITERATION_BAR_Y, ITERATION_BAR_WIDTH * 2000,
                                    ITERATION_BAR_HEIGHT)
        pygame.draw.rect(self.surf, (0, 0, 0), iteration_bar_outer, ITERATION_BAR_BORDER)

        iteration_bar = pygame.Rect(ITERATION_BAR_X, ITERATION_BAR_Y, ITERATION_BAR_WIDTH * self.iteration, ITERATION_BAR_HEIGHT)
        pygame.draw.rect(self.surf, PROGRESS_BAR_COLOR, iteration_bar)

    @property
    def d_rem(self):
        return self.d_f - self.d_done

    @property
    def g(self):
        """
        Return the number of gpus most recently assigned/set for the job.
        :return:
        :rtype: int
        """
        return len(self.gpusassigned_set)

    def update(self, queue, y_coor):
        self.y = y_coor
        self.update_gpu_requirement_bar()
        if len(self.gpusassigned_set) < self.res_vec[0]:
            self.is_running = False
        else:
            self.is_running = True
        if self.is_running:
            pygame.draw.circle(self.surf, BACKGROUND_COLOR, (STOPPED_JOB_X, STOPPED_JOB_Y), STOPPED_JOB_RADIUS)
            triangle = [[RUNNING_ONE_X, RUNNING_ONE_Y], [RUNNING_TWO_X, RUNNING_TWO_Y],
                        [RUNNING_THREE_X, RUNNING_THREE_Y]]
            pygame.draw.polygon(self.surf, RUNNING_COLOR, triangle)
        elif not self.is_running and self.iteration > 0:
            triangle = [[RUNNING_ONE_X, RUNNING_ONE_Y], [RUNNING_TWO_X, RUNNING_TWO_Y],
                        [RUNNING_THREE_X, RUNNING_THREE_Y]]
            pygame.draw.polygon(self.surf, BACKGROUND_COLOR, triangle)
            pygame.draw.circle(self.surf, JOB_STOP_COLOR, (STOPPED_JOB_X, STOPPED_JOB_Y), STOPPED_JOB_RADIUS)
        queue.surf.blit(self.surf, (self.x, self.y))

    def progressing(self):
        if self.iteration_left > 0 and self.is_running:
            self.iteration += self.speed
            self.iteration_left -= self.speed
            iteration_bar = pygame.Rect(ITERATION_BAR_X, ITERATION_BAR_Y, ITERATION_BAR_WIDTH * self.iteration,
                                        ITERATION_BAR_HEIGHT)
            pygame.draw.rect(self.surf, (106, 132, 156), iteration_bar)
            pygame.draw.rect(self.surf, (0, 0, 0), iteration_bar, ITERATION_BAR_BORDER)


    def selected(self):
        self.is_selected = True
        pygame.draw.rect(self.surf, JOB_SELECTED_COLOR, self.rect, JOB_BORDER_SELECTED)

    def unselect(self):
        self.is_selected = False
        pygame.draw.rect(self.surf, BACKGROUND_COLOR, self.rect, JOB_BORDER_SELECTED)
        pygame.draw.rect(self.surf, (0, 0, 0), self.rect, JOB_BORDER)

    def update_gpu_requirement_bar(self):
        gpu_requirment_bar_outer = pygame.Rect(GPU_REQUIRE_X, GPU_REQUIRE_Y, GPU_REQUIRE_WIDTH, GPU_REQUIRE_HEIGHT)
        pygame.draw.rect(self.surf, BACKGROUND_COLOR, gpu_requirment_bar_outer)
        pygame.draw.rect(self.surf, (0, 0, 0), gpu_requirment_bar_outer, GPU_REQUIRE_BORDER)
        for i in range(1, self.res_vec[0] + 1):
            inner = pygame.Rect(GPU_REQUIRE_X + (i - 1) * GPU_REQUIRE_CUBE_WIDTH, GPU_REQUIRE_Y, GPU_REQUIRE_CUBE_WIDTH,
                                    GPU_REQUIRE_CUBE_HEIGHT)
            pygame.draw.rect(self.surf, GPU_REQUIRMENT_COLOR, inner)
            if i <= len(self.gpusassigned_set):
                pygame.draw.rect(self.surf, (206, 227, 43), inner, GPU_REQUIRE_BORDER)
            else:
                pygame.draw.rect(self.surf, (0, 0, 0), inner, GPU_REQUIRE_BORDER)
        for i in range(self.res_vec[0] + 1, len(self.gpusassigned_set) + 1):
            inner = pygame.Rect(GPU_REQUIRE_X + (i - 1) * GPU_REQUIRE_CUBE_WIDTH, GPU_REQUIRE_Y, GPU_REQUIRE_CUBE_WIDTH,
                                GPU_REQUIRE_CUBE_HEIGHT)
            pygame.draw.rect(self.surf, self.color_represent, inner)
            pygame.draw.rect(self.surf, (0, 0, 0), inner, GPU_REQUIRE_BORDER)

    def pick_gpu(self, gpu, jobqueue):
        if not gpu in self.gpusassigned_set:
            if len(self.gpusassigned_set) < 10:
                gpu.pick_up(self)
                self.gpusassigned_set.add(gpu)
                if self.res_vec[0] == self.g:
                    jobqueue.running_jobs[self.id] = self
                    jobqueue.free_jobs.pop(self.id)
        else:
            self.release_gpu(gpu)
            if self.res_vec[0] < self.g and self.id in jobqueue.running_jobs:
                jobqueue.running_jobs.pop(self.id)
            gpu.finished()

        '''
        When a gpu is added to the slot, update the bws
        '''
        get_env().findlimitingbws()

    def release_gpu(self, gpu):
        self.gpusassigned_set.remove(gpu)

