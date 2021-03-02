import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos, radians, ceil
from variable import *

class Job(pygame.sprite.Sprite):
    def __init__(self, x_coor, y_coor):
        self.is_selected = False
        pygame.sprite.Sprite.__init__(self)
        self.id = get_env().id_gen()
        self.asset_gen()
        self.color_represent = get_env().color_gen()
        self.surf = pygame.Surface((JOB_WIDTH, JOB_HEIGHT))
        self.surf.fill(BACKGROUND_COLOR)
        self.rect = self.surf.get_rect()
        self.x = x_coor
        self.y = y_coor
        self.iteration = 0
        self.iteration_left = 2000
        self.angle= random.randint(1, 180)
        self.speed = radians(self.angle)
        self.job_size = random.randint(int(JOB_SIZE_WIDTH/ 10), JOB_SIZE_WIDTH)

        self.res_vec = [random.randint(1, 5),0]
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
        # color_rect = pygame.Rect(COLOR_RECT_X, COLOR_RECT_Y, COLOR_RECT_WIDTH, COLOR_RECT_HEIGHT)

        self.image = pygame.image.load("assets/current.png")
        rect = self.image.get_rect()
        rect.x = ASSET_X
        rect.y = ASSET_Y
        rect.width = ASSET_WIDTH
        rect.height = ASSET_HEIGHT
        self.surf.blit(pygame.transform.scale(self.image, (ASSET_WIDTH, ASSET_HEIGHT)), rect)
        pygame.draw.rect(self.surf, (0, 0, 0), rect, JOB_SIZE_BORDER)

        job_size_outer = pygame.Rect(JOB_SIZE_X, JOB_SIZE_Y, JOB_SIZE_WIDTH, JOB_SIZE_HEIGHT)
        job_size_inner = pygame.Rect(JOB_SIZE_X, JOB_SIZE_Y, self.job_size, JOB_SIZE_HEIGHT)

        pygame.draw.rect(self.surf, JOB_SIZE_COLOR, job_size_outer, JOB_SIZE_BORDER)
        pygame.draw.rect(self.surf, JOB_SIZE_COLOR, job_size_inner)

        self.update_gpu_grid()

        img = speed_img
        rect = img.get_rect()
        rect.x = SPEED_X
        rect.y = SPEED_Y
        self.surf.blit(pygame.transform.scale(img, (SPEED_WIDTH, SPEED_HEIGHT)), rect)
        origin = (SPEED_ORIGIN_X, SPEED_ORIGIN_Y)
        if self.angle < 90:
            end_point = (SPEED_ORIGIN_X - SPEED_RADIUS * abs(cos(self.angle)),
                        SPEED_ORIGIN_Y - SPEED_RADIUS * abs(sin(self.angle)))
        else:
            end_point = (SPEED_ORIGIN_X + SPEED_RADIUS * abs(cos(180 - self.angle)),
                         SPEED_ORIGIN_Y - SPEED_RADIUS * abs(sin(180 - self.angle)))


        pygame.draw.line(self.surf, (0,0,0), origin, end_point, 5)

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
        # self.update_gpu_grid()
        queue.surf.blit(self.surf, (self.x, self.y))

    def progressing(self):
        if len(self.gpusassigned_set) >= self.res_vec[0]:
            self.iteration += self.speed
            self.iteration_left -= self.speed

            arc = [RUNNING_CIRCLE_X, RUNNING_CIRCLE_Y, RUNNING_CIRCLE_WIDTH, RUNNING_CIRCLE_WIDTH]
            pygame.draw.arc(self.surf, RUNNING_COLOR, arc
                             , (pi / 2) - (self.iteration / 1000) * pi, pi / 2, 10)
            #self.surf.blit(arc, (RUNNING_CIRCLE_X, RUNNING_CIRCLE_Y))

            # iteration_bar = pygame.Rect(ITERATION_BAR_X, ITERATION_BAR_Y, ITERATION_BAR_WIDTH * self.iteration,
            #                             ITERATION_BAR_HEIGHT)
            # pygame.draw.rect(self.surf, (106, 132, 156), iteration_bar)
            # pygame.draw.rect(self.surf, (0, 0, 0), iteration_bar, ITERATION_BAR_BORDER)
        else:
            arc = [RUNNING_CIRCLE_X, RUNNING_CIRCLE_Y, RUNNING_CIRCLE_WIDTH, RUNNING_CIRCLE_WIDTH]
            pygame.draw.arc(self.surf, JOB_STOP_COLOR, arc
                            , (pi / 2) - (self.iteration / 1000) * pi, pi / 2, 10)

    def selected(self):
        self.is_selected = True
        pygame.draw.rect(self.surf, JOB_SELECTED_COLOR, self.rect, JOB_BORDER_SELECTED)

    def unselect(self):
        self.is_selected = False
        pygame.draw.rect(self.surf, BACKGROUND_COLOR, self.rect, JOB_BORDER_SELECTED)
        pygame.draw.rect(self.surf, (0, 0, 0), self.rect, JOB_BORDER)

    def update_gpu_grid(self):
        gpu_w = ceil(GPU_GRID_WIDTH/16)
        gpu_h = ceil(GPU_GRID_HEIGHT/16)

        i = 0
        for row in range(0, 16):
            for column in range(0, 16):
                gpu = pygame.Rect(GPU_GRID_X + column * gpu_w, GPU_GRID_Y + row * gpu_h, gpu_w, gpu_h)
                if i <= self.res_vec[0]:
                    if i < self.g:
                        pygame.draw.rect(self.surf, (206, 227, 43), gpu)
                    else:
                        pygame.draw.rect(self.surf, GPU_REQUIRMENT_COLOR, gpu)
                elif i < self.g:
                    pygame.draw.rect(self.surf, self.color_represent, gpu)
                else:
                    pygame.draw.rect(self.surf, (0, 0, 0), gpu, 1)
                i += 1



    def pick_gpu(self, gpu, jobqueue):
        if not gpu in self.gpusassigned_set:
            if len(self.gpusassigned_set) < 256:
                gpu.pick_up(self)
                self.gpusassigned_set.add(gpu)
                if self.res_vec[0] == self.g:
                    jobqueue.running_jobs[self.id] = self
                    if self.iteration == 0:
                        jobqueue.free_jobs.pop(self.id)
        else:
            self.release_gpu(gpu)
            if self.g < self.res_vec[0] and self.id in jobqueue.running_jobs:
                jobqueue.running_jobs.pop(self.id)
            gpu.finished()
        self.update_gpu_grid()

        '''
        When a gpu is added to the slot, update the bws
        '''
        get_env().findlimitingbws()

    def release_gpu(self, gpu):
        self.gpusassigned_set.remove(gpu)

    def asset_gen(self):
        a = (127/255, 82/255, 82/255, 0.5)
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        x = [1, 2, 3, 4]
        y = [10, 20, 30, 40]
        ax.bar(x, y, color=[JOB_SIZE_RGBA, SPEED_RGBA, GPU_REQUIRMENT_RGBA,a])
        ax.set_facecolor(BACKGROUND_RGBA)


        plt.savefig('assets/current.png')
        plt.cla()
        plt.close(fig)
