from variable import *
import pygame
import random

class Job(pygame.sprite.Sprite):
    def __init__(self, x_coor, y_coor, identification):
        self.is_selected = False
        pygame.sprite.Sprite.__init__(self)
        self.surf = pygame.Surface((JOB_WDITH, JOB_HEIGHT))
        self.surf.fill(BACKGROUND_COLOR)
        self.rect = self.surf.get_rect()
        self.x = x_coor
        self.y = y_coor
        self.speed = random.randint(1,3)
        self.job_size = random.randint(int(JOB_SIZE_HEIGHT/10), JOB_SIZE_HEIGHT)
        self.gpu_requirement = random.randint(1,4)
        self.iteration = 0
        self.iteration_left = 2000
        self.id = identification
        self.assigned = []
        self.is_running = False

        pygame.draw.rect(self.surf, (0,0,0), self.rect, JOB_BORDER)
        id_rect = pygame.Rect(ID_RECT_X, ID_RECT_Y, ID_RECT_WIDTH, ID_RECT_WIDTH)

        job_size_outer = pygame.Rect(JOB_SIZE_X, JOB_SIZE_Y, JOB_SIZE_WIDTH, JOB_SIZE_HEIGHT)
        job_size_inner = pygame.Rect(JOB_SIZE_X, JOB_SIZE_MAX-self.job_size, JOB_SIZE_WIDTH, self.job_size)

        pygame.draw.rect(self.surf, self.id, id_rect)
        pygame.draw.rect(self.surf, (0,0,0), id_rect, JOB_SIZE_BORDER)
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


    def update(self, queue):
        self.update_gpu_requirement_bar()
        if len(self.assigned) < self.gpu_requirement:
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
        else:
            pass
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
        for i in range(1, self.gpu_requirement + 1):
            inner = pygame.Rect(GPU_REQUIRE_X + (i - 1) * GPU_REQUIRE_CUBE_WIDTH, GPU_REQUIRE_Y, GPU_REQUIRE_CUBE_WIDTH,
                                    GPU_REQUIRE_CUBE_HEIGHT)
            pygame.draw.rect(self.surf, GPU_REQUIRMENT_COLOR, inner)
            if i <= len(self.assigned):
                pygame.draw.rect(self.surf, (206, 227, 43), inner, GPU_REQUIRE_BORDER)
            else:
                pygame.draw.rect(self.surf, (0, 0, 0), inner, GPU_REQUIRE_BORDER)
        for i in range(self.gpu_requirement + 1, len(self.assigned) + 1):
            inner = pygame.Rect(GPU_REQUIRE_X + (i - 1) * GPU_REQUIRE_CUBE_WIDTH, GPU_REQUIRE_Y, GPU_REQUIRE_CUBE_WIDTH,
                                GPU_REQUIRE_CUBE_HEIGHT)
            pygame.draw.rect(self.surf, self.id, inner)
            pygame.draw.rect(self.surf, (0, 0, 0), inner, GPU_REQUIRE_BORDER)

    def pick_gpu(self, gpu):
        if not gpu in self.assigned:
            if len(self.assigned) < 10:
                gpu.pick_up(self)
                self.assigned.append(gpu)
        else:
            self.release_gpu(gpu)
            gpu.finished()
    def release_gpu(self, gpu):
        self.assigned.remove(gpu)