from Job import Job
from variable import *
import random

class Job_queue(pygame.sprite.Sprite):
    def __init__(self, left, x_coor, y_coor):
        pygame.sprite.Sprite.__init__(self)
        self.surf = pygame.Surface((QUEUE_WIDTH, QUEUE_HEIGHT))
        self.surf.fill(BACKGROUND_COLOR)
        self.rect = self.surf.get_rect()
        pygame.draw.rect(self.surf, (0,0,0), self.rect, 5)
        self.top= None
        self.down = None
        self.left = left
        self.right = None
        self.x = x_coor
        self.y = y_coor
        self.container = [[],[],[],[]]
        self.is_selected = False


    def insert(self, color, preview):
        page = int(self.calculate_total_job()/8)
        i = len(self.container[page]) + 1
        j = Job(JOB_SPACE, int(JOB_SPACE * i + (i - 1) * JOB_HEIGHT), color)
        self.container[page].append(j)


    def update(self, current_page):
        pygame.draw.rect(self.surf, BACKGROUND_COLOR, self.rect)
        self.progressing_all()
        k = -1
        for i in range(len(self.container[current_page])):
            if self.container[current_page][i].iteration_left <= 0:
                k = i
            self.container[current_page][i].update(self)
        if self.is_selected:
            pygame.draw.rect(self.surf, (0, 0, 0), self.rect, QUEUE_BORDER_SELECTED)
        else:
            pygame.draw.rect(self.surf, BACKGROUND_COLOR, self.rect, QUEUE_BORDER_SELECTED)
            pygame.draw.rect(self.surf, (0, 0, 0), self.rect, QUEUE_BORDER)

    def selected(self):
        self.is_selected = True

    def unselect(self):
        self.is_selected = False

    def select_job(self, job_i, current_page):
        if 0 <= job_i < len(self.container[current_page]):
            j = self.container[current_page][job_i]
            j.selected()

    def unselect_job(self, job_i, current_page):
        if job_i < len(self.container[current_page]):
            self.container[current_page][job_i].unselect()

    def pick_up_job(self, job_i, current_page):
        return self.container[current_page][job_i]

    def calculate_total_job(self):
        result = 0
        for i in self.container:
            result += len(i)
        return result

    def progressing_all(self):
        for i in range(len(self.container)):
            for j in range(len(self.container[i])):
                self.container[i][j].progressing()
                if self.container[i][j].iteration_left <= 0:
                    for gpu in self.container[i][j].assigned:
                        gpu.finished()
                    self.container[i].pop(j)
                    color = color_gen(n)
                    job = Job(JOB_SPACE, int(JOB_SPACE * (j + 1) + j * JOB_HEIGHT), color)
                    self.container[i].insert(j, job)