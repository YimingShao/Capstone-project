import pygame
import device
from variable import *
class Cluster(pygame.sprite.Sprite):
    def __init__(self, x_coor, y_coor):
        pygame.sprite.Sprite.__init__(self)
        self.surf = pygame.Surface((CLUSTER_WIDTH, CLUSTER_HEIGHT))
        self.surf.fill(BACKGROUND_COLOR)
        self.rect = self.surf.get_rect()
        pygame.draw.rect(self.surf, (0,0,0), self.rect, CLUSTER_BORDER)
        self.top= None
        self.down = None
        self.left = None
        self.right = None
        self.x = x_coor
        self.y = y_coor
        self.container = []
        self.gpu_all_list = []
        self.is_selected = False

        for row in range(1, 5):
            row_list = []
            row_gpu = []
            for column in range(1, 5):
                d = device.Device(int(DEVICE_SPACE_COL*column + (column-1)*DEVICE_WIDTH), DEVICE_SPACE_FROM_CLUSTER
                + int(DEVICE_SPACE_VER * row + (row - 1) * DEVICE_HEIGHT))
                self.surf.blit(d.surf, (d.x, d.y))
                row_list.append(d)
                row_gpu.extend(d.GPU_list)
            self.container.append(row_list)
            self.gpu_all_list.append(row_gpu)


    def set_up(self, top, down, left, right):
        self.top = top
        self.down = down
        self.left = left
        self.right = right

    def update(self):
        for row in self.container:
            for device in row:
                device.update(self)

        if self.is_selected:
            pygame.draw.rect(self.surf, (0, 0, 0), self.rect, CLUSTER_BORDER_SELECTED)
        else:
            pygame.draw.rect(self.surf, BACKGROUND_COLOR, self.rect, CLUSTER_BORDER_SELECTED)
            pygame.draw.rect(self.surf, (0, 0, 0), self.rect, CLUSTER_BORDER)


    def selected(self):
        self.is_selected = True
        # pygame.draw.rect(self.surf, (0,0,0), self.rect, CLUSTER_BORDER_SELECTED)

    def unselect(self):
        self.is_selected = False
        # pygame.draw.rect(self.surf, BACKGROUND_COLOR, self.rect, CLUSTER_BORDER_SELECTED)
        # pygame.draw.rect(self.surf, (0,0,0), self.rect, CLUSTER_BORDER)

    def get_gpu(self, gpu_r, gpu_c):
        return self.gpu_all_list[gpu_r][gpu_c]

    def select_gpu(self, gpu_r, gpu_c, device_c):
        self.gpu_all_list[gpu_r][gpu_c].selected()

    def unselect_gpu(self, gpu_r, gpu_c, device_c):
        self.gpu_all_list[gpu_r][gpu_c].unselect()

    def pick_up_gpu(self, gpu_r, gpu_c, id, assigned):
        self.gpu_all_list[gpu_r][gpu_c].pick_up(id)
        assigned.append(self.gpu_all_list[gpu_r][gpu_c])

    def gpu_in_job(self, gpu_r, gpu_c, assigned):
        j = self.gpu_all_list[gpu_r][gpu_c]
        for job in assigned:
            if job.x == j.x and job.y == j.y:
                return True
        return False