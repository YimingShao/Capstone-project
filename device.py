from variable import *

class GPU(pygame.sprite.Sprite):
    def __init__(self, x_coor, y_coor):
        pygame.sprite.Sprite.__init__(self)
        self.surf = pygame.Surface((GPU_WIDTH, GPU_HEIGHT))
        self.surf.fill(BACKGROUND_COLOR)
        self.rect = self.surf.get_rect()
        self.x = x_coor
        self.y = y_coor
        self.job = None
        self.is_selected = False

    def selected(self):
        self.is_selected = True

    def unselect(self):
        self.is_selected = False

    def update(self, device):
        if self.is_selected:
            pygame.draw.rect(self.surf, (0, 0, 0), self.rect, GPU_BORDER_SELECTED)
        else:
            pygame.draw.rect(self.surf, BACKGROUND_COLOR, self.rect, GPU_BORDER_SELECTED)
            pygame.draw.rect(self.surf, (0, 0, 0), self.rect, GPU_BORDER)
        device.surf.blit(self.surf, (self.x, self.y))

    def pick_up(self, job):
        if self.job != None:
            self.job.release_gpu(self)
        self.job = job
        pygame.draw.rect(self.surf, self.job.id, self.rect)

    def finished(self):
        self.job = None
        pygame.draw.rect(self.surf, BACKGROUND_COLOR, self.rect)

class Device(pygame.sprite.Sprite):
    def __init__(self, x_coor, y_coor):
        pygame.sprite.Sprite.__init__(self)
        self.surf = pygame.Surface((DEVICE_WIDTH, DEVICE_HEIGHT))
        self.surf.fill(BACKGROUND_COLOR)
        self.rect = self.surf.get_rect()
        self.x = x_coor
        self.y = y_coor
        self.GPU_list = []
        pygame.draw.rect(self.surf, (0,0,0), self.rect, DEVICE_BORDER)
        pygame.draw.circle(self.surf, (0, 0, 0), (CPU_X, CPU_Y), CPU_RADIUS, CPU_BORDER)
        pygame.draw.circle(self.surf, (0, 0, 0), (CPU_X * 3, CPU_Y), CPU_RADIUS, CPU_BORDER)
        for i in range(1, 5):
            gpu = GPU(int(GPU_SPACE * i + (i - 1) * GPU_WIDTH), GPU_SPACE_TOP)
            self.surf.blit(gpu.surf, (gpu.x, gpu.y))
            self.GPU_list.append(gpu)

    def update(self, cluster):
        for gpu in self.GPU_list:
            gpu.update(self)
        cluster.surf.blit(self.surf, (self.x, self.y))

    def selected(self):
        pygame.draw.rect(self.surf, RACK_SELECTED_COLOR, self.rect, DEVICE_BORDER_SELECTED)

    def unselect(self):
        pygame.draw.rect(self.surf, BACKGROUND_COLOR, self.rect, DEVICE_BORDER_SELECTED)
        pygame.draw.rect(self.surf, (0,0,0), self.rect, DEVICE_BORDER)