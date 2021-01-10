from variable import *

class Side_view(pygame.sprite.Sprite):
    def __init__(self, x_coor, y_coor, slot):
        self.x = x_coor
        self.y = y_coor
        self.surf = pygame.Surface((SIDE_VIEW_WIDTH, SIDE_VIEW_HEIGHT))
        self.surf.fill(BACKGROUND_COLOR)
        self.rect = self.surf.get_rect()
        self.container = []

        for i in range(1, 5):
            page = Preview_page(PREVIEW_SPACE, i * PREVIEW_SPACE + (i - 1) * PREVIEW_HEIGHT, [(i-1)*8, i*8], slot)
            self.container.append(page)
            self.surf.blit(page.surf, (page.x, page.y))

    def update(self, current_page):
        self.container[current_page].select()
        pygame.draw.rect(self.surf, BACKGROUND_COLOR, self.rect)
        for view in self.container:
            view.update(self)


class Preview_page(pygame.sprite.Sprite):
    def __init__(self, x_coor, y_coor, focus, slot):
        self.focus = focus
        self.slot = slot
        self.x = x_coor
        self.y = y_coor
        self.surf = pygame.Surface((PREVIEW_WIDTH, PREVIEW_HEIGHT))
        self.surf.fill(BACKGROUND_COLOR)
        self.rect = self.surf.get_rect()
        self.is_selected = False


    def update(self, side_view):
        if self.is_selected:
            pygame.draw.rect(self.surf, (0, 0, 0), self.rect, PREVIEW_BORDER_SELECTED)
            self.is_selected = False
        else:
            pygame.draw.rect(self.surf, BACKGROUND_COLOR, self.rect, PREVIEW_BORDER_SELECTED)
            pygame.draw.rect(self.surf, (0, 0, 0), self.rect, PREVIEW_BORDER)
        p = 1
        for i in range(self.focus[0]+1, self.focus[1]+1):
            if len(self.slot) >= i:
                j = pygame.Rect(PREVIEW_JOB_SPACE, p * PREVIEW_JOB_SPACE + (p-1) * PREVIEW_JOB_HEIGHT, PREVIEW_JOB_WIDTH, PREVIEW_JOB_HEIGHT)
                pygame.draw.rect(self.surf, self.slot[i-1].id, j)
            p += 1
        side_view.surf.blit(self.surf, (self.x, self.y))

    def select(self):
        self.is_selected = True
