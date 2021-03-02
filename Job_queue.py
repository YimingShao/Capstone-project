from Job import Job
from variable import *


class Job_queue(pygame.sprite.Sprite):
    def __init__(self, left, x_coor, y_coor):
        pygame.sprite.Sprite.__init__(self)
        self.surf = pygame.Surface((QUEUE_WIDTH, QUEUE_HEIGHT))
        self.surf.fill(BACKGROUND_COLOR)
        self.rect = self.surf.get_rect()
        self.top= None
        self.down = None
        self.left = left
        self.right = None
        self.x = x_coor
        self.y = y_coor
        self.slot = deque(maxlen=32)
        self.is_selected = False
        self.running_jobs = {}
        self.free_jobs = {}

    def insert_newjob(self):
        i = (len(self.slot) + 1) % 8
        if i == 0:
            i = 8
        j = Job(JOB_SPACE, int(JOB_SPACE * i + (i - 1) * JOB_HEIGHT))
        self.slot.append(j)
        self.free_jobs[j.id] = j


    def update(self, current_page, job_i):
        if current_page == 0:
            focus = [0, 7]
        elif current_page == 1:
            focus = [8, 15]
        elif current_page == 2:
            focus = [16, 23]
        else:
            focus = [24, 31]
        pygame.draw.rect(self.surf, BACKGROUND_COLOR, self.rect)
        self.progressing_all()
        y_coor = 1
        for i in range(focus[0], focus[1] + 1):
            if i < len(self.slot):
                self.slot[i].unselect()
                if job_i >= 0 and (current_page * 8 + job_i) < len(self.slot):
                    self.slot[current_page * 8 + job_i].selected()
                self.slot[i].update(self, int(JOB_SPACE * y_coor + (y_coor - 1) * JOB_HEIGHT))
            y_coor += 1

        if self.is_selected:
            pygame.draw.rect(self.surf, QUEUE_SELECTED_COLOR, self.rect, QUEUE_BORDER_SELECTED)
        else:
            pygame.draw.rect(self.surf, BACKGROUND_COLOR, self.rect, QUEUE_BORDER_SELECTED)
            pygame.draw.rect(self.surf, (0, 0, 0), self.rect, QUEUE_BORDER)

    def selected(self):
        self.is_selected = True

    def unselect(self):
        self.is_selected = False

    def pick_up_job(self, job_i, current_page):
        return self.slot[current_page * 8 + job_i]

    def progressing_all(self):
        has_changed = False
        remove_lst = []
        for job in self.slot:
            job.progressing()
            if job.iteration_left <= 0:
                '''
                Some jobs are finsihed, update bws
                
                call findbws once per iteration
                '''
                for gpu in job.gpusassigned_set:
                    gpu.finished()
                    has_changed = True
                remove_lst.append(job)
                self.running_jobs.pop(job.id)
        for job in remove_lst:
            self.slot.remove(job)
            self.insert_newjob()

        if get_current_job() in self.slot and has_changed:
            index = self.slot.index(get_current_job())
            set_current_page(int(index/8))
            set_job_i(index - get_current_page()*8)

        if len(remove_lst) >= 1:
            get_env().findlimitingbws()

    def lst_running_jobs(self):
        return self.running_jobs.values()

    def lst_free_jobs(self):
        return self.free_jobs.values()

    def find_job(self, id):
        for job in self.slot:
            if job.id == id:
                return job
