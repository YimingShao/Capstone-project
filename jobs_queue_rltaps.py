from collections import deque

import numpy as np


class Job(object):
    """
    Job object
    """

    def __init__(self, res_vec, job_len, job_id, ts_0j, d_ex, m_j,
                 tt_mj, d_fj, gradsize, modelname):
        """
        Instantiate a new Job object.

        This constructor is only called in 1 place, in the
        :py:meth:`environment_gpucluster.Env.get_new_job_from_seq` method,
        which takes
        arguments (self, seq_no, seq_idx). That method calls
        :py:meth:`parameters_gpucluster.DistDnn.dnn_model_props` method,
        which is the source of many of the Job constructors' input arguments.

        `self` below refers to an :py:class:`environment.Env` instance.

        :param res_vec: self.nw_size_seqs[seq_no, seq_idx, :]
        :param job_len: self.nw_len_seqs[seq_no, seq_idx]
        :param job_id: len(self.job_record.record)
        :param ts_0j: env.curr_time. aka 'enter time'. time_0job. 0 is meant
            to be the nought symbol. Finish time will be labelled as ts_fj. The
            difference is written as ts_f0j.
        :param d_ex: complexity of the job's dnn model in flop's
        :param m_j: minibatch size. i.e. number of examples per minibatch.
        :param tt_mj: training time of the job's dnn model in
        seconds for one minibatch.
        :param d_fj: fake computational distance to be travelled by
            having the job running on the cluster.
        """
        self.id = job_id
        # cbb. consider the first and only element of the res_vec to be the
        # minimum number of GPUs requested.
        self.res_vec = res_vec  # The first element is number of GPUs requested.
        self.len = job_len
        self.numcells = res_vec[0] * self.len

        self.slot = None  # will hold the slot that it gets put into.

        self.enter_time = ts_0j
        self.start_time = -1  # not being allocated
        self.finish_time = -1
        self.stepcounter = 0

        # V-cbb. Here we must create a field that holds info for job progress
        # V-cbb. len_left will keep track of how much longer the job has to run.
        # self.len_left = job_len
        # self.len_done = 0  # cbb. No time steps done yet.

        # self.ideal_flops = 0
        # self.actual_flops = 0

        self.d_ex = d_ex  # Model flops in Flops. distance per example
        self.m = m_j
        self.d_m = self.d_ex * self.m  # Model flops computed per minibatch
        self.gradsize = gradsize  # in MB
        self.modelname = modelname

        # V-cbb. The computational length/distance of a job should not
        # oversaturate the capacity of the input image. In reality,
        # the computational distance of a job should be independent of the
        # number of GPU's requested nw_size[j]. It should however be
        # porportional to nw_len. For simplicity, make the computational
        # distance equal nw_len[j] * nw_size[j] * d_ex.
        # Estimate the computational length of the job as the single
        # GPU FLOPS (single gpu speed, i.e. no reduction time) perfectly
        # scaled up to res_vec[0], then multiplied by the job's length.
        # Let's say 1 gpu can do 1000 x 10^a FLOPS. For simplicity,
        # disregard the 10^a as a scaling factor (b/c it CAN be algebraeically).
        self.d_f = d_fj  # formerly self.fakecompdist
        self.d_done = 0  # formerly self.compdistdone
        # self.d_rem = self.getcompdistleft()  # formerly self.compdistleft
        self.fraction_done = 0
        self.tt_m = tt_mj  # minibatch training time
        self.rt_m = 0  # minibatch reduction time
        # Below is the latest speed that will be assigned by a function such
        # as env.job_minbatch_speed_calc
        self.v_m = 0

        self.color = None
        # cbb. current number of gpus assigned. Start off with res_vec[0]. In
        # full preemption, it gets set again in Env.step() function. But
        # setting to self.res_vec[0] here is useful for the static cases.
        # self.g = self.res_vec[0]
        self.ts_togo = 0  # extrapolate forward the number of rows of
        # time needed based on currnumgpusassigned and current multispeed
        self.ts_done = 0  # extrapolate backward the number of
        # rows of time done based on currnumgpusassigned and current multispeed

        self.singlejoblimbw = np.Inf  # since a job's edgeset is associated
        # with the slot, consider associating singlejoblimbw and multijoblimbw
        # with slots instead.
        self.multijoblimbw = np.Inf
        self.scale = 1  # will be assigned
        self.gpusassigned_set = set()  # None for now but will be an np.array

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


class JobSlots(object):
    def __init__(self, jobslot):
        # cbb. pa.num_jobslots. This is the M parameter, the maximum allowed number
        #  of work in the queue. IOWs, how many queue jobslots are visible in
        # the image.
        # self.slots = [None] * pa.num_jobslots
        # self.slots = np.empty(pa.num_jobslots, dtype=object)

        self.slots = jobslot


class JobBacklog(object):
    def __init__(self, pa):
        self.backlog = deque(maxlen=pa.backlog_size)

    @property
    def curr_size(self):
        return len(self.backlog)


class JobRecord(object):
    """
    Class instance will keep a record of jobs that are put into job slots or
    backlog. The job id will be stored (I think).
    """

    def __init__(self):
        self.record = {}


class JobsRunning(object):
    """
    Class instance to hold the jobs that are currently running on the cluster.

    This is a new class by Bon. This is not needed in full pre-emption case
    since the running but unfinished jobs are left in the slots, so running
    jobs (chosen slots with existing jobs) can be obtained from gpu2slots._x
    table.
    This is useful though for static case in which it makes sense to remove
    the job from its slot after allocating resources to that job.

    Alternatively, we can keep a single dict of jobsrunning in
    """

    def __init__(self):
        self.running_jobs = {}

    def updatejobsrunning(self, actions):
        """
        Given the actions for the current step, make a new self.jobsrunning dict
        :param actions:
        :type actions:
        :return:
        :rtype:
        """