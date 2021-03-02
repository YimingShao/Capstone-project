import numpy as np
import math
import matplotlib.pyplot as plt

# parameters_gpucluster = imp.load_source('parameters_gpucluster',
#                                         'parameters.py')
import parameters

# from scipy.optimize import curve_fit

import time
import networkx as nx

from assignment_table import G2J
from cluster_rltaps import Cluster
from common.debug.debug import debuginfo
import job_distributions

from abc import ABCMeta, abstractmethod

from jobs_queue_rltaps import Job, JobBacklog, JobRecord


class Env(object):
    """
    """
    __metaclass__ = ABCMeta

    def __init__(self, pa, snc, jobqueue, nw_len_seqs=None, nw_size_seqs=None,
                 seed=42, render=False, repre='image',
                 end='no_more_jobs_in_seq'):

        self.pa = pa
        self.render = render
        self.repre = repre  # image or compact representation
        self.end = end  # termination type, 'no_more_jobs_in_seq' or
        # 'do_all_jobs_in_seq'. Former names: 'no_new_jobs' and 'all_done'
        # add to termination type, 'not_done_yet'

        # V-cbb. This should be fine. env.nw_dist_func is not used anywhere
        self.nw_dist_func = pa.distobj.new_model_dist_func
        # pa.distobj.bi_model_dist_func

        self.curr_time = 0

        self.status = None  # will be one of 1: Allocate, 2: Add,
        # 3: MoveOn

        # set up random seed
        if seed is None:
            # cbb. Modify the below statement to investigate if doing
            # np.random.seed(), i.e. reset the seed, will lead to better
            # training due to more randomness.
            # np.random.seed(314159)
            np.random.seed()
        else:
            np.random.seed(seed)

        if nw_len_seqs is None or nw_size_seqs is None:
            # generate new work
            # cbb. This should be okay for gpucluster environment.
            self.nw_len_seqs, self.nw_size_seqs = \
                job_distributions.generate_sequence_work(
                    self.pa, seed=seed)

            self.workload = np.zeros(pa.num_res_types)
            for i in range(pa.num_res_types):
                # cbb. do element wise multiplication of nw_size_seqs
                # and nw_len_seqs ndarray tables. I think it should be
                # nw_size_seqs[:,:,i] below instead of nw_size_seqs[:,i]
                # cbb. as noted in the paper, calculate the workload.
                # cbb. this should be okay as is for gpucluster environment.
                # cbb. normalized for number of cluster gpus, and time horizon.
                self.workload[i] = \
                    np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
                    float(pa.num_clustergpus) / \
                    float(len(self.nw_len_seqs))
                print("Load on # " + str(i) + " resource dimension is " + str(
                    self.workload[i]))
            self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                          [self.pa.num_jobsets,
                                           self.pa.t_finaljobarrival])
            self.nw_size_seqs = np.reshape(self.nw_size_seqs,
                                           [self.pa.num_jobsets,
                                            self.pa.t_finaljobarrival,
                                            self.pa.num_res_types])
        else:
            self.nw_len_seqs = nw_len_seqs
            self.nw_size_seqs = nw_size_seqs

        self.seq_no = 0  # which example sequence. <-cbb. Max is pa.num_jobsets
        self.seq_idx = 0  # index in that sequence

        # temporary place to hold negslowdown for each time step of an episode
        # Must be updated when a job is done finished. May need to combine
        # the two below into a class for easy resetting.
        self.negslowdown = 0.0
        self.removedjobscount = 0
        self.extrapenalties = 0.0

        # initialize simple network class instance.
        self.snc = snc

        # cbb. This constructor should be okay for gpucluster environment.
        # cbb. Needed, but your Cluster class might be different.
        self.cluster = Cluster(pa)

        # cbb. This class should be okay for gpucluster environment.
        # Todo-yshao. Try to use it.
        self.job_backlog = JobBacklog(pa)

        # cbb. This class should be okay for gpucluster environment.
        # Todo-yshao. Keep. Used for slowdown. Not mandatory for first game
        #  env prototype.
        self.job_record = JobRecord()
        # cbb. This class should be okay for gpucluster environment.
        # Todo-yshao. Used for extra info input into the neural network. Not
        #  mandatory for first game env prototype.
        self.extra_info = ExtraInfo(pa)
        # cbb. Data structure to hold gpu to job assignments.
        # Todo-yshao. Mandatory
        self.g2j = G2J(pa, jobqueue)
        # cbb. In case some of Mao's code needs env to have job_slots field.
        # Todo-yshao. Mandatory
        self.job_slots = self.g2j.job_slots

    # class Env
    def cluster_time_proceed(self, curr_time):
        """
        Moved from cluster class to env class.
        Called by env.step. When the environment needs to take a step,
        shift the .numavbl_res matrix up by one time step, and also move the
        cluster resource image representation up by one time step.

        This function also checks the cluster.startednotdone_jobs list to see if any
        jobs have 'finished', i.e, their finish time has been reached,
        and if so, remove the job from the cluster.startednotdone_jobs list. In our
        case, we must also remove the job from the job slot, and dequeue the
        backlog onto an available slot (probably onto the slot we just
        emptied is easiest, rather than shifting all the jobs over). This
        should be done here, instead of in the env.step() function. Mao
        dequeues from the backlog to the newly empty slot in env.step()
        because Mao empties a slot during the allocation phase, which is done
        in env.step(), unlike us.

        :param curr_time: The environment's current time step.
        :return: Nothing
        """

        # shift cluster.avgl_slot up by one. In Mao's code,
        # this automatically takes care of updating numavbl_res after a job is
        # done because he does static resource assignment. We cannot rely on
        # this for full pre-emption case.
        self.cluster.avbl_res_moveupone()

        for jid, job in self.g2j.startednotdone_jobs.items():

            if job.finish_time <= curr_time:
                self.g2j.startednotdone_jobs.pop(jid)
                # ^-cbb. Carefull!!! list.remove(object/literal) removes the
                # first occurence of object or literal, and shifts the right
                # hand side of the array to the left by one index.

        # update graphical representation
        # cbb. shift the canvas "image" up by one.
        self.cluster.canvas[:, :-1, :] = self.cluster.canvas[:, 1:, :]
        # cbb. Last time step should be empty after the shift.
        self.cluster.canvas[:, -1, :] = 0

    # class Env
    def get_new_job_from_seq(self, seq_no, seq_idx, testjob=False):
        # Editing

        # V-cbb. Since job's nw_len, nw_size are independent (i.e. no logical
        #  relation) for now, just get choose a random common DNN model,
        # and get the model_size and training time.

        job_len = self.nw_len_seqs[seq_no, seq_idx]
        job_size = self.nw_size_seqs[seq_no, seq_idx, :][0]

        testjobmodel = None
        testjobbatchsize = None
        if testjob and self.pa.testjobmodel:
            testjobmodel = self.pa.testjobmodel
            testjobbatchsize = self.pa.testjobbatchsize

        d_exj, m_j, tt_mj, d_fj, gradsize, modelname = \
            self.pa.distobj.dnn_model_props(job_len, job_size,
                                            testjobmodel=testjobmodel,
                                            testjobbatchsize=testjobbatchsize)

        # V-cbb. job_id=len(self.job_record.record) always ensures that a new
        #  unique job_id is created.
        new_job = Job(res_vec=self.nw_size_seqs[seq_no, seq_idx, :],
                      job_len=self.nw_len_seqs[seq_no, seq_idx],
                      job_id=len(self.job_record.record),
                      ts_0j=self.curr_time,
                      d_ex=d_exj,
                      tt_mj=tt_mj, m_j=m_j,
                      d_fj=d_fj,
                      gradsize=gradsize,
                      modelname=modelname)
        return new_job

    # class Env
    def get_new_job_from_seq_orig(self, seq_no, seq_idx):
        new_job = Job(res_vec=self.nw_size_seqs[seq_no, seq_idx, :],
                      job_len=self.nw_len_seqs[seq_no, seq_idx],
                      job_id=len(self.job_record.record),
                      ts_0j=self.curr_time)
        return new_job

    # class Env
    def make_new_jobsets(self, seed=None):
        """
        Like creating new datasets.
        :param seed:
        :type seed:
        :return:
        :rtype:
        """
        self.nw_len_seqs, self.nw_size_seqs = \
            job_distributions.generate_sequence_work(
                self.pa, seed, distobj_dist_func=None)

    def debug_orno(self, dictin):
        debugthis = False
        if debugthis and self.pa.debug_:
            if self.pa.debug_firstlastiter:
                if 'iter' in dictin and dictin['iter'] is not None:
                    if dictin['iter'] == 1 or \
                            dictin['iter'] == self.pa.num_epochs:
                        debugthis = True
        return debugthis

    def debug_slowdown_get_single_traj(self):
        debugthis = False
        debugthis = debugthis and self.pa.debug_dict['in_slowdown'] \
                    and self.pa.debug_slowdown
        if debugthis and self.pa.debug_:
            if self.pa.debug_dict['sched_type'] == 'Random' and (
                    self.pa.debug_dict['seq_idx'] + 1) == 6:
                debugthis = True
            else:
                debugthis = False
            if debugthis:  # if all conditions met, then finally use debuginfo()
                if self.curr_time % 10 == 0 or self.curr_time >= 430:
                    print("Debugging. env.curr_time: " + str(self.curr_time))
                if self.curr_time == 439:
                    pass  # put breakpoint here
        return debugthis

    def debug_step_jobspeed(self, job):
        debug = True
        if debug and self.pa.debug_step_jobspeed:
            debug = debug and ~ self.pa.debug_dict.get('in_slowdown', False)
            debug = debug and self.pa.debug_dict.get('in_pg_re_step', False)
            debug = debug and self.debug_step_firstlastiter_lastjobset_selecteps()
        else:
            debug = False
        if debug:
            prefix = self.debugjobinfo_makeprefix()
            debugstr = self.debugjobinfo_makestr(job)
            if job.g == 1:
                debuginfo(prefix + " Single GPU in Progress." + debugstr,
                          stackupby=1)
            else:
                debuginfo(prefix + " Multi GPUs in Progress." + debugstr,
                          stackupby=1)
                # check if ts_tgo and ts_done for job agress with job.d_done
                ts_togo, ts_done = self.pa.distobj.jobnumrowstime(job)
                debuginfo(("Compare ts_done/job.job_len: {:.4f}, "
                           "d_done / job.d_f: {:.4f}").format(
                    ts_done * 1.0 / (ts_togo + ts_done),
                    job.d_done / job.d_f), stackupby=1)

    def debugjobinfo_makeprefix(self):
        debugstr = ""
        if self.pa.debug_dict['iter'] is not None:
            debugstr = "I: %d " % self.pa.debug_dict['iter']
        if self.pa.debug_dict['jobseti'] is not None:
            debugstr = debugstr + (" JS: %d" % self.pa.debug_dict['jobseti'])
        if self.pa.debug_dict['episode'] is not None:
            debugstr = debugstr + (" E: %d" % self.pa.debug_dict['episode'])
        return debugstr

    def debugjobinfo_makestr(self, job):
        debugstr = (" j.v_m: {:.3e}, " +
                    "v_c: {:.3e}, tt_mc: {:.3f} " +
                    "j.g: {:d}, j.scale: {:.1f}, " +
                    "j.singlejoblimbw: {:.3f}, " +
                    "j.multijoblimbw: {:.5f}, j.tt_m: {:.3f}, " +
                    "j.rt_m: {:.3f}, " +
                    "j.modelname: {:s}").format(
            job.v_m, self.pa.distobj.v_c, self.pa.distobj.tt_mc,
            job.g, job.scale,
            job.singlejoblimbw, job.multijoblimbw, job.tt_m,
            job.rt_m, job.modelname)
        return debugstr

    def debug_step_firstlastiter_lastjobset_selecteps(self):
        debug = (self.pa.debug_dict['iter'] == 1 or
                 self.pa.debug_dict['iter'] == self.pa.num_epochs or
                 (self.pa.debug_everyoutputfreq and
                  self.pa.debug_dict['iter'] is not None and
                  (self.pa.debug_dict['iter'] % self.pa.output_freq) == 0))
        debug = debug and \
                (self.pa.debug_dict['jobseti'] == self.pa.num_jobsets)
        if debug:
            if self.pa.debug_dict['in_slowdown']:
                debug = True
            elif self.pa.debug_userandomepisodevector and \
                    self.pa.pg_resume_iter is None:
                # iter, jobseti, episode are 1-based indexed
                episodei = ((self.pa.debug_dict['iter'] - 1) *
                            self.pa.num_jobsets +
                            self.pa.debug_dict['jobseti'] - 1)
                debug = self.pa.debug_dict['episode'] == \
                        self.pa.debug_randomepisodes[episodei]
        else:
            debug = False

        return debug

    def debug_step_firstjobinfo(self, job):
        debug = self.pa.debug_step_firstjobinfo and \
                self.debug_step_firstlastiter_lastjobset_selecteps()
        if debug:
            self.debugjobinfo(job, stackupby=2)
        return debug

    def debug_step_removejob_firstjobinfo(self, job):
        debug = self.pa.debug_step_removejob_firstjobinfo and \
                self.debug_step_firstlastiter_lastjobset_selecteps()
        if debug:
            self.debugjobinfo(job, stackupby=2)
        return debug

    def debugjobinfo(self, job, stackupby=1):
        prefix = self.debugjobinfo_makeprefix()

        debugstr = ("j.id: {:d}, d_done: {:.1E}, "
                    "d_f: {:.1E}, "
                    "fraction: {:.3f}, "
                    "env.curr_time: {:d}, "
                    "j.stepcounter: {:d}, "
                    "j.g: {:d}, "
                    "j.singlejoblimbw: {:.2f}, "
                    "j.multijoblimbw: {:.2f}, "
                    "j.scale: {:.1f}, j.m: {:d}, j.modelname: {:s}").format(
            job.id, job.d_done, job.d_f,
            job.d_done / job.d_f, self.curr_time, job.stepcounter,
            job.g, job.singlejoblimbw,
            job.multijoblimbw, job.scale, job.m, job.modelname)
        debuginfo(prefix + " " + debugstr, stackupby=stackupby)

    def output_debug_actions_oneepis_onejobset_shared(self, stackupby=2):
        """
        This will be used in various step functions. env.step() is for only
        PG case. env.step_static_oneaction() is for one action; Random, Tetris, or SJF
        case.
        :return:
        :rtype:
        """
        debug = self.pa.debug_actions and \
                self.debug_step_firstlastiter_lastjobset_selecteps()

        if debug:
            if self.pa.debug_dict['sched_type'] in self.pa.allrl_scheds:

                debugstr = ("step: {:d}, env.curr_time: {:d}, "
                            "sched_type: {:s}, Status: {:s}, "
                            "running: {:s}, "
                            "slots: {:s}".format(
                    self.pa.debug_dict['step'], self.curr_time,
                    self.pa.debug_dict['sched_type'],
                    self.pa.debug_dict['status'],
                    self.pa.debug_dict['ngpus'],
                    str(self.pa.debug_dict['slots'])))
            else:  # Random, Tetris, or SJF
                debugstr = ("step: {:d}, env.curr_time: {:3d}, "
                            "sched_type: {:s}, Status: {:s}, "
                            "action: {:d}, "
                            "running: {:s}, "
                            "slots: {:s}".format(
                    self.pa.debug_dict['step'], self.curr_time,
                    self.pa.debug_dict['sched_type'],
                    self.pa.debug_dict['status'],
                    self.pa.debug_dict['action'],
                    self.pa.debug_dict['ngpus'],
                    str(self.pa.debug_dict['slots'])))
            debuginfo(debugstr, stackupby=stackupby)

    # class Env
    def observelstm1(self):
        """
        Return states, one for job slots and one for gpus
        :return:
        :rtype:
        """

        # Attribute: js-index. Will hold indices of job slots with jobs.
        # Done. Initialize jobslotswithjobs
        # Done. Populate jobslotswithjobs
        # assert that we aren't trying to call the observe function when
        # there are no jobs in the jobslots. If however, we define how to do
        # NN forward pass when no jobs exists then we can call this observe fxn.
        assert self.g2j.jobslotvec.size != 0
        if len(self.job_slots.slots) == 0: # no jobs in deque slots
            jobslotswithjobs = np.array([[self.pa.num_jobslots]], dtype=np.int8)

        else:
            jobslotswithjobs = np.expand_dims(np.flatnonzero(
                self.g2j.jobslotvec).astype('int8'),1)
        # cbb: Done. first try numbers only.
        # Todo-cbb: Later: try env.slots object that is ndarry that holds
        #  objects

        # Attribute: js-bin vector. Will tell us which GPUs that job in js is
        # already using.
        # Done for now. Initialize the array with zeros.
        # LHS: js2gpus_bin.shape should=(num non empty jobslots, numclustergpus)
        js2gpus_binmat = np.zeros((len(jobslotswithjobs), self.g2j.x.shape[0]),
                                  dtype=np.int8)
        # ^-cbb: Do further below: populate the array.

        # Attribute: jobattribs_mat. Will hold the attributes of a job. Will
        # correspond to the job slot index specified by jobslotswithjobs.
        # v-cbb: DONE. Build up the attributes of jobs in the jobslot. Need the
        # following attributes: enter_time, start_time, d_f, d_done, v_m, d_ex,
        # m, gradsize
        # ToDo-cbb: Do Later. Try out these other job attributes:
        # Done. Initialize jobattribs_mat
        # Done. Populate jobattribs_mat
        jobattribs_mat = np.zeros((len(jobslotswithjobs), 8), dtype=np.float32)
        for i in range(len(jobslotswithjobs)):  #si in jobslotswithjobs:
            # v-cbb: DONE. build up the gs2gpus_binmat, each row being a
            # gs2gpus_binvec
            sloti = jobslotswithjobs[i]
            js2gpus_binvec = self.g2j.x[:, sloti].T
            js2gpus_binmat[i, :] = js2gpus_binvec

            if np.squeeze(sloti) != self.pa.num_jobslots:
                job = self.job_slots.slots[np.squeeze(sloti)]

                # build up the job
                jobattribs_mat[i, 0] = job.enter_time
                jobattribs_mat[i, 1] = job.start_time
                jobattribs_mat[i, 2] = job.d_f
                jobattribs_mat[i, 3] = job.d_done
                jobattribs_mat[i, 4] = job.v_m
                jobattribs_mat[i, 5] = job.d_ex
                jobattribs_mat[i, 6] = job.m
                jobattribs_mat[i, 7] = job.gradsize

        # Attribute: gpuindices. Will give vector of GPUs that are available
        # for assigning to jobs in jobslots. One gpu per decoder step.
        # For now use all GPUs
        # Done for now. Initialize AND Populate the array with range() from 0
        gpuindices = np.arange((self.g2j.x.shape[0]), dtype=np.int8)

        # Attribute: gpu2js_int. This will eventually be a column vector,
        # each element telling us which jobslot the current gpu is assigned to.
        # Done. Initialize gpu2js_int with zeros
        # Todo: Later. If a GPU can be assigned to MULTIPLE jobs,
        #  gpu2js_int will have to be a matrix, each row being a binary
        #  vector IFF a gpu can be assigned to multiple jobs.
        gpu2js_int = np.zeros((self.g2j.x.shape[0]), dtype=np.int8)  # a vector
        # ^-cbb. DONE below: populate gpu2js_int

        # Attribute: gpua2gpub_mat. Each row will be a vector that tells us
        # the limiting bw between GPUa and GPUb.
        # Done. Initilize with zeros.
        gpua2gpub_mat = np.zeros((self.g2j.x.shape[0], self.g2j.x.shape[0]),
                                 dtype=np.float32)
        # ^-cbb: DONE below: populate gpua2gpub_mat.

        # for now, let's just send the non zero (or true) column of g2j._x
        # for each row. Thus for each gpu, we record it's current
        # assigned jobset, even if there is no job there. The only rational
        # for doing so is that we can give the NN a sense of what jobset it
        # was previously assigned to. Alternative we can later try updating
        # g2j._x so that if a True value for a gpu row is at an index without a
        # job, then we move the True value to the end of the row to the null
        # jobslot.

        # v-cbb. Below, we will populate gpu2js_int and gpua2gpub_mat
        for a in gpuindices:  # gpuindices go from 0 to numclustergpus for now
            # First populate gpu2js_int
            # find out which js gpu g is assigned to, if no job then, assign
            # null jobslot
            slotis = np.flatnonzero(self.g2j.x[a,:])
            # v-cbb. if below throws error, then problem with actions vector.
            assert slotis.size != 0  # there should be at least a single 1
            # v-cbb. Done: Just using the gpu2js assignment, even if
            # no job in that js exists.
            # Todo-cbb. Think about whether if no job exists at sloti then
            #  should we reassign 1 from sloti to null jobslot?
            gpu2js_int[a] = slotis[0]  # DONE. here populate gpu2js_int

            sloti = slotis[0]  # get the first non zero
            # we need to take into account times when the assignment table
            # updated by g2j.x_update_allgpus and the allocations are not
            # synched. This will happen for full preemption when despite all
            # rows (gpus) of the assignment table at the beginning of each
            # env.step being updated according to the actions, there is no
            # job in the jobslot.

            # Currently, I update all rows of _x every step, and udpate gpu
            # rows of _x when they are release by finished jobs.

            # Todo. Maybe uncomment below to try moving chosen slots without
            #  jobs to the null slot (for each gpu).
            # job = self.job_slots.slots[sloti]
            # if job is None:
            #     # move sloti to the null job slot
            # else:
            #     assert g in job.gpusassigned_set
            # if g in job.gpusassigned_set:
            #     gpu2js_int[g] = sloti
            # else:  # gpu g is not currently assigned
            #     gpu2js_int[g] = len(self.g2j.x.shape[1])  # set to
            #     # nulljobslot

            # v-cbb. Here we populate gpua2gpub_mat
            # gpuindices go from 0 to numclustergpus for now
            for b in gpuindices:
                if a < b:  # only look at unique pairs. Fill the upper triangle
                    # now populate the speed of gpua to all other gpus
                    # first get the graph node int for current gpu g
                    anodeid = self.snc.xrowi2gpunodeid[a]
                    bnodeid = self.snc.gpunodeid2xrowi[b][0]
                    astr = self.snc.gintstr[anodeid]
                    bstr = self.snc.gintstr[bnodeid]
                    gpupairstr = ','.join([astr, bstr])
                    path = self.snc.shortpaths[gpupairstr]  # list of numbers

                    a2bspeed = np.Inf  # temp inf speed between a and b
                    for i in range(len(path) - 1):
                        # we want edge as a tuple of integers
                        edge = (path[i], path[i + 1]) if path[i] < path[
                            i + 1] else (path[i + 1], path[i])
                        multispeed = self.snc.g1.edges[edge]['speed'] / 1.0
                        if self.snc.g1.edges[edge]['jobcount'] > 0:
                            multispeed /= self.snc.g1.edges[edge]['jobcount']
                        if multispeed < a2bspeed:
                            a2bspeed = multispeed
                    assert a2bspeed != np.Inf
                    # ^-cbb. Note: If we don't need multispeed, then we can
                    # build the min speeds between gpus (without any jobs
                    # running in the cluster), just once.

                    # Done. Populate gpua2gpub_mat speeds between gpus.
                    gpua2gpub_mat[a, b] = a2bspeed
                elif a == b: # same gpu. i.e. diagonals
                    pass  # diagonals are already zero

        # transpose and add to reflect the lower triangle
        gpua2gpub_mat = gpua2gpub_mat + gpua2gpub_mat.T  # diags are zeros

        # Done. now output the observation as a single variable that is a
        # list of ndarrays.
        enc_state = np.concatenate((jobslotswithjobs, js2gpus_binmat,
                                    jobattribs_mat), axis=1, )
        gpuindices = np.expand_dims(gpuindices, axis=1)
        gpu2js_int = np.expand_dims(gpu2js_int, axis=1)
        dec_state = np.concatenate((gpuindices, gpu2js_int, gpua2gpub_mat),
                                   axis=1)
        state = [enc_state, dec_state]
        return state  # return to the env.step function.

    # class Env
    def observe(self):
        """
        Return a image representation of the state. The output is a table.

        The table is built from left to right, starting with copying the
        cluster's canvas, then building up the job slot representations. This is
        repeated left to right for each additional resource type. then finally
        'extra_info' is appended to the right. The output is not built from
        top down. I.e. [resources of resource type 1, job slots for resource
        type 1] is left concatenated to [resource of resource type 2, job slots
        for esource type 2], which is left concatenated to [extra_info].

        :return image_repr: "image" representation of the environment.
        """
        if self.repre == 'image':

            backlog_width = int(
                math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))

            image_repr = np.zeros(
                (self.pa.network_input_height, self.pa.network_input_width))

            ir_pt = 0

            for i in range(self.pa.num_res_types):

                image_repr[:,
                ir_pt: ir_pt + self.pa.num_clustergpus] = self.cluster.canvas[i,
                                                         :, :]
                ir_pt += self.pa.num_clustergpus

                for j in range(self.pa.num_jobslots):

                    if j < len(self.job_slots.slots) \
                            and self.job_slots.slots[j] is not None:
                        # fill in a block of work
                        image_repr[: self.job_slots.slots[j].len,
                        ir_pt: ir_pt + self.job_slots.slots[j].res_vec[i]] = 1

                    ir_pt += self.pa.max_job_size
            # cbb. up to and excluding rows
            # job_backlog.curr_size/backlog_width,
            # fill image with 1's (of course fill the 3 columns.
            # if there are 1 or 2 more in the backlog, then also fill 1 or 2
            # more squares in the first empty row of the backlog area.
            image_repr[: int(self.job_backlog.curr_size / backlog_width),
            ir_pt: ir_pt + backlog_width] = 1
            if self.job_backlog.curr_size % backlog_width > 0:
                image_repr[int(self.job_backlog.curr_size / backlog_width),
                ir_pt: ir_pt + self.job_backlog.curr_size % backlog_width] = 1
            ir_pt += backlog_width  # cbb. move to the right to the
            # extra_info columns.

            image_repr[:,
            ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_job / \
                                float(
                                    self.extra_info.max_tracking_time_since_last_job)
            ir_pt += 1
            # cbb. got to the right most side of the image.
            assert ir_pt == image_repr.shape[1]

            return image_repr

        else:
            raise ValueError("env.repre must be 'image' or 'compact'")

    # class Env
    def plot_state(self):
        plt.figure("screen", figsize=(20, 5))

        skip_row = 0

        for i in range(self.pa.num_res_types):

            plt.subplot(self.pa.num_res_types,
                        1 + self.pa.num_jobslots + 1,
                        # first +1 for current work, last +1 for backlog queue
                        i * (
                                    self.pa.num_jobslots + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

            plt.imshow(self.cluster.canvas[i, :, :], interpolation='nearest',
                       vmax=1)

            for j in range(self.pa.num_jobslots):

                job_slot = np.zeros(
                    (self.pa.time_horizon, self.pa.max_job_size))
                if self.job_slots.slots[
                    j] is not None:  # fill in a block of work
                    job_slot[: self.job_slots.slots[j].len,
                    :self.job_slots.slots[j].res_vec[i]] = 1

                plt.subplot(self.pa.num_res_types,
                            1 + self.pa.num_jobslots + 1,
                            # first +1 for current work, last +1 for backlog queue
                            1 + i * (
                                    self.pa.num_jobslots + 1) + j + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

                plt.imshow(job_slot, interpolation='nearest', vmax=1)

                if j == self.pa.num_jobslots - 1:
                    skip_row += 1

        skip_row -= 1
        backlog_width = int(
            math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
        backlog = np.zeros((self.pa.time_horizon, backlog_width))

        backlog[: self.job_backlog.curr_size / backlog_width,
        : backlog_width] = 1
        backlog[self.job_backlog.curr_size / backlog_width,
        : self.job_backlog.curr_size % backlog_width] = 1

        plt.subplot(self.pa.num_res_types,
                    1 + self.pa.num_jobslots + 1,
                    # first +1 for current work, last +1 for backlog queue
                    self.pa.num_jobslots + 1 + 1)

        plt.imshow(backlog, interpolation='nearest', vmax=1)

        plt.subplot(self.pa.num_res_types,
                    1 + self.pa.num_jobslots + 1,
                    # first +1 for current work, last +1 for backlog queue
                    self.pa.num_res_types * (
                            self.pa.num_jobslots + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

        extra_info = np.ones((self.pa.time_horizon, 1)) * \
                     self.extra_info.time_since_last_new_job / \
                     float(self.extra_info.max_tracking_time_since_last_job)

        plt.imshow(extra_info, interpolation='nearest', vmax=1)

        plt.show()  # manual
        # plt.pause(0.01)  # automatic

    # class Env
    def add_to_perstep_slowdown(self, jobs):
        # type: (list) -> None
        """
        Called by env.remove_done_jobs and env.remove_done_jobs_static
        :param jobs:
        :type jobs: list
        :return:
        :rtype:
        """
        for job in jobs:
            self.removedjobscount += 1
            negjobslowdown = - float(job.finish_time - job.enter_time) / job.len
            weightedprevnegslowdown = float(self.negslowdown) * (
                    self.removedjobscount - 1)
            assert weightedprevnegslowdown <= 0
            numerator = weightedprevnegslowdown + negjobslowdown * 1.0  # weighted sum
            # divide by sume of weights
            assert numerator <= 0 and self.removedjobscount > 0
            self.negslowdown = numerator / self.removedjobscount

    def get_costofchoosingemptyslot(self, ngpus=1):
        """
        If an empty slot is chosen, will return the unit cost of choosing the
        empty slot. So that it can be multiplied by the number of gpus assigned
        to the empty slot.
        :return: a penalty that is <= 0
        :rtype:
        """
        extrapenalty = self.get_unitcostofbadchoice()

        assert extrapenalty <= 0
        return extrapenalty * ngpus * 1.0

    def get_unitcostofbadchoice(self):
        if self.pa.rewardname == 'negslowdown':
            extrapenalty = self.pa.negslowdownpergpucost
        else:
            # assume reward type is throughput
            # can't use job's single gpu speed, because there is no job. So
            # just use single gpu speed.
            extrapenalty = - self.pa.p100flops_pers

        assert extrapenalty <= 0
        return extrapenalty

    def get_reward_fullpreempt(self):
        """
        reward function for full pre-emption.

        :return:
        :rtype:
        """

        reward = 0
        extrapenalties = 0
        gpusshort = 0
        if self.pa.rewardname == 'negslowdown':
            reward = self.negslowdown
        else:
            # instead of looking at started jobs, some of which may be paused
            # if not allocated during the current step, look instead at
            # g2j.chosenjobs_slots_int
            for s in self.g2j.job_queue.lst_running_jobs():  # startednotdone_jobs is a
            # dict
                # cbb. running but not yet finished.
                # reward += self.pa.delay_penalty / float(j.len)
                # cbb. continue here
                j = self.g2j.sloti2job(s.id)
                reward += j.v_m

            # cbb. Previously, Mao looked at not yet running jobs that are just
            # held in the visible queue, in the M slots. Instead, we want
            # to looked at the g2j.dismissedjobs_slots_int
            for s in self.g2j.job_queue.lst_free_jobs():
                j = self.g2j.sloti2job(s.id)
                # instead of using hold_penalty, get singlejob speed
                reward -= self.job_minbatch_speed_calc(j, 'single') * \
                          self.pa.penaltyfactor2
                # reward += self.pa.hold_penalty / float(j.len)

            # cbb. not yet running, nor in visible queue, in backlog.
            for j in self.job_backlog.backlog:
                reward -= self.job_minbatch_speed_calc(j, 'single')
                # reward += self.pa.dismiss_penalty / float(j.len)

        # calculate extra penalties
        for s in self.g2j.job_queue.lst_running_jobs():
            j = self.g2j.sloti2job(s.id)
            if j is None:
                # We have chosen a slot with no job. This is bad. Add a
                # penalty
                ngpustoslot = len(self.g2j.jobsloti2gpus(s))
                extrapenalties += self.get_costofchoosingemptyslot(
                    ngpustoslot)
                continue
            else:
                # Gpu's assigned but not enough.
                if j.res_vec[0] > j.g:
                    gpusshort += j.res_vec[0] - j.g

            # What is the penalty of not enough gpu's being allocated?
            if self.pa.fullpreemptalloccase == 2.1:
                extrapenalties += gpusshort * self.get_unitcostofbadchoice()
            elif self.pa.fullpreemptalloccase == 2.2:
                extracost = self.cluster.numavbl_res[0, 0]
                extracost = extracost * self.get_unitcostofbadchoice()
                extrapenalties += extracost
            else:
                raise ValueError('self.pa.fullpreemptalloccase of %f is '
                                 'invalid' % self.pa.fullpreemptalloccase)

        return reward * 1e-15, extrapenalties * 1e-15

    # class Env
    def get_reward_static(self):
        """
        Double check if this function is okay... it looks okay at the moment...

        :return:
        :rtype:
        """
        reward = 0
        if self.pa.rewardname == 'negslowdown':
            # temp field. reset in env.reset(). must be set whenever a job is
            # done.
            return self.negslowdown
        else:
            # instead of looking at started jobs, some of which may be paused if
            # not allocated during the current step, look instead at
            # g2j.chosenjobs_slots_int
            for jid, job in self.g2j.startednotdone_jobs.items():
                # cbb. running but not yet finished.
                # reward += self.pa.delay_penalty / float(j.len)
                # cbb. continue here
                reward += job.v_m

            # cbb. Previously, Mao looked at not yet running jobs that are just
            # held in the visible queue, in one of the M slots. We want to do the
            #  same for static case.
            for job in self.job_slots.slots:  # job_slots.slots in np.array
                if job is not None:
                    # instead of using hold_penalty, get singlejob speed
                    # reward += self.pa.hold_penalty / float(j.len)
                    reward -= self.job_minbatch_speed_calc(job, 'single') * \
                              self.pa.penaltyfactor2

            # cbb. not yet running, nor in visible queue, in backlog.
            for job in self.job_backlog.backlog:
                reward -= self.job_minbatch_speed_calc(job, 'single')
                # reward += self.pa.dismiss_penalty / float(j.len)

            return reward * 1e-15

    # class Env
    def get_reward_orig(self):  # not used anymore

        reward = 0
        for jid, job in self.g2j.startednotdone_jobs.items():
            # cbb. running but not yet finished.
            reward += self.pa.delay_penalty / float(job.len)

        # cbb. not yet running. just held in the visible queue,
        # in one of the M slots.
        for job in self.job_slots.slots:  # job_slots.slots in np.array
            if job is not None:
                reward += self.pa.hold_penalty / float(job.len)

        for job in self.job_backlog.backlog:
            reward += self.pa.dismiss_penalty / float(job.len)

        return reward

    # class Env
    def allocate_job_static(self, job, curr_time, onlydrawtop=True):
        """
        Takes a job from one of the job_slots's and puts it onto one of the
        available slots on the cluster canvas. It simply fills up the next
        available columns.

        This is the original method by Mao.

        Technically, the 'allocate_job_static' function should not exist in the
        cluster class, since we require information about gpus, job slots,
        and jobs. Thus 'allocate_job_static should be a function of the environment.
        Or perhaps consider if it would make sense to create another class
        for the PG agent. This would normally exist in pg_re_gpucluster,
        but since get_single_traj_pg*() methods are in pg_re_gpucluster, and those
        methods call env.step, and allocate_job_static(s) is env.step related task,
        it makes more sense to make 'allocate_job_static(s)' an environment class
        funciton.

        :param job:
        :param curr_time:
        :param onlydrawtop: whether to only allocate to the top of the canvas
        :return: whether or not job has been allocated.
        """
        allocated = False
        if onlydrawtop:
            trowend = 1
        else:
            trowend = self.cluster.time_horizon - job.len

        numgpusassigned = job.res_vec[0]
        job.ts_togo, job.ts_done = \
            self.pa.distobj.jobnumrowstime(job, numgpusassigned)
        ts_todraw = job.ts_togo + 1

        for t in range(0, trowend):
            # V-cbb. self.numavbl_res has shape=(time_horizon,num_res_types)
            new_avbl_res = self.cluster.numavbl_res[
                           t: t + ts_todraw, :] - job.res_vec
            # V-cbb. determine if, starting from and including row t of the
            # input image, the availability of the cluster meets the resource
            #  requirements of the job. If so, then allocate, and break.
            if np.all(new_avbl_res[:] >= 0):

                allocated = True
                # V-cbb. new_avbl_res already has (t,res_type) elements
                # subtracted by job.res_vec(res_type) (see above). So
                # new_avbl_res already represents the avbl resources for the
                # slots in question.
                self.cluster.numavbl_res[t: t + ts_todraw, :] = new_avbl_res
                # V-cbb (V for below). Since t and image row is zero indexed,
                # first row of
                # image is current_time.
                job.start_time = curr_time + t
                # cbbV. by job.finish_time, the job will be gone from the
                # cluster, i.e. when drawing, the finish_time is excluded.
                # job.finish_time = job.start_time + job.len

                self.g2j.startednotdone_jobs[job.id] = job
                # env.step calls cluster.allocate_job_static as well as
                # cluster.cluster_time_proceed, which checks for done jobs and
                # removes them with 'self.startednotdone_jobs.pop(job)'.

                # Don't need to check for available colors now that we're
                #  using a queue, just get an item from the fifo queue.
                job.color = self.cluster.q_colormap.popleft()
                # put the color right back to the end of the queue for re-use
                #  later.
                self.cluster.q_colormap.append(job.color)

                assert job.start_time != -1
                # cbb. we don't know when finish time will be.
                # assert job.finish_time != -1
                # assert job.finish_time > job.start_time
                canvas_start_time = job.start_time - curr_time
                canvas_end_time = canvas_start_time + ts_todraw

                # V-cbb. Search for and update the cells in the canvas that
                # need to be painted with the 'new_color'.
                for res in range(self.cluster.num_res_types):  # for each
                    # resource type
                    # look for resource availability in just the first row,
                    # and these will be the gpusassigned_set to this job.
                    avbl_slot = np.where(
                        self.cluster.canvas[res, 0, :] == 0)[0]
                    columnstofill = avbl_slot[: job.res_vec[res]]
                    self.cluster.canvas[res, :ts_todraw, columnstofill] = \
                        job.color
                    job.gpusassigned_set.update(columnstofill)
                    self.g2j.assignedgpus_set.update(columnstofill)
                    self.g2j.freegpus_set.difference_update(columnstofill)

                break
        # also return job to signal to the user that job may be mutated
        return allocated, job

    @abstractmethod
    def step(self, action, debugstr=''):
        return 'Should never get here'

    # class Env
    def step_orig(self, action, repeat=False):
        """
        This version should be modified. We don't need to choose single
        actions multiple times until void or invalid action.
        We are assigning gpus to jobs simultaneously, in a single allocation
        step.

        Args:
            :param action: Multiple actions. I.e. Each GPU can be assigned
            to one of the M jobs in the queue.
            :param repeat:
        Returns:
            :ob: the observation of the environment.
            :reward: the reward returned from taking a single step
            :done: True if all jobs allocated.
            :info: env.JobRecord object.
        """
        status = None

        done = False
        reward = 0
        info = None

        if action == self.pa.num_jobslots:  # explicit void action
            #  cbb. i.e. last action is void action.
            status = 'MoveOn'
        elif self.job_slots.slots[action] is None:  # implicit void action
            # job_slots.slot[a] s None, i.e. job_slots at position 'a' is
            # empty. This will be selected for when the job_slots first need
            # to be populated. Everytime a'th slot is chosen but a'th slot is
            #  empty, we will "Moveon" but first need to place a new job there.
            status = 'MoveOn'
        else:
            # a'th job slot contains a job. This will now be
            # allocated/scheduled into the cluster/machine.
            # V-cbb. Logically, we wouldn't think of the cluster as doing the
            #  allocation, but rather some intelligent being that would
            # normally be called the scheduler or the agent. But
            # there are some fields of the Cluster instance that are
            # needed to do the multiple tasks of allocation. So for now,
            # use the cluster object.
            allocated = self.cluster.allocate_job_orig(
                self.job_slots.slots[action], self.curr_time)
            if not allocated:  # implicit void action.
                # ^-cbb. allocated is false because job doesn't fit. In our
                # full premption case, job will always fit, so 'not
                # allocated' will never happen.
                status = 'MoveOn'
            else:
                status = 'Allocate'  # we completed the process of allocating
                #  the job into the cluster in the method cluster.allocate_job_static,
                # which also involves removing the job from the slot,
                # updating the cluster.numavbl_res variable, updating the
                # cluster.startednotdone_jobs list (we want to use a dict instead).

        if status == 'Allocate':
            # cbb. update the job_record for the job allocated above (e.g.
            # job has updated start and end times, but unlike Mao, do not
            # remove the job from the job slot.
            # cbb. remember, job_record.record by Mao is a dictionary.
            # key's are job id's
            self.job_record_addjobtoit(action)
            self.job_slots.slots[action] = None

            # dequeue backlog. cbb. understood. For us, we only do this when
            # a job is finished and thus a slot becomes empty.
            self.dequeue_backlog_afterallocate(action)

            # cbb. according to the paper, we do multiple allocations per
            # time step. So after allocating, don't calculate the reward just
            #  yet, let the outside scope function env.step take another
            # step. At the next step, we may again do an allocation,
            # or we may do something else (e.g. sample a new job and place it
            #  into a job slot or backlog, or fail to sample a job and do
            # another step).

        elif status == 'MoveOn':
            # V-cbb. Since we are not allocating in this step, we can advance
            # the timestep. Remember, we don't move on from an allocation. The
            # paper lets multiple allocations occur in a single time step.
            # V-cbb. Three important lines. We must do this as well.

            # Logical Error in Mao's code. Reward calculation should be here
            # for the current time step, not after the time step has been
            # incremented. Mao's version leads to job's with one time step
            # thickness at the top of the cluster canvas never gets counted in
            # the reward (little r) signal because it get's removed by
            # cluster_time_proceed from startednotdone_jobs which is used by Mao as a proxy
            # for allocated jobs.
            self.curr_time += 1
            self.cluster_time_proceed(self.curr_time)
            self.extra_info.time_proceed()

            # add new jobs. Advance one step further in current job sequence.
            self.seq_idx += 1

            if self.end == "no_more_jobs_in_seq":  # end of new job sequence
                # ^-cbb. self.end is either "no_more_jobs_in_seq" or
                # "do_all_jobs_in_seq". Former names: 'no_new_jobs' and
                # 'all_done'
                if self.seq_idx >= self.pa.t_finaljobarrival:
                    done = True
            elif self.end == "do_all_jobs_in_seq":  # everything has to be finished
                # V-cbb. If the following multiple conditions are not met,
                # then it means that in reality, we're not all done. so
                # 'done' variable will remain as false.
                if self.seq_idx >= self.pa.t_finaljobarrival and \
                        len(self.g2j.startednotdone_jobs) == 0 and \
                        all(s is None for s in self.job_slots.slots) and \
                        len(self.job_backlog.backlog) == 0:
                    done = True
                elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
                    done = True
                # ^-cbb. else done remains False.

            if not done:
                # cbb. e.g. self.end == "do_all_jobs_in_seq" in reality, done is still
                # false, because, we haven't even started yet.
                #  e.g. self.seq_idx === 1 (note: self.pa.simul_len === 50t)
                # (i.e. there are no jobs in the slots).
                # cbb. not done and 0 <= seq_idx <= 49 is less then t_finaljobarrival === 50
                # then see if a job has arrived for this time step (i.e. new_job.len > 0)

                if self.seq_idx < self.pa.t_finaljobarrival:  # otherwise, end of new job sequence,
                    # ^-cbb. If the above is true, it means we've reached
                    # the end of the new job sequence, which means no new jobs.
                    # cbb. self.end == "do_all_jobs_in_seq" but self.seq_idx=1,
                    # note: self.pa.sumu_len=50t.
                    # So we aren't actually done, so get a new job from the
                    # job sequence.
                    new_job = self.get_new_job_from_seq(self.seq_no,
                                                        self.seq_idx)

                    if new_job.len > 0:  # a new job comes
                        # cbb. Since we haven't allocated a job, we are
                        # either putting a new job into one of the M job
                        # slots or putting it into the backlog.

                        to_backlog = True  # cbb. set true for now

                        # V-cbb. See if the new job we sampled can be
                        # put into one of the M job slots.
                        for i in range(self.pa.num_jobslots):
                            if self.job_slots.slots[i] is None:
                                # put in new visible job slots.
                                # i.e. because the i'th job slot is empty.
                                to_backlog = False
                                self.job_slots.slots[i] = new_job
                                self.job_record.record[new_job.id] = new_job
                                break

                        # V-cbb. See if the new job that we sampled can be
                        # put into the backlog.
                        if to_backlog:
                            self.tobacklog(new_job)

                        # ^-cbb. As can be seen in the for loop and if
                        # statement above, a job is added to
                        # job_record.record when it is placed into a slot,
                        # or a backlog. But as seen below, also, a job is added
                        #  to the record, even when it's being allocated.
                        # Thus there is no data structure that only contains
                        # the jobs currently running in the
                        # 'cluster'/machine. We must make one.

                        # V-cbb. We simply set the 'extra_info' object's
                        # time_since_last_new_job field to 0. (This is done
                        # when a new job arrives, not when it is allocated).
                        self.extra_info.new_job_comes()

                        # in this nested if statement, we have taken a new
                        # job that has arrived, and placed it either in one
                        # of the M job slots, or to the backlog. Pretty much
                        # the only thing left to do is to calculate the
                        # reward for this time step.

                    else:  # cbb. no job for this time point in the job
                        # sequence. Nothing to do besides calculate the
                        # reward for the current step.
                        pass

            reward = self.get_reward_fullpreempt()

        # cbb. regardless of whether allocate a job to the machine/cluster or
        #  not, we should observer the environment (i.e, get an image of it)
        ob = self.observe()

        # V-cbb. info holds the environment's job record, which is useful for
        #  calculating slow down of jobs of a trajectory.
        info = self.job_record

        if done:
            self.seq_idx = 0

            if not repeat:
                # ^-cbb. if repeat True, then keep self.seq_no the same.
                # if repeat is False (as is inside the if statement here),
                # then advance self.seq_no by one, but also overflow if
                #  seq_no becomes larger than num_jobsets.
                self.seq_no = (self.seq_no + 1) % self.pa.num_jobsets

            self.reset()

        if self.render:
            self.plot_state()

        # cbb. Remember, that if done is true, pg_re_*.run_episode, which builds
        #  a single episode trajectory will break ( # i.e, it will stop
        # building the trajectory for a single episode )
        return ob, reward, done, info

    # class Env
    def step_static_oneaction(self, action, debugstr=''):
        """
        This version is like the original step which only chooses a single
        action. We assume action is a just a single action. The method does
        static allocation, and also looks for contiguous blocks (like Mao),
        but only cares to schedule jobs starting from the first row of the
        image.

        :param debugstr:
        :type debugstr:
        :param action:
        :type action: For now this version assumes single action.
        :param repeat:
        :type repeat:

        :return:
        :rtype:
        """

        if debugstr == 'step: 78 sched_type: Tetris':  # cbb. get rid of this.
            pass

        done = False
        reward = 0

        self.pa.debug_dict['action'] = action

        if self.curr_time == 1:
            pass

        if action == self.pa.num_jobslots:  # explicit void action
            #  cbb. i.e. last action is void action.
            status = 'MoveOn'
        elif action >= len(self.job_slots.slots):
            # elif self.job_slots.slots[action] is None: # implicit
            # void action
            # job_slots.slot[a] s None, i.e. job_slots at position 'a' is
            # empty. This will be selected for when the job_slots first need
            # to be populated. Everytime a'th slot is chosen but a'th slot is
            #  empty, we will "Moveon" but first need to place a new job there.
            status = 'MoveOn'
        else:
            # a'th job slot contains a job. This will now be
            # allocated/scheduled into the cluster/machine.
            status = 'Allocate'  # Try to allocate.

        # if self.pa.debug_dict['sched_type'] == 'sjf':
        #     if self.curr_time == 10:
        #         pass

        if status == "Allocate":  # this replaces the allocat_job_static()
            # function
            # General idea. Since we can't just shift the canvas up,
            # and since in the previous step, we may have come from "moveOn",
            #  which would have incremented the compdisdone of running jobs
            # and removed done jobs, we need to redraw the jobs that started
            # and still have compdist left (found from
            # 'cluster.startednotdone_jobs'), and THEN try to fit a single
            # new job --corresponding to the action taken-- into the cluster.

            # No need to clear and redraw the canvas because that was done
            # during MoveOn in the previous step, and also after a job was
            # allocated.

            # Now try allocating the job corresponding to the action, and keep
            #  track of which GPUs were allocated. allocate_job_static adds
            # to env.startednotdone_jobs

            allocated, job = self.allocate_job_static(
                self.job_slots.slots[action], self.curr_time, onlydrawtop=True)
            if allocated:

                # If allocated is true, then self.g2j.startednotdone_jobs is
                # updated insde self.allocate_job_static

                # cbb. update the job_record for the job allocated above (e.g.
                # job has updated start and end times, but unlike Mao, do not
                # remove the job from the job slot.
                # cbb. remember, job_record.record by Mao is a dictionary.
                # key's are job id's
                self.job_record.record[job.id] = job
                # below should update env.g2j.job_slots as well.
                del self.job_slots.slots[action]  # slots is a deque

                # dequeue backlog. cbb. understood. For us, we only do this when
                # a job is finished and thus a slot becomes empty.
                self.dequeue_backlog_afterallocate2()

                if self.pa.debug_actions:
                    self.writeoutput_debug_dict_shared('After Allocate',
                                                       stackupby=1)
                # cbb. according to the paper, we do multiple allocations per
                # time step. So after allocating, don't calculate the reward just
                #  yet, let the outside scope function env.step take another
                # step. At the next step, we may again do an allocation,
                # or we may do something else (e.g. sample a new job and place it
                #  into a job slot or backlog, or fail to sample a job and do
                # another step).
                # print(debugstr, 'action: ' + str(action))
                status = 'Allocate'

                # After a job allocation, clear and redraw the cluster image
                self.clear_redraw_canvas()
            else:
                # Not Allocated if there is not enough space in the top row
                # columns. We are making this simplification because jobs
                # progress at different speeds depending on their allocation,
                # so it's hard to keep track of where future scheduled jobs
                # should be redrawn (ie. moved upwards in the canvas).
                status = 'MoveOn'

        if status == 'MoveOn':
            # Once we are ready to move on,
            # debuginfo('self.curr_time: %d' % self.curr_time)
            if self.curr_time == 63:
                pass
            self.findlimitingbws_static()
            # After finding the limitingbws for each job, we can now advance
            # each running job's compistdone field.

            self.advance_runningjobs_onestep()

            # before we finish moving on, we we need to get reward for
            # the current step, before we increment the time step and
            # prepare the next state as input to the neural network.
            reward = self.get_reward_static()

            # cbb. now move on in the sequence (so must first advance the time
            # step). i.e. prepare the environment for next time step.
            self.curr_time += 1
            # Even if no gpus were allocated this round, existing jobs may
            # have progressed in their compdist so always check to remove
            # done jobs
            removedjobs = self.remove_done_jobs_static(self.curr_time - 1,
                                                       updatefields=True)
            self.add_to_perstep_slowdown(removedjobs)

            self.extra_info.time_proceed()  # okay. Should not need to change.

            # add new jobs. Advance one step further in current job sequence.
            self.seq_idx += 1

            # Redraw the canvas to prepare for next step since jobs
            # were advanced and possibly even removed.
            self.clear_redraw_canvas()

            # cbb. check if by moving on we have reached the end. Mao never
            # changes self.end. He sets it during launch to signal whether to
            # finish after just a single job sequence (self.end == "no_more_jobs_in_seq")
            # OR also require in addition that 1) all started jobs finish,
            # 2) there are no more jobs in job_slots.slots, and 3) that there
            # are no more jobs in job_backlog.backlog.
            if self.end == "no_more_jobs_in_seq":  # end of new job sequence
                # ^-cbb. self.end is either "no_more_jobs_in_seq" or "do_all_jobs_in_seq"
                if self.seq_idx >= self.pa.t_finaljobarrival:
                    done = True
            elif self.end == "do_all_jobs_in_seq":  # everything has to be finished
                # V-cbb. If the following multiple conditions are not met,
                # then it means that in reality, we're not all done. so
                # 'done' variable will remain as false.
                # NOTE!!! pg_re is launched with end="do_all_jobs_in_seq"!!!
                if self.seq_idx >= self.pa.t_finaljobarrival and \
                        len(self.g2j.startednotdone_jobs) == 0 and \
                        all(s is None for s in self.job_slots.slots) and \
                        len(self.job_backlog.backlog) == 0:
                    done = True
                elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
                    done = True
                # ^-cbb. else done remains False.

            if not done:
                # cbb. e.g. self.end == "do_all_jobs_in_seq" in reality, done is still
                # false, because, we haven't even started yet.
                #  e.g. self.seq_idx === 1 (note: self.pa.simul_len === 50t)
                # (i.e. there are no jobs in the slots).
                # cbb. not done and 0 <= seq_idx <= 49 is less then t_finaljobarrival === 50
                # then see if a job has arrived for this time step (i.e. new_job.len > 0)

                # if t_finaljobarrival not reached, but no job, get a job.
                if self.seq_idx < self.pa.t_finaljobarrival:  # otherwise, end of new job sequence,
                    # ^-cbb. If the above is true, it means we've reached
                    # the end of the new job sequence, which means no new jobs.
                    # cbb. self.end == "do_all_jobs_in_seq" but self.seq_idx=1,
                    # note: self.pa.sumu_len=50t.
                    # So we aren't actually done, so get a new job from the
                    # job sequence.
                    if self.pa.testjobmodel:
                        new_job = self.get_new_job_from_seq(self.seq_no,
                                                            self.seq_idx,
                                                            testjob=True)
                    else:
                        new_job = self.get_new_job_from_seq(self.seq_no,
                                                            self.seq_idx)

                    if new_job.len > 0:  # a new job comes
                        # cbb. Since we haven't allocated a job, we are
                        # either putting a new job into one of the M job
                        # slots or putting it into the backlog.

                        to_backlog = True  # cbb. set true for now

                        # V-cbb. See if the new job we sampled can be
                        # put into one of the M job slots.
                        # This for loop would not be necessary if after a
                        # finished job is deleted from its slot, we simply do
                        #  slots[emptyslot:-1] = slots[emptyslot+1:]. Maybe
                        # do this later. The following for loop is okay for now.

                        # for i in range(self.pa.num_jobslots):
                        #     if self.job_slots.slots[i] is None:
                        #         # put in new visible job slots.
                        #         # i.e. because the i'th job slot is empty.
                        #         to_backlog = False
                        #         # cbb. not all the mutations by the below
                        #         # function are neccessary because the below
                        #         # function uses g2j._x but static case
                        #         # doesn't use g2j._x.
                        #         self.g2j.assigntoslots_updatefields_static(
                        #             i, new_job)
                        #
                        #         self.job_record.record[new_job.id] = new_job
                        #         break
                        if len(self.job_slots.slots) < \
                                self.job_slots.slots.maxlen:
                            to_backlog = False
                            # cbb. not all the mutations by the below
                            # function are neccessary because the below
                            # function uses g2j._x but static case
                            # doesn't use g2j._x.
                            self.g2j.assigntoslots_updatefields_static(
                                job=new_job)
                            self.job_record.record[new_job.id] = new_job

                        # V-cbb. See if the new job that we sampled can be
                        # put into the backlog.
                        if to_backlog:
                            self.tobacklog(new_job)

                        # ^-cbb. As can be seen in the for loop and if
                        # statement above, a job is added to
                        # job_record.record when it is placed into a slot,
                        # or a backlog. But as seen below, also, a job is added
                        #  to the record, even when it's being allocated.
                        # Thus there is no data structure that only contains
                        # the jobs currently running in the
                        # 'cluster'/machine. We must make one. For our case
                        # though, the currently running jobs can be queried
                        # with: self.g2j.chosenjobs_slots_int,
                        # which prefilters for slots that actually have jobs.

                        # V-cbb. We simply set the 'extra_info' object's
                        # time_since_last_new_job field to 0. (This is done
                        # when a new job arrives, not when it is allocated).
                        self.extra_info.new_job_comes()

                        # in this nested if statement, we have taken a new
                        # job that has arrived, and placed it either in one
                        # of the M job slots, or to the backlog. Pretty much
                        # the only thing left to do is to calculate the
                        # reward for this time step.

                    else:  # cbb. no job for this time point in the job
                        # sequence. Nothing to do. We already calculated the
                        # reward for the current step, and we just move on
                        pass
                # print(debugstr + ", " +
                #       "Status MoveOn.\t action: {}, slotsbool: {}".format(
                #         action, slotsbool))
                if self.pa.debug_actions:
                    self.writeoutput_debug_dict_shared('Endof MoveOn',
                                                       stackupby=1)

        ob = self.observe()

        # V-cbb. info holds the environment's job record, which is useful for
        #  calculating slow down of jobs of a trajectory.
        info = self.job_record

        if done:
            self.seq_idx = 0
            #
            # if not repeat:
            #     # ^-cbb. if repeat True, then keep self.seq_no the same.
            #     # if repeat is False (as is inside the if statement here),
            #     # then advance self.seq_no by one, but also overflow if
            #     #  seq_no becomes larger than num_jobsets.
            #     self.seq_no = (self.seq_no + 1) % self.pa.num_jobsets
            for job in self.g2j.startednotdone_jobs:
                job.finish_time = self.curr_time

            self.reset()

        if self.render:
            self.plot_state()

        # cbb. Remember, that if done is true, pg_re_*.run_episode, which builds
        #  a single episode trajectory will break ( # i.e, it will stop
        # building the trajectory for a single episode )

        return ob, reward, done, info

    # class Env
    def step_full_preempt(self, actions, repeat=False):
        """
        Multiple simultaneous actions via multiple softmaxes.

        Args:
            :param actions: Multiple actions. I.e. Each GPU can be assigned
                to one of the M jobs in the queue. Index is for gpu, value is
                which job to assign the gpu to.
            :param repeat:
        Returns:
            :ob: the observation of the environment.
            :reward: the reward returned from taking a single step
            :done: True if all jobs allocated.
            :info: env.JobRecord object.
        """
        status = None

        done = False
        reward = 0
        info = None
        extrapenalties = 0

        # V-cbb. Assume actions is a vector. Index is gpu, element value is
        # job slot.g
        # allocate jobs to machine/cluster. then moveon. The other
        # option is when non gpus are assigned, is to keep the previous gpu
        # assignments. This is not as sensical.
        # Thus in our case, we always allocate, then move on. By move on,
        # it means we mean that we increment the time step (and do so for all
        #  data structures that need the time step incremented).

        allocated = False
        # cbb. update the g2j table. This also updates the
        # G2J.running_jobs index vector.
        self.g2j.x_update_allgpus(actions)

        # If there are no chosen slots with jobs then must move on.
        # In theory chosenjobs_slots_int will also be an empty ndarray if there
        # are no jobs in the the job slots.
        if not self.g2j.chosen_slots_bool.any():
            status = 'MoveOn'
        else:
            status = 'Allocate'
            # cbb. Since we are doing full pre-emption, we will always
            # allocate if there is atleast one chosen slot with a job.

        if status == "Allocate":  # this replaces the allocat_job_static()
            # function
            # cbb. clear the canvas so we can paint it fresh soon.
            self.cluster.canvas[:] = 0

            # since we do full pre-emption, just reset the numavbl_res variable.
            self.cluster.avbl_res_reset()

            # clear/reset both assignedgpus_set and freegpus_set.
            self.g2j.assignedgpus_set.clear()
            self.g2j.freegpus_set.clear()
            self.g2j.freegpus_set.update(self.g2j.allgpusset)

            # for full pre-emption, here, before we begin allocating for jobs
            #  that are assigned GPUs, we must look at the previous step
            # g2j table to see which jobs are meant to be paused. This
            # way we can actually pause these jobs. Then with the rest of the
            #  jobs that are allocated gpu's, we can proceed below.

            # just look at chosen jobs. Once we MoveOn, we will calculate
            # penalties such as 1. Gpu's assigned to empty slots, 2. Job in
            # the slot, but no gpu's assigned (dismissed jobs). 3. Gpu(s)
            # assigned but not enough.
            for s in self.g2j.chosenjobs_slots_int:
                # This for loop implicitly only considers slots that are
                # both chosen and has jobs. So we don't need special cases for
                # 'allocate' and 'move on'. If no jobs in the slots, then it's
                # like 'move on'.
                job = self.job_slots.slots[s]

                # If we've gotten this far, then there is a job in the slot.
                # AND gpu(s) were assigned to the job.
                # Unlike in static case, we don't care as much that some
                # gpu's are already assigned to static jobs, i.e. we don't
                # have static jobs. We want to care about cost of
                # moving/changing a gpu's job assignment, but do that later.
                # gpusrequested = self.g2j.jobsloti2gpus(s)
                # gpuswecangivetojob.clear()
                job.gpusassigned_set.clear()
                # lengpuswecangivetojob = len(gpuswecangivetojob)
                # assert lengpuswecangivetojob > 0

                # regardless of wether or not lengpuswecangivetojob >
                # job.res_vec[0], we allocate. This is case 2.

                # cbb. If the job has not yet started, then indicate that we
                # are starting it, and assign it a color
                if job.start_time == -1:
                    job.start_time = self.curr_time
                    # assign new color to the job and add the job to
                    # self.startednotdone_jobs. Reconsider to possibly differentiate
                    #  between started and running jobs. Just because a job
                    # has started, it doesn't mean it's currently running.
                    # Don't need to check for available colors now that we're
                    #  using a queue, just get an item from the fifo queue.
                    job.color = self.cluster.q_colormap.popleft()  # by default
                    self.cluster.q_colormap.append(job.color)
                    # Queue class raises exception if no item available.
                    self.g2j.startednotdone_jobs[job.id] = job

                    # cbb. update the job_record outside this if statement.

                # if job.start_time != -1 then we don't have to assign a
                # color because one was already assigned.

                gpuschosen = self.g2j.jobsloti2gpus(s)
                # makes sense for full pre-emption, for now, where we don't
                # count the cost of moving gpu's.
                job.gpusassigned_set.clear()
                job.gpusassigned_set.update(gpuschosen)

                self.g2j.assignedgpus_set.update(gpuschosen)
                self.g2j.freegpus_set.difference_update(gpuschosen)

                # Since we have just assigned gpus, we can now calculate the
                # amount of distance left for the job, and thus the number of
                # rows of time to draw.

                job.ts_togo, job.ts_done = \
                    self.pa.distobj.jobnumrowstime(job, job.g)
                # G2J.jobsloti2gpus(sloti) will only return indices of
                # gpus that were assigned to sloti AND a job at sloti exists,
                #  otherwise, it returns None. Luckily in the above if
                # statement, we already checked that a job at sloti exists.
                res = 0  # resource type. We only have gpus as resources.

                # now paint the canvas in one shot for the current job.
                # But make sure to take into account the job's current
                # distance left, and evenly spread it across the gpu's
                # assigned. Must also first count how many gpus have been
                # assigned.
                self.cluster.canvas[res, :job.ts_togo, gpuschosen] = \
                    job.color
                # Even though we don't use numavbl_res, we can do an update of
                # it here. Originally, Mao does this in allocate_job_static(). Note.
                #  numavbl_res has shape=(time_horizon, num_res_types)
                self.cluster.avbl_res_updateafterjobassigned(job, job.g)

                # cbb. update the job_record for the new job that we are
                # going to run on the cluster (e.g. job has updated start
                #  time, but unlike Mao, do not remove the job
                # from the job slot unless the job finishes
                # cbb. remember, job_record.record is a dictionary.
                # key's are job id's
                self.job_record.record[job.id] = job

                # Do Not de-queue the backlog onto the chosen slot because we
                #  leave the allocated job on the slot for the current step.
                # We do this in env.time_proceed_fullpreemption()

            if self.pa.debug_actions:
                self.writeoutput_debug_dict_shared('After Allocate', 1)
            # We will always move on after allocating in the full premption
            # case.
            status = 'MoveOn'
            allocated = True

        if status == "MoveOn":
            # Must update limitingbws of jobs regardless of whether or not we
            # allocate because even without allocation, we may have removed
            # a finished job from the cluster in the previous 'MoveOn' phase.
            # finlimitingbws() does multiple passes through chosenslots that
            # have jobs. It's also needed to properly calculate rewards.
            self.findlimitingbws()

            # After finding the limitingbws for each job, we can now advance
            # each chosen job's compistdone field. We need to do it here
            # regardless of whether or not we allocated, since even without new
            # allocation, there still may exist jobs that are running on the
            # cluster.
            self.advance_runningjobs_onestep()

            # If actions are taken, we must get the reward after. Also, we need
            #  to get reward for the current step, before we increment the
            # time step and prepare the next state as input to the neural
            # network.
            reward, extrapenalties = self.get_reward_fullpreempt()

            # cbb. now move on in the sequence (so must first advance the time
            # step). i.e. prepare the environment for next time step.
            self.curr_time += 1
            # Even if no gpus were allocated this round, existing jobs may
            # have progressed in their compdist so always check to remove
            # done jobs
            removedjobs = self.remove_done_jobs(self.curr_time - 1,
                                                updatefields=True)
            self.add_to_perstep_slowdown(removedjobs)

            self.extra_info.time_proceed()  # okay. Should not need to change.

            # add new jobs. Advance one step further in current job sequence.
            self.seq_idx += 1

            # cbb. check if by moving on we have reached the end. Mao never
            # changes self.end. He sets it during launch to signal whether to
            # finish after just a single job sequence (self.end == "no_more_jobs_in_seq")
            # OR also require in addition that 1) all started jobs finish,
            # 2) there are no more jobs in job_slots.slots, and 3) that there
            # are no more jobs in job_backlog.backlog.
            if self.end == "no_more_jobs_in_seq":  # end of new job sequence
                # ^-cbb. self.end is either "no_more_jobs_in_seq" or "do_all_jobs_in_seq"
                if self.seq_idx >= self.pa.t_finaljobarrival:
                    done = True
            elif self.end == "do_all_jobs_in_seq":  # everything has to be finished
                # V-cbb. If the following multiple conditions are not met,
                # then it means that in reality, we're not all done. so
                # 'done' variable will remain as false.
                # NOTE!!! pg_re is launched with end="do_all_jobs_in_seq"!!!
                if self.seq_idx >= self.pa.t_finaljobarrival and \
                        len(self.g2j.startednotdone_jobs) == 0 and \
                        all(s is None for s in self.job_slots.slots) and \
                        len(self.job_backlog.backlog) == 0:
                    done = True
                elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
                    done = True
                # ^-cbb. else done remains False.

            if not done:
                # cbb. e.g. self.end == "do_all_jobs_in_seq" in reality, done is still
                # false, because, we haven't even started yet.
                #  e.g. self.seq_idx === 1 (note: self.pa.simul_len === 50t)
                # (i.e. there are no jobs in the slots).
                # cbb. not done and 0 <= seq_idx <= 49 is less then t_finaljobarrival === 50
                # then see if a job has arrived for this time step (i.e. new_job.len > 0)

                # if t_finaljobarrival not reached, but no job, get a job.
                if self.seq_idx < self.pa.t_finaljobarrival:  # otherwise, end of new job sequence,
                    # ^-cbb. If the above is true, it means we've reached
                    # the end of the new job sequence, which means no new jobs.
                    # cbb. self.end == "do_all_jobs_in_seq" but self.seq_idx=1,
                    # note: self.pa.sumu_len=50t.
                    # So we aren't actually done, so get a new job from the
                    # job sequence.
                    if self.pa.testjobmodel:
                        new_job = self.get_new_job_from_seq(self.seq_no,
                                                            self.seq_idx,
                                                            testjob=True)
                    else:
                        new_job = self.get_new_job_from_seq(self.seq_no,
                                                            self.seq_idx)

                    if new_job.len > 0:  # a new job comes
                        # cbb. Since we haven't allocated a job, we are
                        # either putting a new job into one of the M job
                        # slots or putting it into the backlog.

                        to_backlog = True  # cbb. set true for now

                        # V-cbb. See if the new job we sampled can be
                        # put into one of the M job slots.
                        # This for loop would not be necessary if after a
                        # finished job is deleted from its slot, we simply do
                        #  slots[emptyslot:-1] = slots[emptyslot+1:]. Maybe
                        # do this later. The following for loop is okay for now.

                        # for i in range(self.pa.num_jobslots):
                        #     if self.job_slots.slots[i] is None:
                        #         # put in new visible job slots.
                        #         # i.e. because the i'th job slot is empty.
                        #         to_backlog = False
                        #         self.g2j.assigntoslots_updatefields_fullpreempt(
                        #             i, new_job)
                        #
                        #         self.job_record.record[new_job.id] = new_job
                        #         break
                        if len(self.job_slots.slots) < \
                                self.job_slots.slots.maxlen:
                            to_backlog = False
                            self.g2j.assigntoslots_updatefields_fullpreempt(
                                job=new_job)
                            self.job_record.record[new_job.id] = new_job

                        # V-cbb. See if the new job that we sampled can be
                        # put into the backlog.
                        if to_backlog:
                            self.tobacklog(new_job)
                            # exit(1)

                        # ^-cbb. As can be seen in the for loop and if
                        # statement above, a job is added to
                        # job_record.record when it is placed into a slot,
                        # or a backlog. But as seen below, also, a job is added
                        #  to the record, even when it's being allocated.
                        # Thus there is no data structure that only contains
                        # the jobs currently running in the
                        # 'cluster'/machine. We must make one. For our case
                        # though, the currently running jobs can be queried
                        # with: self.g2j.chosenjobs_slots_int,
                        # which prefilters for slots that actually have jobs.

                        # V-cbb. We simply set the 'extra_info' object's
                        # time_since_last_new_job field to 0. (This is done
                        # when a new job arrives, not when it is allocated).
                        self.extra_info.new_job_comes()

                        # in this nested if statement, we have taken a new
                        # job that has arrived, and placed it either in one
                        # of the M job slots, or to the backlog. Pretty much
                        # the only thing left to do is to calculate the
                        # reward for this time step.

                    else:  # cbb. no job for this time point in the job
                        # sequence. Nothing to do. We already calculated the
                        # reward for the current step, and we just move on
                        pass

            if self.pa.debug_actions:
                self.writeoutput_debug_dict_shared('Endof MoveOn', 2)
        # v-cbb. state_obs returned by observelstm1 is a list
        state_obs = self.observe()

        # V-cbb. info holds the environment's job record, which is useful for
        #  calculating slow down of jobs of a trajectory.
        info = self.job_record

        if done:
            self.seq_idx = 0

            # if not repeat:
            #     # ^-cbb. if repeat True, then keep self.seq_no the same.
            #     # if repeat is False (as is inside the if statement here),
            #     # then advance self.seq_no by one, but also overflow if
            #     #  seq_no becomes larger than num_jobsets.
            #     self.seq_no = (self.seq_no + 1) % self.pa.num_jobsets

            for job in self.g2j.startednotdone_jobs:
                job.finish_time = self.curr_time

            self.reset()

        if self.render:
            self.plot_state()

        # cbb. Remember, that if done is true, pg_re_*.get_single_traj_pg, which builds
        #  a single episode trajectory will break ( # i.e, it will stop
        # building the trajectory for a single episode )

        self.extrapenalties = extrapenalties
        if self.pa.applyextrapenalties:
            reward += self.extrapenalties

        # v-cbb. state_obs returned by observelstm1 is a list of ndarrays,
        # not a single ndarray
        # Note: below returns to en
        return state_obs, reward, done, info

    # # class Env
    # def step_static_rlalloc(self, actions, repeat=False, debugstr=''):
    #     """
    #     Mostly a copy of Mao's job picker, but we allocate multiple gpu's
    #     at a time. Allocating GPU's to a job means that job is picked. Just
    #     like Mao, we remove job's from the slot once picked/(assigned
    #     gpus). If a gpu that is already assigned to a job in
    #     started_not_done_jobs is reassigned this step, then count a penalty
    #     for that gpu assignment. If a gpu that is already assigned to a job
    #     in started_not_done_jobs is given assigned the null slot (unit 11
    #     in the softmax), ie. it's not asigned, then give a reward.
    #
    #     Args:
    #         :param actions: Multiple actions. I.e. Each GPU can be assigned
    #             to one of the M jobs in the queue. Index is for gpu, value is
    #             which job to assign the gpu to.
    #         :param repeat:
    #     Returns:
    #         :ob: the observation of the environment.
    #         :reward: the reward returned from taking a single step
    #         :done: True if all jobs allocated.
    #         :info: env.JobRecord object.
    #     """
    #
    #     done = False
    #     reward = 0
    #     extrapenalties = 0
    #
    #     # keep here for now.
    #     self.pa.debug_dict['action'] = actions
    #
    #     allocated = False
    #     # cbb. update the g2j table. This also updates the
    #     # G2J.running_jobs index vector.
    #     # Todo. this may cause problems because information about void action
    #     #  for each gpu is not included in the _x table.
    #     self.g2j.x_update_allgpus(actions)
    #
    #     if not self.g2j.chosenjobs_slots_bool.any():
    #         #  cbb. i.e. last action is void action.
    #         status = 'MoveOn'
    #     else:
    #         # a'th job slot contains a job. This will now be
    #         # allocated/scheduled into the cluster/machine.
    #         status = 'Allocate'  # Try to allocate.
    #
    #     if status == "Allocate":  # this replaces the allocat_job_static()
    #         # function
    #         # General idea. Since we can't just shift the canvas up,
    #         # and since in the previous step, we may have come from "moveOn",
    #         #  which would have incremented the compdisdone of running jobs
    #         # and removed done jobs, we need to redraw the jobs that started
    #         # and still have compdist left (found from
    #         # 'cluster.startednotdone_jobs'), and THEN try to fit a single
    #         # new job --corresponding to the action taken-- into the cluster.
    #
    #         # cbb. clear the canvas so we can paint it fresh soon.
    #         self.cluster.canvas[:] = 0
    #
    #         # Even though we do static, since we redraw the jobs that take up
    #         #  the first row time step, we can reset the numavbl_res variable.
    #         self.cluster.avbl_res_reset()
    #
    #         # since we redraw the cluster.canvas and reset the
    #         # cluster.avbl_res, let's also reset g2j.assignedgpus_set and
    #         # g2j.freegpus_set. We technically don't have to if these sets
    #         # are properly updated when a job finishes. Since currently we
    #         # don't have a proper 'runningjobs' function for each child
    #         # environment, there are multiple points in the code for
    #         # potential error in these sets. So just reset them to avoid
    #         # having to deal with the potential errors in set membership.
    #
    #         # clear/reset both assignedgpus_set and freegpus_set.
    #         self.g2j.assignedgpus_set.clear()
    #         self.g2j.freegpus_set.clear()
    #         self.g2j.freegpus_set.update(self.g2j.allgpusset)
    #
    #         # for batch static scheduling, let's first draw the jobs that
    #         # are not done yet.
    #
    #         # cbb. Begin Redrawing startednotdone jobs.
    #         # Todo. create special child runningjobs function/property.
    #         for jid in self.g2j.startednotdone_jobs:
    #             # cluster.startednotdone_jobs dict might be empty such as
    #             # when we just start or all the started jobs finished from
    #             # the previous step's MoveOn stage simultaneously, but that's
    #             # okay.
    #             job = self.g2j.startednotdone_jobs[jid]
    #
    #             if len(job.gpusassigned_set) > 0:
    #                 # it's best to make job.numgpsallocated be a property
    #                 # based on len(job.gpusassigned_set)
    #                 # Note: numgpusallocated and numgpusassigned might mean
    #                 # different things, consider settling on definitions.
    #
    #                 # We don't need an order for the jobs to be redraw, we can
    #                 # simply draw each job according to the list of GPU's
    #                 # assigned, starting from the first time row.
    #                 # cbbb. Important, just redraw first the jobs
    #                 job.ts_togo, job.ts_done = \
    #                     self.pa.distobj.jobnumrowstime(job, job.g)
    #                 ts_togo = job.ts_togo
    #                 gpusassigned = np.array(list(job.gpusassigned_set),
    #                                         dtype=int)
    #                 res = 0  # resource type. We only have gpus as resources.
    #                 self.cluster.canvas[res, :ts_togo, gpusassigned] = job.color
    #                 # job is not assigned gpu's again, but the function below
    #                 # still works for updating cluster.avbl_slot
    #                 self.cluster.avbl_res_updateafterjobassigned(job, job.g)
    #
    #                 self.g2j.assignedgpus_set.update(gpusassigned)
    #                 self.g2j.freegpus_set.difference_update(gpusassigned)
    #
    #         # Cbb. The below should have been (or should be) correctly
    #         # replaced. The below only works for static allocation to single
    #         # jobs.
    #         # Now try allocating the job corresponding to the actions of this
    #         # step, and keep track of which GPUs were allocated.
    #         # allocate_job_static adds to env.startednotdone_jobs
    #
    #         # try to allocate more jobs from the slots. Make sure to remove
    #         # the job from the slot if allocated.
    #         gpuswecangivetojob = set()
    #         for s in self.g2j.chosen_slots_int:
    #             # Todo. Try to use chosen_slots_int instead so we can
    #             #  penalize choosing slots that have no jobs in them !!!!
    #             #  This would need to be tried for all the other step functions!
    #
    #             # This for loop implicitly only considers slots that are
    #             # both chosen and has jobs.
    #             job = self.job_slots.slots[s]
    #             if self.job_slots.slots[s] is None:
    #                 # We have chosen a slot with no job. This is bad. Ada a
    #                 # penalty.
    #                 # Todo. Do similar for all other step functions. Add a
    #                 #  penalty to reward to help train but reverse it later
    #                 #  to compare with baselines.
    #                 # Find the number of gpus assigned to the empty slot.
    #                 ngpustoslot = len(self.g2j.jobsloti2gpus(s))
    #                 extrapenalties += self.get_costofchoosingemptyslot(
    #                     ngpustoslot)
    #                 continue
    #
    #             # If we've gotten this far, then there is a job in the slot.
    #             # But don't just start a job yet, we don't even know if the gpus
    #             # are available, and even if available, we don't yet know if any
    #             # of the available onese have been assigned to the the
    #             # current job in question.
    #             # .... if job.start_time == -1: ... # delay doing this for now.
    #
    #             # cbb. Below we find out if the gpus chosen for the job in
    #             # slot s are available. If avaialbe, we add them to
    #             # gpuswecangivetojob
    #             gpusrequested = self.g2j.jobsloti2gpus(s)
    #             gpuswecangivetojob.clear()
    #             job.gpuassigned_set.clear()
    #             for gpu in gpusrequested:
    #                 # make sure that the gpu is not already in use. If it is
    #                 # in use already, then do reward -= penalty.
    #                 if gpu in self.g2j.assignedgpus_set:
    #                     # Todo. Think about what the penalty should be.
    #                     #  Consider a unit cost. We want take the unit cost
    #                     #  and mke the penalty equal to numgpusXunitcost
    #                     # Penalty for choosing a gpu that is already assigned.
    #                     extrapenalties -= self.get_costofchoosingemptyslot(1)
    #                 else:
    #                     gpuswecangivetojob.add(gpu)
    #             lengpuswecangivetojob = len(gpuswecangivetojob)
    #
    #             if lengpuswecangivetojob >= job.res_vec[0]:
    #                 # It means that we have added gpus to job.gpusassigned_set
    #                 # and thus there are enough gpus available that we can
    #                 # give to the job at slot s.
    #                 if job.start_time == -1:
    #                     job.start_time = self.curr_time
    #                     # assign new color to the job and add the job to
    #                     # self.startednotdone_jobs. Reconsider to possibly differentiate
    #                     #  between started and running jobs. Just because a job
    #                     # has started, it doesn't mean it's currently running.
    #                     # Don't need to check for available colors now that we're
    #                     #  using a queue, just get an item from the fifo queue.
    #                     job.color = self.cluster.q_colormap.popleft()  # by default
    #                     self.cluster.q_colormap.append(job.color)
    #                     # Queue class raises exception if no item available.
    #                     self.g2j.startednotdone_jobs[job.id] = job
    #
    #                 gpusassigned = np.array(list(job.gpusassigned_set),
    #                                         dtype=int)
    #
    #                 self.g2j.assignedgpus_set.update(gpusassigned)
    #                 self.g2j.freegpus_set.difference_update(gpusassigned)
    #
    #                 job.ts_togo, job.ts_done = \
    #                     self.pa.distobj.jobnumrowstime(job, job.g)
    #                 res = 0  # resource type. We only have gpus as resources.
    #                 self.cluster.canvas[res, :job.ts_togo, gpusassigned] = \
    #                     job.color
    #                 # job is not assigned gpu's again, but the function below
    #                 # still works for updating cluster.avbl_slot
    #                 self.cluster.avbl_res_updateafterjobassigned(job, job.g)
    #
    #                 self.job_record.record[job.id] = job
    #                 # below should update env.g2j.job_slots as well.
    #                 self.job_slots.slots[actions] = None
    #
    #                 # dequeue backlog, because we don't want to schedule the
    #                 # same job again (e.g. pause or changes its allocation).
    #                 self.dequeue_backlog_afterallocate(actions)
    #
    #                 allocated = True
    #                 continue
    #             else:
    #                 # not enough available gpu's chosen for this job.
    #                 # Now we have to choose between two possible cases.
    #
    #                 # Case 1. Not adequate, don't allocate the gpu's to the
    #                 # job, but impose a penalty
    #                 # case 1.1. #gpus_short x unitcost
    #                 # case 1.2. #gpus_requested x unitcost
    #                 # case 1.3. #gpus_chosen but not allocated x unitcost.
    #                 #  This makes sense if we think the NN should not have
    #                 #  even chosen the gpu's it did for this job. Instead it
    #                 #  should have chosen null job for these gpu's
    #                 #  softmaxes.
    #
    #                 # Case 2. Not adequate, 'assign what we can' to the job.
    #                 # What is the penalty?
    #                 # case 2.1 #gpus_short x unit_cost
    #                 #  This has a huge negative consequence,
    #                 #  which is that the job can take super long to run if
    #                 #  numgpus requested >> numgpus statically assigned.
    #                 # case 2.2 For the cluster as a whole...
    #                 #  #gpu's not being used by the cluster x unitcost
    #
    #                 # Opinion: Do case 1.3 for now.
    #                 unitcost = self.get_unitcostofbadchoice()
    #                 if np.floor(self.pa.staticrlalloccase) == 1:
    #                     # Note, with case 1, we don't want to additionally penalize
    #                     # not using up all the gpu's, because case 1 implies we don't
    #                     # want NN to assign gpu's to a job if we can't meet the job's
    #                     # gpu requirements.
    #                     if self.pa.staticrlalloccase == 1.1:
    #                         gpusshort = job.res_vec[0] - lengpuswecangivetojob
    #                         extrapenalties += gpusshort * unitcost
    #                     if self.pa.staticrlalloccase == 1.2:
    #                         extrapenalties += job.res_vec[0] * unitcost
    #                     elif self.pa.staticrlalloccase == 1.3:
    #                         extrapenalties += lengpuswecangivetojob * unitcost
    #                     continue
    #                 elif np.floor(self.pa.staticrlalloccase) == 2:
    #                     raise NotImplementedError('Should not get here for now')
    #                 else:
    #                     raise ValueError('self.pa.staticrlalloccase of %f is '
    #                                      'invalid' % self.pa.staticrlalloccase)
    #
    #         if self.pa.debug_actions:
    #             self.writeoutput_debug_dict_shared('After Allocate', 2)
    #         # Should we always move on or allow another allocation step to
    #         # take place before we increment env.step???? For now,
    #         # just always move on. Let's hope that
    #         status = 'MoveOn'
    #
    #     if status == 'MoveOn':
    #         # Once we are ready to move on,
    #         # debuginfo('self.curr_time: %d' % self.curr_time)
    #         if self.curr_time == 63:
    #             pass
    #         self.findlimitingbws_static()
    #         printfirstjob = True
    #         # After finding the limitingbws for each job, we can now advance
    #         # each chosen job's compistdone field.
    #
    #         for jid in self.g2j.startednotdone_jobs:
    #             job = self.g2j.startednotdone_jobs[jid]
    #             job.v_m, job.rt_m = self.job_minbatch_speed_calc(
    #                 job, 'multi', outdetails=True)
    #
    #             d_delta = self.pa.distobj.jobadvancecompdist(job)
    #             job.d_done += d_delta
    #
    #             self.debug_step_jobspeed(job)
    #
    #             job.stepcounter += 1  # count steps taken for the job
    #             if printfirstjob:
    #                 self.debug_step_firstjobinfo(job)
    #                 printfirstjob = False
    #             pass
    #
    #         # before we finish moving on, we we need to get reward for
    #         # the current step, before we increment the time step and
    #         # prepare the next state as input to the neural network.
    #         reward = self.get_reward_static()
    #
    #         # cbb. now move on in the sequence (so must first advance the time
    #         # step). i.e. prepare the environment for next time step.
    #         self.curr_time += 1
    #         # Even if no gpus were allocated this round, existing jobs may
    #         # have progressed in their compdist so always check to remove
    #         # done jobs. Below also adds a job's calculated negslowdown to the
    #         # environment's negslowdown.
    #         self.remove_done_jobs_static(self.curr_time - 1, updatefields=True)
    #
    #         self.extra_info.time_proceed()  # okay. Should not need to change.
    #
    #         # add new jobs. Advance one step further in current job sequence.
    #         self.seq_idx += 1
    #
    #         self.clear_redraw_canvas()
    #         # cbb. check if by moving on we have reached the end. Mao never
    #         # changes self.end. He sets it during launch to signal whether to
    #         # finish after just a single job sequence (self.end == "no_more_jobs_in_seq")
    #         # OR also require in addition that 1) all started jobs finish,
    #         # 2) there are no more jobs in job_slots.slots, and 3) that there
    #         # are no more jobs in job_backlog.backlog.
    #         if self.end == "no_more_jobs_in_seq":  # end of new job sequence
    #             # ^-cbb. self.end is either "no_more_jobs_in_seq" or "do_all_jobs_in_seq"
    #             if self.seq_idx >= self.pa.t_finaljobarrival:
    #                 done = True
    #         elif self.end == "do_all_jobs_in_seq":  # everything has to be finished
    #             # V-cbb. If the following multiple conditions are not met,
    #             # then it means that in reality, we're not all done. so
    #             # 'done' variable will remain as false.
    #             # NOTE!!! pg_re is launched with end="do_all_jobs_in_seq"!!!
    #             if self.seq_idx >= self.pa.t_finaljobarrival and \
    #                     len(self.g2j.startednotdone_jobs) == 0 and \
    #                     all(s is None for s in self.job_slots.slots) and \
    #                     len(self.job_backlog.backlog) == 0:
    #                 done = True
    #             elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
    #                 done = True
    #             # ^-cbb. else done remains False.
    #
    #         if not done:
    #             # cbb. e.g. self.end == "do_all_jobs_in_seq" in reality, done is still
    #             # false, because, we haven't even started yet.
    #             #  e.g. self.seq_idx === 1 (note: self.pa.simul_len === 50t)
    #             # (i.e. there are no jobs in the slots).
    #             # cbb. not done and 0 <= seq_idx <= 49 is less then t_finaljobarrival === 50
    #             # then see if a job has arrived for this time step (i.e. new_job.len > 0)
    #
    #             # if t_finaljobarrival not reached, but no job, get a job.
    #             if self.seq_idx < self.pa.t_finaljobarrival:  # otherwise, end of new job sequence,
    #                 # ^-cbb. If the above is true, it means we've reached
    #                 # the end of the new job sequence, which means no new jobs.
    #                 # cbb. self.end == "do_all_jobs_in_seq" but self.seq_idx=1,
    #                 # note: self.pa.sumu_len=50t.
    #                 # So we aren't actually done, so get a new job from the
    #                 # job sequence.
    #                 if self.pa.testjobmodel:
    #                     new_job = self.get_new_job_from_seq(self.seq_no,
    #                                                         self.seq_idx,
    #                                                         testjob=True)
    #                 else:
    #                     new_job = self.get_new_job_from_seq(self.seq_no,
    #                                                         self.seq_idx)
    #
    #                 if new_job.len > 0:  # a new job comes
    #                     # cbb. Since we haven't allocated a job, we are
    #                     # either putting a new job into one of the M job
    #                     # slots or putting it into the backlog.
    #
    #                     to_backlog = True  # cbb. set true for now
    #
    #                     # V-cbb. See if the new job we sampled can be
    #                     # put into one of the M job slots.
    #                     # This for loop would not be necessary if after a
    #                     # finished job is deleted from its slot, we simply do
    #                     #  slots[emptyslot:-1] = slots[emptyslot+1:]. Maybe
    #                     # do this later. The following for loop is okay for now.
    #
    #                     for i in range(self.pa.num_jobslots):
    #                         if self.job_slots.slots[i] is None:
    #                             # put in new visible job slots.
    #                             # i.e. because the i'th job slot is empty.
    #                             to_backlog = False
    #                             # cbb. not all the mutations by the below
    #                             # function are neccessary because the below
    #                             # function uses g2j._x but static case
    #                             # doesn't use g2j._x.
    #                             self.g2j.assigntoslots_updatefields_static(
    #                                 i, new_job)
    #
    #                             self.job_record.record[new_job.id] = new_job
    #                             break
    #
    #                     # V-cbb. See if the new job that we sampled can be
    #                     # put into the backlog.
    #                     if to_backlog:
    #                         self.tobacklog(new_job)
    #
    #                     # ^-cbb. As can be seen in the for loop and if
    #                     # statement above, a job is added to
    #                     # job_record.record when it is placed into a slot,
    #                     # or a backlog. But as seen below, also, a job is added
    #                     #  to the record, even when it's being allocated.
    #                     # Thus there is no data structure that only contains
    #                     # the jobs currently running in the
    #                     # 'cluster'/machine. We must make one. For our case
    #                     # though, the currently running jobs can be queried
    #                     # with: self.g2j.chosenjobs_slots_int,
    #                     # which prefilters for slots that actually have jobs.
    #
    #                     # V-cbb. We simply set the 'extra_info' object's
    #                     # time_since_last_new_job field to 0. (This is done
    #                     # when a new job arrives, not when it is allocated).
    #                     self.extra_info.new_job_comes()
    #
    #                     # in this nested if statement, we have taken a new
    #                     # job that has arrived, and placed it either in one
    #                     # of the M job slots, or to the backlog. Pretty much
    #                     # the only thing left to do is to calculate the
    #                     # reward for this time step.
    #
    #                 else:  # cbb. no job for this time point in the job
    #                     # sequence. Nothing to do. We already calculated the
    #                     # reward for the current step, and we just move on
    #                     pass
    #             # print(debugstr + ", " +
    #             #       "Status MoveOn.\t action: {}, slotsbool: {}".format(
    #             #         action, slotsbool))
    #             if self.pa.debug_actions:
    #                 self.writeoutput_debug_dict_shared('Endof MoveOn',
    #                                                    stackupby=1)
    #
    #     ob = self.observe()
    #
    #     # V-cbb. info holds the environment's job record, which is useful for
    #     #  calculating slow down of jobs of a trajectory.
    #     info = self.job_record
    #
    #     if done:
    #         self.seq_idx = 0
    #
    #         if not repeat:
    #             # ^-cbb. if repeat True, then keep self.seq_no the same.
    #             # if repeat is False (as is inside the if statement here),
    #             # then advance self.seq_no by one, but also overflow if
    #             #  seq_no becomes larger than num_jobsets.
    #             self.seq_no = (self.seq_no + 1) % self.pa.num_jobsets
    #
    #         self.reset()
    #
    #     if self.render:
    #         self.plot_state()
    #
    #     # cbb. Remember, that if done is true, pg_re_*.run_episode, which builds
    #     #  a single episode trajectory will break ( # i.e, it will stop
    #     # building the trajectory for a single episode )
    #
    #     self.extrapenalties = extrapenalties
    #     if self.pa.applyextrapenalties:
    #         reward += self.extrapenalties
    #
    #     return ob, reward, done, info

    @abstractmethod
    def writemore_debug_dict_child(self):
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def make_debug_str_child(self):
        raise NotImplementedError("Subclasses should implement this!")

    def writeoutput_debug_dict_shared(self, status, stackupby=1):
        self.pa.debug_dict['slots'] = self.g2j.jobsinslots_fordebug_fullpre()
        self.pa.debug_dict['status'] = status

        # write more to sel.fpa.debug_dict, specific to the env child class
        self.writemore_debug_dict_child()
        # self.output_debug_actions_oneepis_onejobset_shared(stackupby=stackupby)

        debug = self.pa.debug_actions and \
                self.debug_step_firstlastiter_lastjobset_selecteps()

        if debug:  # output if debug is true
            debugstr = self.make_debug_str_child()
            if 'backlogfull' in self.pa.debug_dict:
                if self.pa.debug_dict['backlogfull']:
                    debugstr += ', Backlog Full'
            debuginfo(debugstr, stackupby=stackupby)

    def make_debug_str_one_action(self):
        pass
        debugstr = ("step: {:3d}, " +
                    "==== env.curr_time: {:3d}, ==== " +
                    "sched_type: {:s}, Status: {:s}, " +
                    "action: {:d}, " +
                    "\n" + "\t" * 5 +
                    "=" * 5 + " running: {:s}, " +
                    "=" * 5 + " slots: {:s}").format(
                        self.pa.debug_dict['step'],
                        self.curr_time,
                        self.pa.debug_dict['sched_type'],
                        self.pa.debug_dict['status'],
                        self.pa.debug_dict['action'],
                        str(self.pa.debug_dict['running']),
                        str(self.pa.debug_dict['slots'])
                    )
        return debugstr

    def make_debug_str_manyactions(self):
        pass
        debugstr = ("step: {:d}, " +
                    "==== env.curr_time: {:3d}, ==== " +
                    "sched_type: {:s}, Status: {:s}, " +
                    "actions: {:s}, " +
                    "\n" + "\t" * 5 +
                    "=" * 5 + " running: {:s}, " +
                    "=" * 5 + " slots: {:s}").format(
                        self.pa.debug_dict['step'], self.curr_time,
                        self.pa.debug_dict['sched_type'],
                        self.pa.debug_dict['status'],
                        str(self.pa.debug_dict['action']),
                        str(self.pa.debug_dict['running']),
                        str(self.pa.debug_dict['slots'])
                    )
        return debugstr

    # class Env
    def tobacklog(self, new_job):
        if self.job_backlog.curr_size < self.pa.backlog_size:
            self.job_backlog.backlog.append(new_job)
            self.job_record.record[new_job.id] = new_job
            self.pa.debug_dict['backlogfull'] = False
        else:  # abort, backlog full
            self.pa.debug_dict['backlogfull'] = True
            # print("Backlog is full.")

    # class Env
    def chosenslotsthathavejobs(self):
        """
        Memorable and easily understood function to return only the indices of
        the slots that have been chosen by neural network AND have jobs.

        :return: Indices of slots that were chosen by actions AND have jobs
        :rtype: np.ndarray of int
        """
        return self.g2j.chosenjobs_slots_int

    # class Env
    def runningjobs_slots_fullpreemption(self):
        """
        Easy to remember function name to get running jobs by slot indices

        :return: Indices of job slots that have jobs that are currently
            actively runing on the cluster (i.e. because they were chosen
            during the current step)

        :rtype: np.ndarray of int
        """
        return self.chosenslotsthathavejobs()

    # class Env
    def runningjobs_fullpreemption(self):
        """

        :return: list of jobs
        :rtype: list
        """
        # since we converted slots to an ndarray, we can just slice.
        # [self.job_slots.slots[s] for s in
        # self.runningjobs_slots_fullpreemption()]
        jobs = [self.job_slots.slots[i] for i in self.g2j.chosenjobs_slots_int]
        # return list(self.job_slots.slots[self.g2j.chosenjobs_slots_int])
        return jobs

    def runningjobs_static(self):
        """

        :return: list of jobs
        :rtype: list
        """
        # startednotdone_jobs is a dict
        return self.g2j.startednotdone_jobs.values()

    # class Env
    @abstractmethod
    def runningjobs_list_abstmeth(self):
        return NotImplementedError("Should be implemented by child Env class")

    # class Env
    def dequeue_backlog_afterallocate(self, chosen_slot, updatefields=False):
        """
        Called by Env.step by Mao once we know we've allocated one job.
        Here we treat slots as an numpy vector.

        :param chosen_slot:
        :type chosen_slot:
        :param updatefields: Whether or relevant fields should be updated after
            a change to g2j.job_slots.slots
        :type updatefields: bool
        :return: True if dequeued from backlog to job_slots.slots,
            False otherwise (because nothing in backlog)
        :rtype: bool
        """
        # de-queue backlog. cbb. understood. For us, we only do this when
        # a job is finished and thus a slot becomes empty.
        if self.job_backlog.curr_size > 0:
            # instead of list, consider using python's double ended queue.
            # Mao used a list, but that seems inappropriate.
            self.g2j.assigntoslots(chosen_slot,
                                   self.job_backlog.backlog.popleft())
            # self.g2j.job_slots.slots.append(self.job_backlog.backlog.popleft())
            if updatefields:
                self.g2j.updatefields_after_slot_update()
                self.g2j.updatefields_after_x_slots_update()
            self.pa.debug_dict['backlogfull'] = False
            return True
        else:
            return False

    def dequeue_backlog_afterallocate2(self, updatefields=False):
        """
        Similar to `dequeue_backlog_afterallocate_old`, but here we treat both
        backlog and slots as a deque, so we don't need to know what slot a
        job was deleted from.
        :param updatefields:
        :type updatefields:
        :return:
        :rtype:
        """
        if self.job_backlog.curr_size > 0:
            self.g2j.job_slots.slots.append(self.job_backlog.backlog.popleft())
            if updatefields:
                self.g2j.updatefields_after_slot_update()
                self.g2j.updatefields_after_x_slots_update()
            self.pa.debug_dict['backlogfull'] = False
            return True
        else:
            return False

    # class Env
    def action2jobid(self, action):
        """
        Return job id given a single action.

        slots is an ndarray so action must be ndarray integer type or single
        integer number
        :param action:
        :type action:
        :return:
        :rtype:
        """
        return self.job_slots.slots[action].id

    # class Env
    def job_record_addjobtoit(self, chosenslot):
        """
        Update the job_record for the job allocated above (e.g. job has
        updated start and end times, but unlike Mao, do not remove the job
        from the job slot.

        Remember, job_record.record is a dictionary. Key's are job id's
        :param chosenslot: the slot in which a job exists and the job was chosen
        for allocation
        :type chosenslot: int
        :return:
        :rtype:
        """
        self.job_record.record[self.job_slots.slots[chosenslot].id] = \
            self.job_slots.slots[chosenslot]

    # class Env
    def remove_done_jobs(self, finishtime, updatefields=True):
        """
        This function also checks the cluster.startednotdone_jobs dict to see
        if any jobs have 'finished', i.e, their finish time has been reached,
        and if so, remove the job from the cluster.startednotdone_jobs dict.

        In our case, we must also remove the job from the job slot,
        and dequeue the backlog onto an available slot (probably onto the
        slot we just emptied is easiest, rather than shifting all the jobs
        over).

        The above tasks should be done here, instead of in the env.step()
        function. Mao dequeues from the backlog to the newly empty slot in
        env.step() because Mao empties a slot during the allocation phase,
        which is done in env.step(), unlike us.

        :return: True if job_slot has been changed, False otherwise
        :rtype: bool
        """

        # full or partial pre-emption precludes the use of
        # self.cluster.avbl_res_moveupone()

        # cbb. check cluster.startednotdone_jobs to see if any jobs have 'finished'.
        #  Remember, this function is called after env.curr_time is advanced
        # by 1.

        job_slots_changed = False
        printfirstjob = True
        # cbb. Normally we would look at gpu2job's x table. But in general we
        #  should look at "running jobs" data structure.
        removedjobs = []
        # IMPORTANT! Must used reversed because deleting from middle of the
        # queue shifts down the indices of jobs in the latter half.
        for s in reversed(self.g2j.chosenjobs_slots_int):
            # ^-cbb. For now self.gpu2job.chosenjobs_slots_int is equivalent to
            # the running jobs, because with full premption, we can always
            # run jobs that exist in chosen slots. Very important point!!!!
            job = self.g2j.sloti2job(s)  # slots is now an ndarray
            assert job is not None
            if job.d_done >= job.d_f:
                # Job is done, so ...
                # set Job's finish time. Subtract one, since curr_time has
                # advanced forward by one already to prepare for the next
                # step. But the current step really is really at curr_time - 1.

                job.finish_time = finishtime  # we expect finish time to
                # correspond to the greatest row for which the job existed

                if printfirstjob:
                    self.debug_step_removejob_firstjobinfo(job)
                    printfirstjob = False

                # remove from env.startednotdone_jobs and
                # env.job_slots.slots
                self.g2j.startednotdone_jobs.pop(job.id)

                # ^-cbb. Simply del the job from the slots deque,
                # using index. We probably can't use deque.remove(x) because
                # x is not hashable?
                del self.g2j.job_slots.slots[s]
                # self.g2j.assigntoslots(s, None)
                # ^-cbb. Removing from a list Takes too long. Just set to
                # None. Even Mao doesn't use list.remove for
                # env.job_slots.slots in env.step. He only uses list.remove for
                # env.startednotdone_jobs.

                # dequeue the backlog onto the newly empty slot.
                # For us, we only do this when a job is finished and thus a
                # slot becomes empty. We need some sort of linked list data
                # structure to delete stuff from the middle when number of
                # slots is huge. Python uses an array to implement a list,
                # which is bad for deleting from the middle. It's simplest
                # to just dequeue from the backlog onto the newly empty slot.
                #  But for small number of slots, can easily just do slots[
                # emptyslot:-1] = slots[emptyslot+1:].
                self.dequeue_backlog_afterallocate2(updatefields=False)
                # would normally update G2J.jobslotvec here but
                # G2J.jobslotvec is an on the fly comptuted property so
                # no need.

                # update g2j.x to reflect the release of GPUs
                gpustorelease = job.gpusassigned_set
                self.g2j.x_update_releasegpus(gpustorelease)

                job_slots_changed = True
                removedjobs.append(job)

                self.g2j.assignedgpus_set.difference_update(
                    job.gpusassigned_set)
                self.g2j.freegpus_set.update(job.gpusassigned_set)
                job.gpusassigned_set.clear()

        if job_slots_changed and updatefields:
            self.g2j.updatefields_after_x_update()
            self.g2j.updatefields_after_slot_update()
            self.g2j.updatefields_after_x_slots_update()

        # update graphical representation. Important for setting up the state
        #  for next time point in the static allocation case. In the
        # partial and full pre-emption case, it's useless because we wipe the
        #  canvase before allocation.
        # cbb. shift the canvas "image" up by one.
        # self.cluster.canvas[:, :-1, :] = self.cluster.canvas[:, 1:, :]
        # cbb. Last time step should be empty after the shift.
        # self.cluster.canvas[:, -1, :] = 0

        return removedjobs

    # class Env
    def remove_done_jobs_static(self, finishtime, updatefields=True):
        """
        This function also checks the cluster.startednotdone_jobs dict to see
        if any jobs have 'finished', i.e, their finish time has been reached,
        and if so, remove the job from the cluster.startednotdone_jobs dict.

        In our case, we must also remove the job from the job slot,
        and dequeue the backlog onto an available slot (probably onto the
        slot we just emptied is easiest, rather than shifting all the jobs
        over).

        The above tasks should be done here, instead of in the env.step()
        function. Mao dequeues from the backlog to the newly empty slot in
        env.step() because Mao empties a slot during the allocation phase,
        which is done in env.step(), unlike us.

        :return: True if job_slot has been changed, False otherwise
        :rtype: bool
        """

        # full or partial pre-emption precludes the use of
        # self.cluster.avbl_res_moveupone()

        # cbb. check cluster.startednotdone_jobs to see if any jobs have 'finished'.
        #  Remember, this function is called after env.curr_time is advanced
        # by 1.

        job_slots_changed = False
        printfirstjob = True
        # cbb. Normally we would look at gpu2job's x table. But in general we
        #  should look at "running jobs" data structure.
        jobstoremove = []
        for jid in self.g2j.startednotdone_jobs:
            # ^-cbb. For now self.gpu2job.chosenjobs_slots_int is equivalent to
            # the running jobs, because with full premption, we can always
            # run jobs that exist in chosen slots. Very important point!!!!
            job = self.g2j.startednotdone_jobs[jid]
            assert job is not None
            if job.d_done >= job.d_f:
                # Job is done, so ...
                # set Job's finish time. Subtract one, since curr_time has
                # advanced forward by one already to prepare for the next
                # step. But the current step really is really at curr_time - 1.

                job.finish_time = finishtime  # we expect finish time to
                # correspond to the greatest row for which the job existed

                if printfirstjob:
                    self.debug_step_removejob_firstjobinfo(job)
                    printfirstjob = False

                jobstoremove.append(job)

                # CBB. How do we empty the slot ??? For static case,
                # it's already emptied!! so updating slot with non here is
                # not needed.
                # ^-cbb. Poping from a dict is fast
                # self.g2j.assigntoslots(s, None)
                # ^-cbb. Removing from a list Takes too long. Just set to
                # None. Even Mao doesn't use list.remove for
                # env.job_slots.slots in env.step. He only uses list.remove for
                # env.startednotdone_jobs.

                # dequeue the backlog onto the newly empty slot.
                # For us, we only do this when a job is finished and thus a
                # slot becomes empty. We need some sort of linked list data
                # structure to delete stuff from the middle when number of
                # slots is huge. Python uses an array to implement a list,
                # which is bad for deleting from the middle. It's simplest
                # to just dequeue from the backlog onto the newly empty slot.
                #  But for small number of slots, can easily just do slots[
                # emptyslot:-1] = slots[emptyslot+1:].

                # CBB. Don't need to dequque from the backlog since that is
                # done after an allocate step for static case.
                # self.dequeue_backlog_afterallocate(s, updatefields=False)
                # would normally update G2J.jobslotvec here but
                # G2J.jobslotvec is an on the fly comptuted property so
                # no need.
                # job_slots_changed = True

        # don't need to do this, because we are not concerned about .x for
        # tetris and sjf since they only choose one job at a time.
        # if job_slots_changed and updatefields:
        #     self.g2j.updatefields_after_slot_update()
        #     self.g2j.updatefields_after_x_slots_update()

        # update graphical representation. Important for setting up the state
        #  for next time point in the static allocation case. In the
        # partial and full pre-emption case, it's useless because we wipe the
        #  canvase before allocation.
        # cbb. shift the canvas "image" up by one.
        # self.cluster.canvas[:, :-1, :] = self.cluster.canvas[:, 1:, :]
        # cbb. Last time step should be empty after the shift.
        # self.cluster.canvas[:, -1, :] = 0
        for job in jobstoremove:
            self.g2j.assignedgpus_set.difference_update(job.gpusassigned_set)
            self.g2j.freegpus_set.update(job.gpusassigned_set)
            job.gpusassigned_set.clear()
            self.g2j.startednotdone_jobs.pop(job.id)
            # cbb. We know that for the static case set of startednotdone
            # jobs is equal to running jobs. So also increment the
            # cluster.numavbl_res
            # Actually, don't change cluster.avbl_res. Expect it to be
            # reset/redrawn in the scope of the function that calls this one.

        return jobstoremove

    # class Env
    def clear_redraw_canvas(self):
        """
        Clear and Redraw the cluster image (aka canvas). This is confirmed to
        works for both full pre-emption and all static cases. Not sure about
        partial pre-emption.
        :return:
        :rtype:
        """

        # cbb. clear the canvas so we can paint it fresh soon.
        self.cluster.canvas[:] = 0

        # Even though we do static, since we redraw the jobs that take up
        #  the first row time step, we can reset the numavbl_res variable.
        self.cluster.avbl_res_reset()

        # cbb. Begin Redrawing startednotdone jobs.
        for job in self.runningjobs_list_abstmeth():
            # cluster.startednotdone_jobs dict might be empty such as
            # when we just start or all the started jobs finished from
            # the previous step's MoveOn stage simultaneously, but that's
            # okay.

            if len(job.gpusassigned_set) > 0:
                # it's best to make job.numgpsallocated be a property
                # based on len(job.gpusassigned_set)
                # Note: numgpusallocated and numgpusassigned might mean
                # different things, consider settling on definitions.

                # We don't need an order for the jobs to be redraw, we can
                # simply draw each job according to the list of GPU's
                # assigned, starting from the first time row.
                # cbbb. Important, just redraw first the jobs
                job.ts_togo, job.ts_done = \
                    self.pa.distobj.jobnumrowstime(job, job.g)
                gpusassigned = np.array(list(job.gpusassigned_set),
                                        dtype=int)
                res = 0  # resource type. We only have gpus as resources.
                self.cluster.canvas[res, :job.ts_togo, gpusassigned] = job.color
                # job is not assigned gpu's again, but the function below
                # still works for updating cluster.avbl_slot
                self.cluster.avbl_res_updateafterjobassigned(job, job.g)

    # class Env
    def advance_runningjobs_onestep(self, printfirstjob=True):
        """

        :param printfirstjob:
        :type printfirstjob:
        :return:
        :rtype:
        """
        for job in self.runningjobs_list_abstmeth():
            job.v_m, job.rt_m = self.job_minbatch_speed_calc(
                job, 'multi', outdetails=True)

            d_delta = self.pa.distobj.jobadvancecompdist(job)
            job.d_done += d_delta

            self.debug_step_jobspeed(job)

            job.stepcounter += 1  # count steps taken for the job
            if printfirstjob:
                self.debug_step_firstjobinfo(job)
                printfirstjob = False

    # class Env
    def job_minbatch_speed_calc(self, job, singleormulti='multi',
                                outdetails=False):
        """
        Return the job's average speed per mini batch.

        The job's 'single' is the job's ring reduction speed when the job is
        running alone, i.e job.scale is 1. The single speed will be used as
        the penalty for delaying a job in one of the slots (i.e. job is not
        assigned any gpus), and also for dismming a job (i.e. job is in the
        backlog). Since a non assigned job has no gpu's assigned, we cannot
        define dimx/dimy/dimz, so we just use the ring reduction formula.

        Thus the single speed of a job is like an opportunity cost of not
        running the job with the requested number of gpus. It is calculated
        as a cost when calculating the reward for jobs that are in the queue
        or backlog and not assigned.

        The job's 'multi' speed is the job's ring reduction speed that takes
        into consideration a non unit value for job.scale.

        Job.scale is the limiting bandwidth for a job when it is running
        alone (i.e. jobcount for all edges is 1) aka limitingbwsingle,
        divided by the limiting bandwidth for a job when considering non unit
        jobcounts for edges, aka limitingbwmulti.

        :param job:
        :type job: Job
        :param singleormulti: Either 'single' or 'multi'
        :type singleormulti: str
        :param outdetails: Whether or not to return more than one value
        :type outdetails: bool
        :return: job speed per mini batch
        :rtype: float
        """

        if singleormulti == 'single':  # pretend speed if job running alone.
            # this formulation is used for when a job is not running but we
            # want to calculated the job's average speed when running alone,
            # i.e. to calcualte the opportunity cost of not running the job
            # and keeping the job in the either the queue or the backlog.
            rt_m = self.cluster.get_ret_asym(job.gradsize, job.res_vec[0])
            rt_m = rt_m * 0.001  # convert from ms to s
            rt_m = rt_m * self.pa.ret_reducer
            minbatch_speed = job.d_m * 1.0 / (job.tt_m + 1.0 * rt_m)
            # treat res_vec[0] as min #gpu's requested.
            minbatch_speed *= job.res_vec[0]

        elif singleormulti == 'multi':  # actual speed
            # job.scale is computed when env.findlimitingbws() is called.
            assert ~np.isnan(job.scale)
            rt_m = self.cluster.get_ret_asym(job.gradsize, job.g)
            rt_m = rt_m * 0.001  # convert from ms to s
            rt_m = rt_m * self.pa.ret_reducer
            minbatch_speed = job.d_m * 1.0 / (job.tt_m + 1.0 * job.scale *
                                              rt_m)
            minbatch_speed *= job.g
        else:
            raise ValueError("Input parameter 'singleormulti' must be either "
                             "'single or 'multi'")

        assert ~np.isnan(minbatch_speed)
        if not outdetails:
            return minbatch_speed
        else:
            return minbatch_speed, rt_m

    # class Env
    def get_jobspeed_fromsloti(self, slot_int):
        """
        Calculates the average speed of the job per minibatch.

        The numerator includes a multiplication by the current number of gpus
        assigned to the job of interest.

        Currently, we only consider ring reduction. Later when we solve how
        to do reduction algorithm assignment, we will need to modify this
        function.

        :param slot_int: Job slot that holds the job of interest
        :type slot_int: int
        :return: Average speed of the job per minibatch
        :rtype:
        """
        j = self.job_slots.slots[slot_int]
        j.v_m = self.job_minbatch_speed_calc(j, 'multi')
        return j.v_m

    # class Env
    def findlimitingbws_static(self):
        """
        This function will find a job's limiting bandwidths, both for single
        and multiple job scenarios.

        Since findlimitingbws is an expensive task, we should only do it if
        there are changes to jobs running on the cluster.

        :return:
        :rtype:
        """
        snc = self.snc
        g = snc.g1
        # reset jobscount attribute for all edges of the network
        nx.set_edge_attributes(g, 0, 'jobcount')
        # Since dictofedgesets is a dict, and we are building up this dict
        # fresh every step, we can simply clear it first.
        self.g2j.dictofedgesets = {}

        # since we are not dealing with job slots since running jobs are
        # removed from the job slots, just get total number of jobs running
        # and use the range(numberofjobsrunning) as a proxy for slot s_int

        for jid in self.g2j.startednotdone_jobs:
            job = self.g2j.startednotdone_jobs[jid]
            # we ultimately want to increment the jobs count for the links in
            #  the network. We first need the gpus per job. The from the gpus
            #  pairs per job, find the set of links(edges) per job. Then
            # increment the jobscount attribute of each edge of a job once.
            # Then we need to iterate through the links of a job again to
            # determine singlejoblimbw and multijoblimbw.

            # get gpu index from gpu2job._x table
            gpuis_from_x = job.gpusassigned_set  # selected gpus

            # Only bother trying to build up the edge set if numgpus assigned
            #  to job is greater than 1. Otherwise do nothing for now,
            # which should keep the resetted edge set empty.
            if len(gpuis_from_x) > 1:
                # look at unique gpu pairs using two for loops and an if statement.
                # build up the set of links (edges) per job. We already built the
                #  shortpaths using NetworkX. Shorpaths is a dict from 'gpua_str,
                #  gpub_str':list of integers mapping. The value of the mapping
                # includes the gpus of the pair as first and last elements of the
                #  list.
                for a in gpuis_from_x:  # a is an integer
                    for b in gpuis_from_x:  # b is an integer
                        if a < b:  # only look at unique pairs
                            # we will build up the set of edges between the gpu
                            # pair and add it to the slot's gpuset

                            # first convert from int to str
                            # 'nodeid' is the id of a node in the graph object
                            # 'a' or 'b' is the gpu index in the _x table.
                            # gpunodeid2xrowi[index a] is a 2-tuple. First
                            # element is the index of the node in the graph,
                            # second element is the gpu number from 0 .. numgpus-1.
                            anodeid = snc.gpunodeid2xrowi[a][0]
                            bnodeid = snc.gpunodeid2xrowi[b][0]
                            astr = snc.gintstr[anodeid]
                            bstr = snc.gintstr[bnodeid]
                            # add edges as a tuples since tuples are hashable
                            gpupairstr = ','.join([astr, bstr])
                            path = snc.shortpaths[gpupairstr]  # list of numbers
                            for i in range(len(path) - 1):
                                edge = (path[i], path[i + 1]) if path[i] < path[
                                    i + 1] else (path[i + 1], path[i])
                                # finally add (int,int) tuple as edge to slot's set
                                if jid not in self.g2j.dictofedgesets:
                                    self.g2j.dictofedgesets[jid] = set()
                                self.g2j.dictofedgesets[jid].add(edge)

                # now that the job slot's edgeset is complete, we can visit each
                # edge (in the graph object) in the corresponding job's edgeset
                # and increase the jobcount attribute by 1.
                for nodea, nodeb in self.g2j.dictofedgesets[jid]:
                    g.edges[nodea, nodeb]['jobcount'] += 1

                # After increasing the jobcount for the job's edges, we move on
                #  to the next job
        # After increasing the job count for all jobs' edges, we iterate
        # through each job again, and find both singlejoblimbw and multijoblimbw
        for jid in self.g2j.startednotdone_jobs:  # s_int is job slot
            job = self.g2j.startednotdone_jobs[jid]
            # get gpu index from gpu2job._x table. Once we know our code
            # works, assert that len(gpuis_from_x) == job.g
            # job.g can be used instead without assertion.

            assert job.g >= 1
            if job.g == 1:
                job.singlejoblimbw = np.Inf
                job.multijoblimbw = np.Inf
                job.scale = 0
            else:
                job.singlejoblimbw = np.Inf
                job.multijoblimbw = np.Inf
                for edge in self.g2j.dictofedgesets[jid]:
                    speed = g.edges[edge]['speed'] / 1.0
                    if speed < job.singlejoblimbw:
                        job.singlejoblimbw = speed
                    multispeed = speed * 1.0 / g.edges[edge]['jobcount']
                    if multispeed < job.multijoblimbw:
                        job.multijoblimbw = multispeed
                job.scale = job.singlejoblimbw / job.multijoblimbw
            assert ~np.isnan(job.singlejoblimbw)
            assert ~np.isnan(job.multijoblimbw)
            assert ~np.isnan(job.scale)

    # class Env
    # Todo modify
    def findlimitingbws(self):
        """
        This function will find a job's limiting bandwidths, both for single
        and multiple job scenarios.

        DONE for now. This function needs access to the jobs currently running
        on the cluster as well as the cluster.snc object. We might be able to
        put this function in the Cluster class. Confirm later.

        Since findlimitingbws is an expensive task, we should only do it if
        there are changes to jobs running on the cluster.

        :return:
        :rtype:
        """
        snc = self.snc
        g = snc.g1
        # reset jobscount attribute for all edges of the network
        nx.set_edge_attributes(g, 0, 'jobcount')
        # reset the edge sets
        self.g2j.dictofedgesets = {}

        # Important change. since chosejobs_slots_int can be different from
        # initially chosen jobs, we will pass in the set of jobs that we want
        #  to calculate the speeds for.
        for s_int in self.g2j.job_queue.lst_running_jobs():  # s_int is job slot number
            # get job object from slot int.
            job = self.g2j.sloti2job(s_int.id)
            jid = job.id
            # we ultimately want to increment the jobs count for the links in
            #  the network. We first need the gpus per job. The from the gpus
            #  pairs per job, find the set of links(edges) per job. Then
            # increment the jobscount attribute of each edge of a job once.
            # Then we need to iterate through the links of a job again to
            # determine singlejoblimbw and multijoblimbw.

            # get gpu index from gpu2job._x table
            # gpuis_from_x = self.g2j.jobsloti2gpus(s_int)  # selected gpus
            gpuis_from_x = job.gpusassigned_set  # selected gpus
            # gpunodeid2xrowi = snc.gpunodeid2xrowi  # all gpus

            # Only bother trying to build up the edge set if numgpus assigned
            #  to job is greater than 1. Otherwise do nothing for now,
            # which should keep the resetted edge set empty.
            if len(gpuis_from_x) > 1:
                # look at unique gpu pairs using two for loops and an if statement.
                # build up the set of links (edges) per job. We already built the
                #  shortpaths using NetworkX. Shorpaths is a dict from 'gpua_str,
                #  gpub_str':list of integers mapping. The value of the mapping
                # includes the gpus of the pair as first and last elements of the
                #  list.
                for a in range(len(gpuis_from_x)):  # a is an integer
                    for b in range(len(gpuis_from_x)):  # b is an integer
                        if a < b:  # only look at unique pairs
                            # we will build up the set of edges between the gpu
                            # pair and add it to the slot's gpuset

                            # first convert from int to str
                            # 'nodeid' is the id of a node in the graph object
                            # 'a' or 'b' is the gpu index in the _x table.
                            # gpunodeid2xrowi[index a] is a 2-tuple. First
                            # element is the index of the node in the graph,
                            # second element is the gpu number from 0 .. numgpus-1.
                            anodeid = snc.gpunodeid2xrowi[a][0]
                            bnodeid = snc.gpunodeid2xrowi[b][0]
                            astr = snc.gintstr[anodeid]
                            bstr = snc.gintstr[bnodeid]
                            # add edges as a tuples since tuples are hashable
                            gpupairstr = ','.join([astr, bstr])
                            path = snc.shortpaths[gpupairstr]  # list of numbers
                            for i in range(len(path) - 1):
                                edge = (path[i], path[i + 1]) if path[i] < path[
                                    i + 1] else (path[i + 1], path[i])
                                # finally add (int,int) tuple as edge to slot's set
                                if jid not in self.g2j.dictofedgesets:
                                    self.g2j.dictofedgesets[jid] = set()
                                self.g2j.dictofedgesets[jid].add(edge)

                # now that the job slot's edgeset is complete, we can visit each
                # edge (in the graph object) in the corresponding job's edgeset
                # and increase the jobcount attribute by 1.
                for nodea, nodeb in self.g2j.dictofedgesets[jid]:
                    g.edges[nodea, nodeb]['jobcount'] += 1

                # After increasing the jobcount for the job's edges, we move on
                #  to the next job
        # After increasing the job count for all jobs' edges, we iterate
        # through each job again, and find both singlejoblimbw and multijoblimbw

        # Todo: Loop the running jobs, update the assignment table
        for s_int in self.g2j.job_queue.lst_running_jobs():  # s_int is job slot number
            # get job object from slot int.
            job = self.g2j.sloti2job(s_int.id)
            jid = job.id
            # get gpu index from gpu2job._x table. Once we know our code
            # works, assert that len(gpuis_from_x) == job.g
            # job.g can be used instead without assertion.
            # gpuis_from_x = self.g2j.jobsloti2gpus(s_int)  # selected gpus
            gpuis_from_x = job.gpusassigned_set
            # gpuis_from_x should be equal to the job.assignedset()

            assert len(gpuis_from_x) == job.g
            assert job.g >= 1

            if job.g == 1:
                job.singlejoblimbw = np.Inf
                job.multijoblimbw = np.Inf
                job.scale = 0
            else:
                job.singlejoblimbw = np.Inf
                job.multijoblimbw = np.Inf
                for edge in self.g2j.dictofedgesets[jid]:
                    speed = g.edges[edge]['speed'] / 1.0
                    if speed < job.singlejoblimbw:
                        job.singlejoblimbw = speed
                    multispeed = speed * 1.0 / g.edges[edge]['jobcount']
                    if multispeed < job.multijoblimbw:
                        job.multijoblimbw = multispeed
                job.scale = job.singlejoblimbw / job.multijoblimbw
            assert ~np.isnan(job.singlejoblimbw)
            assert ~np.isnan(job.multijoblimbw)
            if np.isnan(job.scale):
                debuginfo('j.scale is nan, env.step: %d' % self.curr_time)
                assert ~np.isnan(job.scale)

    # class Env
    def reset(self):
        """
        Environment should be reset before an episode is run.

        :return:
        :rtype:
        """
        self.seq_idx = 0
        self.curr_time = 0

        # temporary place to hold cumulative negslowdown for each time step.
        self.negslowdown = 0.0  # May need to refractor and put somehwere else.
        self.removedjobscount = 0
        self.extrapenalties = 0.0

        # initialize system
        self.cluster = Cluster(self.pa)  # Needed, but your Cluster class
        # might be different.
        self.job_backlog = JobBacklog(self.pa)  # Usefull. Try to use it.
        self.job_record = JobRecord()  # Keep. Used for slowdown. Not
        # mandatory for first game env prototype.
        self.extra_info = ExtraInfo(self.pa)  # Used for extra info input
        # into the neural network. Not mandatory for first game env prototype.
        self.g2j = G2J(self.pa)  # Mandatory
        self.job_slots = self.g2j.job_slots  # Mandatory
        # self.job_slots.slots.clear()  # not needed bc G2J is newly constructed
        self.snc.reset_g1(jobcount=0)  # Mandatory

        return self.observe()


class EnvStaticJobPick(Env):
    """

    """

    def runningjobs_list_abstmeth(self):
        return self.runningjobs_static()

    def writemore_debug_dict_child(self):
        self.pa.debug_dict['running'] = self.g2j.jobsrunning_static()

    def make_debug_str_child(self):
        return self.make_debug_str_one_action()

    def __init__(self, pa, snc, nw_len_seqs=None, nw_size_seqs=None,
                 seed=42, render=False, repre='image',
                 end='no_more_jobs_in_seq'):
        super(EnvStaticJobPick, self).__init__(pa, snc, nw_len_seqs,
                                               nw_size_seqs, seed,
                                               render, repre, end)

    def step(self, action, debugstr=''):
        """
        Static one action. NN Jobpick, simple heuristic allocation.

        :param action:
        :type action:
        :param repeat:
        :type repeat:
        :param debugstr:
        :type debugstr:
        :return:
        :rtype:
        """
        return self.step_static_oneaction(action, debugstr='')


class EnvFullPreempt(Env):
    """

    """
    '''
    Rewrite
    '''
    def runningjobs_list_abstmeth(self):
        """

        :return: list of jobs
        :rtype: list
        """
        return self.runningjobs_fullpreemption()

    def writemore_debug_dict_child(self):
        self.pa.debug_dict['running'] = self.g2j.jobsrunning_fullpre()
        self.pa.debug_dict['action'] = self.g2j.sum_x_over_gpus()
        # self.pa.debug_dict['running']

    def make_debug_str_child(self):
        return self.make_debug_str_manyactions()

    def __init__(self, pa, snc, nw_len_seqs=None, nw_size_seqs=None,
                 seed=42, render=False, repre='image',
                 end='no_more_jobs_in_seq'):
        super(EnvFullPreempt, self).__init__(
            pa, snc, nw_len_seqs, nw_size_seqs, seed, render, repre, end)

    def step(self, action, debugstr=''):
        """
        Full Pre-emption step

        :param action:
        :type action:
        :param repeat:
        :type repeat:
        :param debugstr:
        :type debugstr:
        :return:
        :rtype:
        """
        return self.step_full_preempt(action, debugstr)


class EnvFullPreemptLstm1(EnvFullPreempt):
    def __init__(self, pa, snc, nw_len_seqs=None, nw_size_seqs=None,
                 seed=42, render=False, repre='image',
                 end='no_more_jobs_in_seq'):
        super(EnvFullPreempt, self).__init__(
            pa, snc, nw_len_seqs, nw_size_seqs, seed, render, repre, end)
        assert pa.modelname == 'lstm1'

    def observe(self):
        return self.observelstm1()


class EnvFullPreemptMLP(EnvFullPreempt):
    def __init__(self, pa, snc, nw_len_seqs=None, nw_size_seqs=None,
                 seed=42, render=False, repre='image',
                 end='no_more_jobs_in_seq'):
        super(EnvFullPreempt, self).__init__(
            pa, snc, nw_len_seqs, nw_size_seqs, seed, render, repre, end)
        assert pa.modelname == 'mlp'


class ExtraInfo(object):
    """
    Class to hold information related to time since last job.

    This is the last column of the input table.
    """

    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        """
        ExtraInfo.time_since_last_new_job is a counter that counts the steps
        since the last new job arrival (not allocation). It goes up to and
        including a maximum = ExtraInfo.max_tracking_time_since_last_job.
        :return:
        """
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1


class SpeedTests(object):
    slotsstr = ['a', 'b', 'c', 'd', 'e']
    slotsint = np.arange(len(slotsstr))

    def __init__(self):
        pass

    @classmethod
    def option1_0(cls):
        """
        Given a numpy array of running jobs' ids, we return a collection of job
        objects.

        Both job ids and the returned collection of jobs is a numpy array.
        There is no intermediate datastructure, because we can slice a numpy
        array with another numpy array.

        This is fastest
        :return:
        :rtype:
        """
        slotsarray = np.empty(len(cls.slotsstr), dtype=object)
        for i in range(5):
            slotsarray[i] = Job((10, 10), 10, 10, i, 100, 100, 100)

        start = time.clock()
        for i in range(100):
            slotsarray[cls.slotsint]
        end = time.clock()
        print("option1_0 is {0:1.4e} seconds ".format((end - start) / 1000))

    @classmethod
    def option1_1(cls):
        """
        Given a numpy array of running jobs' ids, we return a list of job
        objects.

        The intermediate data structure, slotsstr is a python list.

        This is slowest by just a little bit.
        :return:
        :rtype:
        """
        chosen = np.arange(len(cls.slotsstr))
        start = time.clock()
        for i in range(100):
            [cls.slotsstr[s] for s in chosen]
        end = time.clock()
        print("option1_1 is {0:1.4e} seconds ".format((end - start) / 1000))

    @classmethod
    def option1_2(cls):
        """
        Given a numpy array of running jobs' ids, we return a list of job
        objects.

        The intermediate data structure, startednotdone_jobs is a dict that maps
        from job id's to jobs.

        This is second fastest
        :return:
        :rtype:
        """

        cls.jobids = np.random.randint(4, size=4)
        cls.started_jobs = zip(cls.slotsint, cls.slotsstr)
        start = time.clock()
        for i in range(100):
            [cls.started_jobs[jobid] for jobid in cls.jobids]
        end = time.clock()
        print("option1_2 is {0:1.4e} seconds ".format((end - start) / 1000))

    @classmethod
    def option2_0(cls):
        """
        Set a selected group of edges in a networkX graph. Define the graph
        first.
        :return: G, edges
        :rtype:
        """
        g = nx.Graph()
        g.add_path([1, 2, 3, 4])
        nx.set_edge_attributes(g, 0, 'jobcount')
        edgeaccessor = 1
        if edgeaccessor == 1:
            edges = [(1, 2), (2, 3), (3, 4)]
        elif edgeaccessor == 2:
            # also try nx.edges(G, nbunch=None)
            edges = nx.edges(g)
        return g, edges

    @classmethod
    def option2_1(cls):
        """
        This option is faster than option2_2 by a factor of ~ 40x.
        :return:
        :rtype:
        """
        g, edges = cls.option2_0()

        # Time below. We don't need to get edges as a dict with
        # nx.get_edge_attributes because we are accessing the dict edge in
        # place using subscript notation.
        start = time.clock()
        for i in range(1000):
            for edge in edges:
                g[edge[0]][edge[1]]['jobcount'] += 1
        end = time.clock()
        print("option2_1 is {0:1.4e} seconds ".format((end - start) / 1000))

    @classmethod
    def option2_2(cls):
        g, edges = cls.option2_0()

        # Time below. We need to get edges as a dict with
        # nx.get_edge_attributes so we can modifiy each edges' jobcount
        # entry. We do this for the set of edges return first, then use the
        # modified edges dict as input to nx.set_edge_attributes
        start = time.clock()
        for i in range(1000):
            edges = nx.get_edge_attributes(g, 'jobcount')
            for edge in edges:
                edges[edge] += 1  # edges[edge]= 0
            nx.set_edge_attributes(g, edges, 'jobcount')
        end = time.clock()
        print("option2_2 is {0:1.4e} seconds ".format((end - start) / 1000))

    @classmethod
    def option2_3(cls):
        """
        Test difference in mutability between subgraph and edge_subgraph

        Result: They both return the same kind of graph. Structurally frozen
        (ie. 'frozen' attribute is 'True') but their edges are shared with
        the original graph and changes to edge attributes in the subgraph are
        viewable in the original graph.

        :return:
        :rtype:
        """

        g, edges = cls.option2_0()

        # Time below. First get the readonly subgraph view from edges.
        # not sure how to proceed with read only view.
        # g.subgraph returns subgraph view. Structure is read only,
        # but node/edge attribtues are shared with original graph and can be
        # updated. Investigate how g.subgraph and g.edge_subgraph are
        # different wrt read/settability.

        subgraphtype = 'subgraph'
        if subgraphtype == 'subgraph':
            # updates to this subgraph's edges are reflected in g's edges
            subgraph = g.subgraph(g.nodes)

        elif subgraphtype == 'edge':
            # updates to this subgraph's edges are also reflected in g's edges
            subgraph = g.edge_subgraph(edges)

        for edge in subgraph.edges:
            subgraph.edges[edge]['jobcount'] += 1  #

        # Result. Both edge_subgraph and subgraph return structurally frozen
        # (i.e. their 'frozen' attribute is 'True') graphs but their node and
        #  edge attributes are shared with the original graph and edits to
        # them are reflected in the original graph.
        pass


# ==========================================================================
# ------------------------------- Unit Tests -------------------------------
# ==========================================================================


def test_backlog():
    pa = parameters.Parameters()
    pa.num_jobslots = 5
    pa.t_finaljobarrival = 50
    pa.num_jobsets = 10
    pa.new_job_rate = 1
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='image')

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    env.step(5)
    assert env.job_backlog.backlog[0] is not None
    assert env.job_backlog.backlog[1] is None
    print("New job is backlogged.")

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slots.slots[0] == job

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slots.slots[0] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slots.slots[1] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slots.slots[1] == job

    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(3)
    assert env.job_slots.slots[3] == job

    print("- Backlog test passed -")


def test_compact_speed():
    pa = parameters.Parameters()
    pa.t_finaljobarrival = 50
    pa.num_jobsets = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='compact')

    import other_agents
    import time

    start_time = time.time()
    for i in range(100000):
        a = other_agents.get_sjf_action(env.cluster, env.job_slots)
        env.step(a)
    end_time = time.time()
    print("- Elapsed time: ", end_time - start_time, "sec -")


def test_image_speed():
    pa = parameters.Parameters()
    pa.t_finaljobarrival = 50
    pa.num_jobsets = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='image')

    import other_agents
    import time

    start_time = time.time()
    for i in range(100000):
        a = other_agents.get_sjf_action(env.cluster, env.job_slots)
        env.step(a)
    end_time = time.time()
    print("- Elapsed time: ", end_time - start_time, "sec -")


if __name__ == '__main__':
    pass
    # V-cbb. three lines of testing by Mao.
    # test_backlog()
    # test_compact_speed()
    # test_image_speed()

    # V-cbb. test read_ret_sym
    # test_ret_sym()

    # V-cbb. Speed test different ways of setting subset of NetworkX edge
    # attributes. SpeedTests work and analyses are finished.
    # SpeedTests.option2_1()
    # SpeedTests.option2_2()
    # SpeedTests.option2_3()
