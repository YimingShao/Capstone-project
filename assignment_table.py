from collections import OrderedDict

import numpy as np

from jobs_queue_rltaps import JobSlots


class G2J(object):
    """
    Class to hold and handle the assignment of gpus to jobs.

    For now this class will only be valid for whole gpu assignments, i.e. a
    single gpu can be assigned to only one job.

    Consider what other fields from cluster class we might want to bring into this class, and what to call the resulting class.

    """

    def __init__(self, pa, jobqueue):
        """

        :param pa: parameters object
        :type pa:
        :param job_slots: job_slot object passed in from env object's scope.
        This is needed by the jobslotvec property

        :type job_slots:
        """
        # cbb. consider inputting the values of pa's fields, instead of the
        # pa object. This is helpful for future proofing the field names of
        # pa object. We need the following varaibles from pa instance:
        # - pa.num_clustergpus, pa.num_jobslots
        self.pa = pa
        # use 'x' as the assignment table, like how we set X as a matrix of
        # decisions variables. But use lower case since it's a python variable.
        self._x = np.full((self.pa.num_clustergpus, self.pa.num_jobslots + 1),
                          False, dtype=bool)
        self._x[:, -1] = True  # initialize all GPUs to be assigned to null slot
        self._xo = self._x  # maybe try to hold the previous step's version
        # of _x, not used currently

        # self.job_slots also points to env.job_slots
        self.job_slots = JobSlots(jobqueue.slot)
        self.job_queue = jobqueue

        self.startednotdone_jobs = OrderedDict()  # cbb. use this new version,
        # which is a dict that maps from job id to job object.

        # Started_jobs is a super set of running_jobs.
        # In static case and partial pre-emption case when all running jobs
        # are never paused and always running on the cluster,
        # startednotdone_jobs is equal to runningjobs
        # cluster, startednotdone_jobs is equal to
        # In full pre-emption case when jobs can be paused,
        # startednotdone_jobs is the SET summation of running_jobs and
        # paused_jobs. Consider making a paused_jobs dict or set.
        # Note: Mao removes from 'startednotdone_jobs' any finished jobs.
        self.running_jobs = {}

        # Any job that enters the slots will be put in here. And jobs that
        # is removed from slots is removed from here.
        self.jobi_toaddgpusto = None
        self.allgpusset = frozenset(range(self.pa.num_clustergpus))
        self.freegpus_set = set(self.allgpusset)
        self.assignedgpus_set = set()
        self.gpusrun_byslot = None

        # # keep track of number of gpus requested per slot. 0 if no job.
        # self.ngpusrequested = np.zeros(self.pa.num_new_m, dtype=int)

        # associate the gpu set for a job with a JobSlot object, instead of a
        #  Job object, because there are only a set number of slots (much
        # lower) that the number of possible jobs.
        # cbb. instead of using a ndarray to hold sets, use a dict. the key
        # can be job slot or job.id (for static case where jobs are moved off
        #  of slots) and values are sets.
        self.dictofedgesets = {}
        self.premakegpusets()

        # the job slot vector to show which job slots have/don't have jobs.
        # If a job slot has a job, element is True, o/w element is False.
        self.jobslotvec = np.full(self.job_slots.slots.maxlen, False,
                                  dtype=np.bool)

        # self._x related fields. needs updating after change to _x
        # self.chosen_slots_bool = None
        self.chosen_slots_int = None

        # fields that are only related to self.job_slots that needs updating
        # after change to job_slots are updated in job_slots class.

        # self._x and self.job_slots related fields. needs updating after
        # change to _x or job_slots
        self.dismissedjobs_slots_bool = None
        self.dismissedjobs_slots_int = None
        self.chosenjobs_slots_bool = None
        self.chosenjobs_slots_int = None

        # updatefields_after_x_update() needs to be called first because
        # update_fields_after_slot_update() assumes self._x related fields are
        # well formed np arrays.
        self.updatefields_after_x_update()
        self.updatefields_after_slot_update()
        self.updatefields_after_x_slots_update()

    @property
    def chosen_slots_bool(self):
        return np.any(self.x[:, :-1], axis=0)

    def sum_x_over_gpus(self):
        """
        This is like seeing how many gpus were assigned to each slot, even,
        the null slot.
        :return:
        :rtype:
        """
        return np.sum(self._x, axis=0)

    def runningjobs_fullpre(self):
        """
        Returns a list of running jobs for the full pre-emption case.

        Should we return instead a dict of job id to job mappings? How do we
        maintain this dict? This dict ideally should not be cleared at the
        beginnin of a step (inside the step) function. i.e. it would be
        useful to know what jobs were running in the previous step. But if
        need be, for simplicity, we can decide to clear it, and repopulate
        during every step (but try not to). Perhaps instead it should be a set.
        :return:
        :rtype:
        """

    def runningjobsslots_fullpre(self):
        """
        Returns a list of slots with running jobs for the full pre-emption
        case. This function has no counterpart for satic case because in
        static case, we always remove jobs from the slots.
        :return:
        :rtype:
        """

    def jobsrunning_fullpre(self):
        """
        For full pre-emption, we count the number of gpu's being used by
        currently running jobs (i.e. those with gpus assigned by actions).
        We then return a list of tuples, one tuple per job running.
        Tuple takes the form (job.len, job.ts_togo). Feel free to change The
        attributes included in the tuple as needed. The purpose of this
        function is for displaying debugging output.
        :return:
        :rtype:
        """
        jobsassignedlist = [0] * self.pa.num_jobslots
        for s in self.chosenjobs_slots_int:
            job = self.sloti2job(s)
            jobsassignedlist[s] = (job.len, job.g)
        return jobsassignedlist

    def jobsrunning_static(self):
        """
        For static cases, we count the number of gpu's being used by
        currently running jobs (i.e. those with gpus assigned by actions).
        We then return a list of tuples, one tuple per job running.
        Tuple takes the form (job.len, job.ts_togo). Feel free to change The
        attributes included in the tuple as needed. The purpose of this
        function is for displaying debugging output.
        :return:
        :rtype:
        """
        jobsassignedlist = []
        for jid, job in self.startednotdone_jobs.items():
            jobsassignedlist.append((job.len, job.ts_togo, job.g))
        return jobsassignedlist

    def ngpusrun_fullpre(self):
        """
        For full pre-emption, we count the number of gpu's being used by
        currently running jobs (i.e. those with gpus assigned by actions).
        :return:
        :rtype:
        """
        ngpusassignedlist = [0] * self.pa.num_jobslots
        for s in self.chosenjobs_slots_int:
            ngpusassignedlist[s] = self.numgpusassigned(s)
        return ngpusassignedlist

    def ngpusrun_staticppre(self):
        """
        For static and partial preemption case, we look at
        startednotdone_jobs variable which is logically equivalent to a
        runningjobs variable if we had a working one. We list the number of
        gpus being used by currently running jobs
        :return:
        :rtype:
        """
        ngpusbeingused = []
        for jid, job in self.startednotdone_jobs.items():
            ngpusbeingused.append(job.g)
        return ngpusbeingused

    def ngpusrequested(self):
        """
        Returns the number of GPUs requested by jobs in the slots.
        Works for multi allocate as well
        :return:
        :rtype:
        """
        numgpusrequestedperslot = np.zeros(self.pa.num_jobslots, dtype=int)
        for i in range(self.pa.num_jobslots):
            if self.job_slots.slots[i] is not None:
                # Must use job.res_vec[0] instead of job.g because job.g gets
                #  overwritten with number of gpu's assigned.
                numgpusrequestedperslot[i] = self.job_slots.slots[i].res_vec[0]
        return numgpusrequestedperslot

    def jobsinslots_fordebug_fullpre(self):
        """
        Like ngpusrequested, but gives more info
        :return:
        :rtype:
        """
        jobsinslots = [0] * self.pa.num_jobslots
        for i in range(len(self.job_slots.slots)):
            job = self.job_slots.slots[i]
            if job is not None:
                # Must use job.res_vec[0] instead of job.g because job.g gets
                #  overwritten with number of gpu's assigned.
                jobsinslots[i] = (job.len, job.res_vec[0], job.g)
        return jobsinslots

    def jobsinslots_fordebug_static(self):
        """
        Like ngpusrequested, but gives more info
        :return:
        :rtype:
        """
        jobsinslots = [0] * self.pa.num_jobslots
        for i in range(len(self.job_slots.slots)):
            job = self.job_slots.slots[i]
            if job is not None:
                # Must use job.res_vec[0] instead of job.g because job.g gets
                #  overwritten with number of gpu's assigned.
                jobsinslots[i] = (job.id, job.len)
        return jobsinslots

    def gpusassigned_pg(self):
        """
        Returns arrays of gpus assigned to each slot by the rl method
        :param sloti:
        :type sloti:
        :return:
        :rtype:
        """
        temp = [None] * self.pa.num_jobslots
        for s in range(self.pa.num_jobslots):
            gpuslist = list(self.jobsloti2gpus(s))
            temp[s] = sorted(gpuslist) if gpuslist else gpuslist
        return temp

    def gpusassigned_jobs_actual(self):
        temp = []
        for s in self.chosenjobs_slots_int:
            temp.append(sorted(list(self.sloti2job(s).gpusassigned_set)))
        return temp

    def gpusassigned_jobs_pg(self):
        """
        Returns arrays of gpus assigned if a job exists
        :return:
        :rtype:
        """
        temp = []
        for s in self.chosenjobs_slots_int:
            temp.append(list(self.jobsloti2gpus(s)))
        return temp

    def sloti2job(self, sloti):
        """

        :param sloti: job slot number
        :type sloti: int
        :return: job
        :rtype: Job
        """
        return self.job_queue.find_job(sloti)

    def premakegpusets(self):
        for i in range(len(self.dictofedgesets)):
            self.dictofedgesets[i] = set()

    def assigntoslots(self, sloti=None, job=None):
        """

        :param sloti:
        :type sloti: int
        :param job:
        :type job: Job
        :return: None
        :rtype: None
        """
        if sloti is not None:
            self.job_slots.slots.insert(sloti, job)
        else:
            assert job is not None
            self.job_slots.slots.append(job)

    def assigntoslots_updatefields_fullpreempt(self, sloti=None, job=None):
        """

        :param sloti:
        :type sloti:
        :param job:
        :type job:
        :return:
        :rtype:
        """
        self.assigntoslots(sloti=sloti, job=job)
        self.updatefields_after_slot_update()  # updates jobslot_vec
        self.updatefields_after_x_slots_update()

    def assigntoslots_updatefields_static(self, sloti=None, job=None):
        """

        :param job:
        :type job:
        :return:
        :rtype:
        """
        self.assigntoslots(sloti=sloti, job=job)
        self.job_slots.slots.append(job)
        self.updatefields_after_slot_update()  # updates jobslot_vec

    def updatefields_after_x_update(self):
        """
        Update fields that depend only on _x, after updating _x.

        :return:
        :rtype:
        """
        # this is the only place where we need to compute the
        # following fields.
        # self.chosen_slots_bool = np.any(self.x[:, :-1], axis=0)
        self.chosen_slots_int = np.flatnonzero(self.chosen_slots_bool)

    def updatefields_after_slot_update(self):
        """
        Update fields that depend only on job_slots, after job_slots is updated.

        :return:
        :rtype:
        """
        # self.jobslotvec = self.job_slots.slots != None
        # v-cbb. Since we are using a queue, we just set the indices
        # corresponding to the queue spots occupied by jobs to True.
        self.jobslotvec[:len(self.job_slots.slots)+1] = True
        self.jobslotvec[len(self.job_slots.slots):] = False

    def updatefields_after_x_slots_update(self):
        """
        Updates fields dependent on both _x and _slots.

        These fields are

        :dismissedjobs_slots_bool: Bool vector representation of jobslotvec
            where elements are True if job exists and job was not chosen
        :dismissedjobs_slots_int: Indices of dismissed jobs that exist in
            job_slots
        :chosenjobs_slots_bool: Bool vector representation of jobslotvec
            where elements are True if job exists and it's chosen
        :chosenjobs_slots_int: Indices of chosen jobs in jobslotvec

        :return: None
        """
        self.dismissedjobs_slots_bool = np.logical_and(
            ~self.chosen_slots_bool, self.jobslotvec)

        self.dismissedjobs_slots_int = np.flatnonzero(
            self.dismissedjobs_slots_bool)

        self.chosenjobs_slots_bool = np.logical_and(self.chosen_slots_bool,
                                                    self.jobslotvec)
        self.chosenjobs_slots_int = np.flatnonzero(self.chosenjobs_slots_bool)

    # class G2J
    def jobsloti2gpus(self, sloti):
        """
        Return vector of gpus assigned to job at sloti. Only returns non False
        indices.
        :param sloti:
        :type sloti:
        :return: A vector of which gpus are assigned to sloti.
        :rtype: np.ndarray
        """
        return np.flatnonzero(self._x[:, sloti])

    # class G2J
    def numgpusassigned(self, sloti):
        """
        Count the number of GPUs assigned to job at sloti

        Done. As long as self.x has been updated, since self.x is a simple
        ndarray matrix, this function will always work.

        :param sloti:
        :type sloti:
        :return: number of GPUs assigned to job at sloti
        :rtype: int
        """
        return len(self.jobsloti2gpus(sloti))

    # Done. class G2J
    @property
    def x(self):
        """
        Simply return the internal representation of the gpu to jobs
        assignment table

        :return: The gpu to jobs assignment table.
        :rtype: np.ndarray
        """
        return self._x

    def x_update_releasegpus(self, gpus):
        """
        Some gpu rows of x should be set to only null jobset being True,
        because the GPU is being released from a job.

        :param gpus:
        :type gpus:
        :return:
        :rtype:
        """
        for g in gpus:
            self._x[g, :] = False  # cbb. first zero/False out the g'th row.
            self._x[g, -1] = True  # set x[g, null job slot] to True

    # Done. class G2J
    def x_update_allgpus(self, actions):
        """
        Uses the actions vector to set x, the gpu to jobs assignment table.

        x must be set at every step. There is no return, we simply set x.

        :param actions: Actions vector from the neural network. Assume this
            is one dimensional array with size=(1, numgpus)
        :type actions: np.ndarray
        """
        # cbb. each of the elements of actions correspond to a gpu. Simplest
        # way to populate x is to iterate through each element.
        for g in range(actions.size):
            # g is dummy index for gpu. goes from 1 .. G
            self._x[g, :] = False  # cbb. first zero/False out the g'th row.

            # V-cbb. bcause of zero indexing, actions[g] == num_jobslots means no
            # job slot is selected for that gpu's softmax. _x has columns
            # for jobslots and rows for gpu's. _x does not have a column
            # for null choice
            # debuginfo(str(actions))
            # if actions[g] != self.pa.num_jobslots:
            self._x[g, actions[g]] = True  # set x[g, j] to True

        """        
        Calculate and return the currently running jobs. This can be done
        algebraically.
        Done for now. Don't use a set becuase algebraic method on numpy array is
        faster
        
        The np.any inside looks at each column of x, to see if the column
        has any True values. Thus a vector is created, whose indices are
        job slots and a True value means the job slot is chosen to be run
        on the cluster/machine. This vector is element wise logically AND
        compared to the vector whose elements are True for job slots with
        jobs.

        """
        self.updatefields_after_x_update()
        self.updatefields_after_x_slots_update()