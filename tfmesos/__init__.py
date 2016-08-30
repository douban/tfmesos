from contextlib import contextmanager
from tfmesos.scheduler import Job, TFMesosScheduler


@contextmanager
def cluster(jobs, master=None, name=None, quiet=False,
            local_task=None):
    if isinstance(jobs, dict):
        jobs = [Job(**jobs)]

    if isinstance(jobs, Job):
        jobs = [jobs]

    jobs = [job if isinstance(job, Job) else Job(**job)
            for job in jobs]
    s = TFMesosScheduler(jobs, master=master, name=name, quiet=quiet,
                         local_task=local_task)
    yield s.start()
    s.stop()
