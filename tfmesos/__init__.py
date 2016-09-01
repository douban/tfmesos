from contextlib import contextmanager
from tfmesos.scheduler import Job, TFMesosScheduler

__VERSION__ = '0.0.1'


@contextmanager
def cluster(jobs, master=None, name=None, quiet=False,
            volumes={}, local_task=None):
    if isinstance(jobs, dict):
        jobs = [Job(**jobs)]

    if isinstance(jobs, Job):
        jobs = [jobs]

    jobs = [job if isinstance(job, Job) else Job(**job)
            for job in jobs]
    s = TFMesosScheduler(jobs, master=master, name=name, quiet=quiet,
                         volumes=volumes, local_task=local_task)
    yield s.start()
    s.stop()
