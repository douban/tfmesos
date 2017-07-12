from contextlib import contextmanager
from tfmesos.scheduler import Job, TFMesosScheduler

__VERSION__ = '0.0.5'


@contextmanager
def cluster(jobs, **kw):
    if isinstance(jobs, dict):
        jobs = [Job(**jobs)]

    if isinstance(jobs, Job):
        jobs = [jobs]

    jobs = [job if isinstance(job, Job) else Job(**job)
            for job in jobs]
    s = TFMesosScheduler(jobs, **kw)
    try:
        s.start()
        yield s
    finally:
        s.stop()
