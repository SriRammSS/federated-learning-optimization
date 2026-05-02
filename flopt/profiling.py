
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timed(stage:str,rows:list[dict]):
    start=perf_counter()
    yield
    rows.append({"stage":stage,"seconds":perf_counter()-start})
