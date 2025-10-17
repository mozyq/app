from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timer(label=""):
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"{label} took {(end - start)*1000:.2f} ms")
