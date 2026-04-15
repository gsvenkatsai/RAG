import time

def measure_time(function, *args, **kwargs):
    start_time = time.time()

    result = function(*args, **kwargs)

    end_time = time.time()

    execution_time = round(end_time - start_time, 3)

    return result, execution_time