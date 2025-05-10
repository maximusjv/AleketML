import functools
import time



class Profiler:
    profilers = {}
    
    def __init__(self, name: str):
        self.runs = []
        self.name = name
        self.start_time = 0
        
    def start_run(self):
        if self.start_time != 0:
            raise Exception("Run already started")
        self.start_time = time.time()
    
    def end_time(self):
        if self.start_time == 0:
            raise Exception("Run not started")
        
        end_time = time.time()
        self.runs.append(end_time - self.start_time)
        self.start_time = 0
        
    @property
    def total_time(self):
        return sum(self.runs)

    @property
    def total_runs(self):
        return len(self.runs)
    
    @property
    def average_time(self):
        return self.total_time / self.total_runs if self.total_runs != 0 else 0
    
    def __str__(self):
        return f"{self.name}: {self.average_time}ms"
    def __repr__(self):
        return self.__str__()
 
    @staticmethod
    def profile(func):
        profiler = Profiler(func.__name__) 
        Profiler.profilers[f"{func.__name__}_{hash(func)}"] = profiler
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler.start_run()
            result = func(*args, **kwargs)
            profiler.end_time()
            return result
        
        return wrapper
            
        
        