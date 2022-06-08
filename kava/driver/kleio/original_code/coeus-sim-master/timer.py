from timeit import default_timer as timer
import statistics

class TimerHandle:
    def __init__(self, parent, name):
        self.name = name
        self.parent = parent

    def start(self):
        self.t0 = timer()

    def stop(self):
        elapsed = timer() - self.t0
        self.parent._report_time(self.name, elapsed)

    #context methods taken from https://gist.github.com/sumeet/1123871
    def __enter__(self):
        self.start() 
        return self

    def __exit__(self, type, value, traceback):
        self.stop()


class TimerClass:
    
    def __init__(self):
        self.times = {}

    def get_handle(self, name):
        return TimerHandle(self, name)

    def _report_time(self, name, elapsed):
        if name not in self.times.keys():
            self.times[name] = []
        self.times[name].append(elapsed)

    def print(self):
        #print("Time report\n")
        print("*"*50)
        print("name, avg, stddev")
        for name, times in self.times.items():
            avg    = statistics.mean(times)
            stddev = statistics.pstdev(times)
            print(f"{name}, {avg}, {stddev}")
        print("*"*50)

    def reset(self):
        self.times = {}

Timer = TimerClass()
