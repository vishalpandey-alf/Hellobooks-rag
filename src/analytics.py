import time

class Analytics:

    def __init__(self):
        self.total_queries = 0
        self.total_latency = 0

    def record(self, latency):
        self.total_queries += 1
        self.total_latency += latency

    def avg_latency(self):
        if self.total_queries == 0:
            return 0
        return self.total_latency / self.total_queries