class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.n = 0
        self.sum = 0.0
    def update(self, val, k=1):
        self.n += k
        self.sum += float(val) * k
    @property
    def avg(self):
        return self.sum / max(1, self.n)
