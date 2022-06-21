class AverageMeter():

    def __init__(self, name:str) -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.count = 0
        self.total = 0

    def update(self, count: int, total: int) -> None:
        self.count += count 
        self.total += total 
        self.val = self.count/(self.total+1)