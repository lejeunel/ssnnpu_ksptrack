from bisect import bisect_right, bisect_left
from collections import Counter


class TrainCycleScheduler:
    """
    """

    def __init__(self,
                 periods,
                 max_epochs,
                 cycle_names):
        assert(len(periods) == len(cycle_names)), print('periods should be same length as cycles')
        self.max_epochs = max_epochs
        self.periods = periods
        self.cycle_names = cycle_names
        self.milestones = [self.periods[0]]

        last_ms_i = 0
        next_period_i = 1
        for i in range(self.max_epochs):
            if(i == self.milestones[last_ms_i] + self.periods[next_period_i % len(periods)]):
                next_period_i += 1
                self.milestones.append(i)
                last_ms_i += 1
                
        self.epochs_per_cycle = {k: 0 for k in self.cycle_names}

        self.curr_epoch = 0

    def get_cycle(self):
        # return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
        #         for base_lr in self.base_lrs]
        idx = min((bisect_right(self.milestones, self.curr_epoch), len(self.milestones)))
        return self.cycle_names[idx % len(self.cycle_names)]

    def step(self):
        self.curr_epoch += 1
        self.epochs_per_cycle[self.get_cycle()] += 1



if __name__ == "__main__":

    sch = TrainCycleScheduler([10, 20], 100, ['feats', 'siam'],
                              [0.1, 0.01], [.9, .9])

    for ep in range(100):
        print('ep: {}, cycle: {}'.format(ep, sch.get_cycle()))
        sch.step()
