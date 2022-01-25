import numpy as np

class Replay_Buffer:
    def __init__(self, capacity = 1024) -> None:
        self.capacity = capacity
        self.count = 0
        self.buffer = [] # list of transition-dicts
        self.minibatch_size = 40

    def store(self, transition : dict) -> bool:

        if self.capacity == self.count:
            raise Exception("Unable to add as replay buffer is at its full capacity({})".format(self.capacity))
        if list(transition.keys()) != ['obs','action','reward','next_obs']:
            raise Exception("Transition should contain 'obs','action','reward','next_obs'.")

        self.buffer.append( transition )


    def random_sample(self, sample_size : int) -> np.array:
        # sample random mini-batch
        return np.random.choice(a = self.buffer, size = sample_size, replace = False)