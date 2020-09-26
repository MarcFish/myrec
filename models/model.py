import abc


class Model(abc.ABC):
    @abc.abstractmethod
    def train(self):
        return NotImplementedError

    @abc.abstractmethod
    def make_rec(self, u, item_cand):
        return NotImplementedError
