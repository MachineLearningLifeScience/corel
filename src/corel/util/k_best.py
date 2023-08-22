__author__ = 'Simon Bartels'

from typing import Callable

import numpy as np


class KeepKBest:
    def __init__(self, k: int, copy: Callable = lambda x: x):
        """
        Keeps track of k-best values.
        :param k:
            the number of values to keep track of
        :param copy:
            a function that makes a copy of an element if necessary
        """
        self.k = k
        self.copy = copy
        self.vals = -np.infty * np.ones(k)
        self.elements = np.array(k * [object()])

    def new_val(self, val, element) -> int:
        if val > self.vals[-1]:
            idx = np.argmax(val > self.vals)
            #self.vals = np.stack([self.vals[:idx], [val], self.vals[idx:-1]])
            #self.elements = np.stack([self.elements[:idx], [element], self.elements[idx:-1]])
            # Careful when switching the underlying framework, the code below may have side effects!
            self.vals[idx+1:] = self.vals[idx:-1]
            self.vals[idx] = val
            self.elements[idx+1:] = self.elements[idx:-1]
            self.elements[idx] = self.copy(element)
            return idx
        return -1

    def get(self):
        assert(self.vals.shape[0] == self.k)
        assert(self.elements.shape[0] == self.k)
        return self.elements, self.vals
