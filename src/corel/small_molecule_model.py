from typing import Callable

import tensorflow as tf
import tensorflow_probability as tfp
from trieste.data import Dataset

from trieste.models import TrainableProbabilisticModel

from corel.protein_model import ProteinModel

tfd = tfp.distributions


class SmallMoleculeModel(ProteinModel):
    def __init__(
        self,
        distribution: Callable[[tf.Tensor], tf.Tensor],
        alphabet_length: int,
    ) -> None:
        super().__init__(distribution, AA=alphabet_length)
