__author__ = "Richard Michael"
"""rfp_fam dataset."""

import os
from pathlib import Path
import numpy as np
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for rfp_fam dataset."""

  VERSION = tfds.core.Version('0.0.1')
  RELEASE_NOTES = {
      '0.0.1': 'Preliminary alpha release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(rfp_fam): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'sequence': tfds.features.Text(),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage=None,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(rfp_fam): Downloads the data and defines the splits
    #path = dl_manager.download_and_extract('https://todo-data-url')
    path = "/Users/rcml/home/corel/experiments/assets/data/rfp_fam/"

    # Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(Path(os.path.join(path, 'train_seqs'))),
        'test': self._generate_examples(Path(os.path.join(path,'test_seqs')))
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(rfp_fam): Yields (key, example) tuples from the dataset
    for f in path.glob('*.fasta'):
        seq = "".join(np.loadtxt(f, dtype=str)[1:])
        yield 'key', {
            'sequence': seq,
        }
