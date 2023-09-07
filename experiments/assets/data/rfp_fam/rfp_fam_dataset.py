import os
from pathlib import Path

import numpy as np
import tensorflow as tf


train_data_path = Path(os.path.join(Path(__file__).parent.resolve(), 'train_seqs'))

test_data_path = Path(os.path.join(Path(__file__).parent.resolve(), 'test_seqs'))


train_files = [f_name for f_name in train_data_path.glob('*.fasta')]
test_files = [f_name for f_name in test_data_path.glob('*.fasta')]


rfp_train_dataset = tf.data.TextLineDataset(train_files)

rfp_test_dataset = tf.data.TextLineDataset(test_files)