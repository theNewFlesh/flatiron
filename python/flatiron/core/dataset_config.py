import schematics as scm
import schematics.types as scmt

import flatiron.core.validators as vd
# ------------------------------------------------------------------------------


class DatasetConfig(scm.Model):
    '''
    Configuration for Dataset.

    Attributes:
        source (str): Dataset directory or CSV filepath.
        limit (str or int): Limit data by number of samples or memory size.
            Default: None.
        shuffle (bool): Shuffle chunks before loading. Default: False.
        index (int): Index of axis to split on.
        axis (int): Axis to split data on. Default: -1.
        test_size (float): Test size. Default: 0.2
        train_size (float): Train size. Default: None
        random_state (int): Seed for shuffling randomness.
        shuffle (bool): Shuffle data rows. Default: True.
    '''
    source = scmt.StringType(required=True)
    load_limit = scmt.UnionType(
        (scmt.IntType, scmt.StringType), required=True, serialize_when_none=True
    )
    load_shuffle = scmt.BooleanType(required=True, default=False)
    split_index = scmt.IntType(required=True)
    split_axis = scmt.IntType(required=True, default=-1)
    split_test_size = scmt.FloatType(
        required=True, default=0.2, validators=[lambda x: vd.is_gte(x, 0)]
    )
    split_train_size = scmt.FloatType(
        serialize_when_none=True, validators=[lambda x: vd.is_gte(x, 0)]
    )
    split_random_state = scmt.IntType(required=True, default=42)
    split_shuffle = scmt.BooleanType(required=True, default=True)
