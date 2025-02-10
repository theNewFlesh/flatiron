import unittest

import flatiron.tf.metric as f_tfmetric
# ------------------------------------------------------------------------------


class TFMetricTests(unittest.TestCase):
    def test_get(self):
        f_tfmetric.get('intersection_over_union')
        f_tfmetric.get('jaccard')
        f_tfmetric.get('dice')
