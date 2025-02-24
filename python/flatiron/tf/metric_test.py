import unittest

import flatiron.tf.metric as fi_tfmetric
# ------------------------------------------------------------------------------


class TFMetricTests(unittest.TestCase):
    def test_get(self):
        fi_tfmetric.get(dict(name='intersection_over_union'))
        fi_tfmetric.get(dict(name='jaccard'))
        fi_tfmetric.get(dict(name='dice'))
