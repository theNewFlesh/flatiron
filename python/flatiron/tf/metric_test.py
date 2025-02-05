import unittest

import flatiron.tf.metric as ftfm
# ------------------------------------------------------------------------------


class TFMetricTests(unittest.TestCase):
    def test_get(self):
        ftfm.get('intersection_over_union')
        ftfm.get('jaccard')
        ftfm.get('dice')
