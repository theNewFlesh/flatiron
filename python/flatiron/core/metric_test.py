import unittest

import flatiron.core.metric as ficm
# ------------------------------------------------------------------------------


class MetricTests(unittest.TestCase):
    def test_get(self):
        ficm.get('intersection_over_union')
        ficm.get('jaccard')
        ficm.get('dice')
