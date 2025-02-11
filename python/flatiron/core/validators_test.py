import unittest

import flatiron.core.validators as vd
# ------------------------------------------------------------------------------


class ValidatorsTests(unittest.TestCase):
    def test_is_gte(self):
        vd.is_gte(10, 1)

        expected = '5 is not greater than or equal to 10.'
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_gte(5, 10)

    def test_is_even(self):
        vd.is_even(10)

        expected = '3 is not an even number.'
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_even(3)

    def test_is_odd(self):
        vd.is_odd(11)

        expected = '2 is not an odd number.'
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_odd(2)

    def test_is_base_two(self):
        vd.is_base_two(128)

        expected = '11 is not a base two number.'
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_base_two(11)

    def test_is_padding(self):
        vd.is_padding('same')
        vd.is_padding('valid')

        expected = 'foobar is not a legal padding type. '
        expected += 'Legal types:.*valid.*same'
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_padding('foobar')

    def test_is_callback_mode(self):
        vd.is_callback_mode('auto')
        vd.is_callback_mode('min')
        vd.is_callback_mode('max')

        expected = 'foobar is not a legal callback mode. '
        expected += 'Legal types:.*auto.*min.*max'
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_callback_mode('foobar')

    def test_is_pipeline_methods(self):
        legal = [
            'load',
            'train_test_split',
            'unload',
            'build',
            'compile',
            'fit',
        ]
        vd.is_pipeline_methods(legal)

        expected = r'\[foo, bar\] are not legal pipeline methods. '
        expected += 'Legal methods:.*' + '.*'.join(legal)
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_pipeline_methods(['load', 'foo', 'bar'])

    def test_is_engine(self):
        legal = ['tensorflow', 'pytorch']
        for method in legal:
            vd.is_engine(method)

        expected = 'foobar is not a legal deep learning framework. '
        expected += 'Legal engines:.*' + '.*'.join(legal)
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_engine('foobar')
