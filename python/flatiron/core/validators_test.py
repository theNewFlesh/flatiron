import unittest

import flatiron.core.validators as vd
# ------------------------------------------------------------------------------


class ValidatorsTests(unittest.TestCase):
    def test_is_even(self):
        self.assertEqual(vd.is_even(10), 10)

        expected = '3 is not an even number.'
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_even(3)

    def test_is_odd(self):
        self.assertEqual(vd.is_odd(11), 11)

        expected = '2 is not an odd number.'
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_odd(2)

    def test_is_base_two(self):
        self.assertEqual(vd.is_base_two(128), 128)

        expected = '11 is not a base two number.'
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_base_two(11)

    def test_is_padding(self):
        self.assertEqual(vd.is_padding('same'), 'same')
        self.assertEqual(vd.is_padding('valid'), 'valid')

        expected = 'foobar is not a legal padding type. '
        expected += 'Legal types:.*valid.*same'
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_padding('foobar')

    def test_is_callback_mode(self):
        self.assertEqual(vd.is_callback_mode('auto'), 'auto')
        self.assertEqual(vd.is_callback_mode('min'), 'min')
        self.assertEqual(vd.is_callback_mode('max'), 'max')

        expected = 'foobar is not a legal callback mode. '
        expected += 'Legal types:.*auto.*min.*max'
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_callback_mode('foobar')

    def test_is_pipeline_method(self):
        legal = [
            'load',
            'train_test_split',
            'unload',
            'build',
            'compile',
            'train',
        ]
        for method in legal:
            self.assertEqual(vd.is_pipeline_method(method), method)

        expected = 'foobar is not a legal pipeline method. '
        expected += 'Legal methods:.*' + '.*'.join(legal)
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_pipeline_method('foobar')

    def test_is_engine(self):
        legal = ['tensorflow', 'torch']
        for method in legal:
            self.assertEqual(vd.is_engine(method), method)

        expected = 'foobar is not a legal deep learning framework. '
        expected += 'Legal engines:.*' + '.*'.join(legal)
        with self.assertRaisesRegex(ValueError, expected):
            vd.is_engine('foobar')
