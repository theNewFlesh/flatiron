from pathlib import Path
import unittest

from schematics.exceptions import ValidationError

import flatiron.core.validators as vd
# ------------------------------------------------------------------------------


class ValidatorsTests(unittest.TestCase):
    def test_is_gte(self):
        vd.is_gte(10, 1)

        expected = '5 is not greater than or equal to 10.'
        with self.assertRaisesRegex(ValidationError, expected):
            vd.is_gte(5, 10)

    def test_is_even(self):
        vd.is_even(10)

        expected = '3 is not an even number.'
        with self.assertRaisesRegex(ValidationError, expected):
            vd.is_even(3)

    def test_is_odd(self):
        vd.is_odd(11)

        expected = '2 is not an odd number.'
        with self.assertRaisesRegex(ValidationError, expected):
            vd.is_odd(2)

    def test_is_base_two(self):
        vd.is_base_two(128)

        expected = '11 is not a base two number.'
        with self.assertRaisesRegex(ValidationError, expected):
            vd.is_base_two(11)

    def test_is_padding(self):
        vd.is_padding('same')
        vd.is_padding('valid')

        expected = 'foobar is not a legal padding type. '
        expected += 'Legal types:.*valid.*same'
        with self.assertRaisesRegex(ValidationError, expected):
            vd.is_padding('foobar')
