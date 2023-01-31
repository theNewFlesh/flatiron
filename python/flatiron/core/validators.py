from typing import Any
from schematics.exceptions import ValidationError

import math
# ------------------------------------------------------------------------------


'''
The validators module is function library for validating config attributes.
'''


def is_gte(a, b):
    # type: (Any, Any) -> bool
    '''
    Validates that a is greater than or equal to b.

    Args:
        a (object): Object.
        b (object): Object.

    Raises:
        ValidationError: If a is not greater than or equal to b.

    Returns:
        bool: A is greater than or equal to b.
    '''
    if not a >= b:
        msg = f'{a} is not greater than or equal to {b}.'
        raise ValidationError(msg)


def is_even(number):
    # type: (int) -> None
    '''
    Validates that number is even.

    Args:
        number (int): Number.

    Raises:
        ValidationError: If number is not even.
    '''
    if number % 2 != 0:
        msg = f'{number} is not an even number.'
        raise ValidationError(msg)


def is_odd(number):
    # type: (int) -> None
    '''
    Validates that number is odd.

    Args:
        number (int): Number.

    Raises:
        ValidationError: If number is not odd.
    '''
    if number % 2 == 0:
        msg = f'{number} is not an odd number.'
        raise ValidationError(msg)


def is_base_two(number):
    # type: (int) -> None
    '''
    Validates that number is base two.

    Args:
        number (int): Number.

    Raises:
        ValidationError: If number is not base two.
    '''
    exp = math.log2(number)
    if exp != int(exp):
        msg = f'{number} is not a base two number.'
        raise ValidationError(msg)


def is_padding(ptype):
    # type: (str) -> None
    '''
    Validates that ptype is a legal padding type.

    Args:
        ptype (str): Padding type.

    Raises:
        ValidationError: If padding type is not legal.
    '''
    legal = ['valid', 'same']
    if ptype not in legal:
        msg = f'{ptype} is not a legal padding type. Legal types: {legal}.'
        raise ValidationError(msg)
