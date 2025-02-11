from typing import Any  # noqa F401

import math
# ------------------------------------------------------------------------------


'''
The validators module is function library for validating config attributes.
'''


def is_gte(a, b):
    # type: (Any, Any) -> None
    '''
    Validates that a is greater than or equal to b.

    Args:
        a (object): Object.
        b (object): Object.

    Raises:
        ValueError: If a is not greater than or equal to b.

    Returns:
        bool: A is greater than or equal to b.
    '''
    if not a >= b:
        msg = f'{a} is not greater than or equal to {b}.'
        raise ValueError(msg)


def is_even(number):
    # type: (int) -> None
    '''
    Validates that number is even.

    Args:
        number (int): Number.

    Raises:
        ValueError: If number is not even.
    '''
    if number % 2 != 0:
        msg = f'{number} is not an even number.'
        raise ValueError(msg)


def is_odd(number):
    # type: (int) -> None
    '''
    Validates that number is odd.

    Args:
        number (int): Number.

    Raises:
        ValueError: If number is not odd.
    '''
    if number % 2 == 0:
        msg = f'{number} is not an odd number.'
        raise ValueError(msg)


def is_base_two(number):
    # type: (int) -> None
    '''
    Validates that number is base two.

    Args:
        number (int): Number.

    Raises:
        ValueError: If number is not base two.
    '''
    exp = math.log2(number)
    if exp != int(exp):
        msg = f'{number} is not a base two number.'
        raise ValueError(msg)


def is_padding(pad_type):
    # type: (str) -> None
    '''
    Validates that pad_type is a legal padding type.

    Args:
        pad_type (str): Padding type.

    Raises:
        ValueError: If padding type is not legal.
    '''
    legal = ['valid', 'same']
    if pad_type not in legal:
        msg = f'{pad_type} is not a legal padding type. Legal types: {legal}.'
        raise ValueError(msg)


def is_callback_mode(mode):
    # type: (str) -> None
    '''
    Validates that mode is a legal calback mode.

    Args:
        mode (str): Callback mode.

    Raises:
        ValueError: If mode type is not legal.
    '''
    legal = ['auto', 'min', 'max']
    if mode not in legal:
        msg = f'{mode} is not a legal callback mode. Legal types: {legal}.'
        raise ValueError(msg)


def is_pipeline_methods(methods):
    # type: (list[str]) -> None
    '''
    Validates that each method in a list is a legal pipeline method.

    Args:
        mode (list[str]): Pipeline methods.

    Raises:
        ValueError: If any method illegal.
    '''
    legal = [
        'load',
        'train_test_split',
        'unload',
        'build',
        'compile',
        'fit',
    ]
    bad = list(filter(lambda x: x not in legal, methods))
    if bad != []:
        msg = ', '.join(bad)
        msg = f'[{msg}] are not legal pipeline methods. Legal methods: {legal}.'
        raise ValueError(msg)


def is_engine(engine):
    # type: (str) -> None
    '''
    Validates that engine is a legal deep learning framework.

    Args:
        engine (str): Deep learning framework.

    Raises:
        ValueError: If engine is not legal.
    '''
    legal = ['tensorflow', 'pytorch']
    if engine not in legal:
        msg = f'{engine} is not a legal deep learning framework. Legal engines: {legal}.'
        raise ValueError(msg)
