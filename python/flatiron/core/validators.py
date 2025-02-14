import math
# ------------------------------------------------------------------------------


'''
The validators module is function library for validating config attributes.
'''


def is_even(number):
    # type: (int) -> int
    '''
    Validates that number is even.

    Args:
        number (int): Number.

    Raises:
        ValueError: If number is not even.

    Returns:
        int: Input number.
    '''
    if number % 2 != 0:
        msg = f'{number} is not an even number.'
        raise ValueError(msg)
    return number


def is_odd(number):
    # type: (int) -> int
    '''
    Validates that number is odd.

    Args:
        number (int): Number.

    Raises:
        ValueError: If number is not odd.

    Returns:
        int: Input number.
    '''
    if number % 2 == 0:
        msg = f'{number} is not an odd number.'
        raise ValueError(msg)
    return number


def is_base_two(number):
    # type: (int) -> int
    '''
    Validates that number is base two.

    Args:
        number (int): Number.

    Raises:
        ValueError: If number is not base two.

    Returns:
        int: Input number.
    '''
    exp = math.log2(number)
    if exp != int(exp):
        msg = f'{number} is not a base two number.'
        raise ValueError(msg)
    return number


def is_padding(pad_type):
    # type: (str) -> str
    '''
    Validates that pad_type is a legal padding type.

    Args:
        pad_type (str): Padding type.

    Raises:
        ValueError: If padding type is not legal.

    Returns:
        str: Input padding type.
    '''
    legal = ['valid', 'same']
    if pad_type not in legal:
        msg = f'{pad_type} is not a legal padding type. Legal types: {legal}.'
        raise ValueError(msg)
    return pad_type


def is_callback_mode(mode):
    # type: (str) -> str
    '''
    Validates that mode is a legal calback mode.

    Args:
        mode (str): Callback mode.

    Raises:
        ValueError: If mode type is not legal.

    Returns:
        str: Input callback mode.
    '''
    legal = ['auto', 'min', 'max']
    if mode not in legal:
        msg = f'{mode} is not a legal callback mode. Legal types: {legal}.'
        raise ValueError(msg)
    return mode


def is_pipeline_method(method):
    # type: (str) -> str
    '''
    Validates that method is a legal pipeline method.

    Args:
        mode (str): Pipeline method.

    Raises:
        ValueError: If method is not legal.

    Returns:
        str: Input pipeline method.
    '''
    legal = [
        'load',
        'train_test_split',
        'unload',
        'build',
        'compile',
        'train',
    ]
    if method not in legal:
        msg = f'{method} is not a legal pipeline method. Legal methods: {legal}.'
        raise ValueError(msg)
    return method


def is_engine(engine):
    # type: (str) -> str
    '''
    Validates that engine is a legal deep learning framework.

    Args:
        engine (str): Deep learning framework.

    Raises:
        ValueError: If engine is not legal.

    Returns:
        str: Input engine.
    '''
    legal = ['tensorflow', 'torch']
    if engine not in legal:
        msg = f'{engine} is not a legal deep learning framework. Legal engines: {legal}.'
        raise ValueError(msg)
    return engine
