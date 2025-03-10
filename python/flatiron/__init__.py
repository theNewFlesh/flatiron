import importlib.metadata as __meta

import flatiron.api  # noqa F401
import flatiron.command  # noqa F401
import flatiron.core  # noqa F401

try:
    __extras = __meta.metadata('flatiron').get_all('Provides-Extra', [])
except __meta.PackageNotFoundError:
    __extras = ['all']

if 'all' in __extras or 'tensorflow' in __extras:
    import flatiron.tf  # noqa F401

if 'all' in __extras or 'torch' in __extras:
    import flatiron.torch  # noqa F401
