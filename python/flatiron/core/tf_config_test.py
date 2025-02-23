import inspect
import re
import unittest

import flatiron.core.tf_config as fi_tfconfig
import flatiron.tf.optimizer as fi_tfoptim
import flatiron.tf.loss as fi_tfloss
# ------------------------------------------------------------------------------


def get_classes(module):
    members = inspect.getmembers(module)
    return dict(filter(lambda x: inspect.isclass(x[1]), members))


def find_classes(module, regex):
    classes = get_classes(module)
    output = filter(lambda x: re.search(regex, x), classes.keys())
    output = filter(lambda x: not re.search('BaseConfig', x), output)
    output = [classes[k] for k in output]
    return output


def config_class_to_library_class(config_class, prefix, module):
    name = re.sub(prefix, '', config_class.__name__)
    return module.get(dict(name=name))


def get_class_and_kwargs(prefix, config_module, library_module, required):
    configs = find_classes(config_module, prefix)
    for config_class in configs:
        config = config_class.model_validate(required).model_dump()
        name = re.sub(prefix, '', config_class.__name__)
        config['name'] = name
        try:
            yield library_module.get(config)
        except Exception as e:
            raise e


class TFConfigTests(unittest.TestCase):
    def test_optimizer(self):
        req = dict(name='test')
        results = get_class_and_kwargs('TFOpt', fi_tfconfig, fi_tfoptim, req)
        self.assertGreater(len(list(results)), 0)

    def test_loss(self):
        req = dict(name='test', dtype='float16')
        results = get_class_and_kwargs('TFLoss', fi_tfconfig, fi_tfloss, req)
        self.assertGreater(len(list(results)), 0)
