from copy import deepcopy
import inspect
import re
import unittest

import flatiron.tf.config as fi_tfconfig
import flatiron.tf.loss as fi_tfloss
import flatiron.tf.metric as fi_tfmetric
import flatiron.tf.optimizer as fi_tfoptim
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
    lut = dict(
        IoU=dict(num_classes=1, target_class_ids=[0]),
        OneHotIoU=dict(num_classes=1, target_class_ids=[0]),
        MeanIoU=dict(num_classes=1),
        OneHotMeanIoU=dict(num_classes=1),
        PrecisionAtRecall=dict(recall=1.0),
        RecallAtPrecision=dict(precision=1.0),
        SensitivityAtSpecificity=dict(specificity=1.0),
        SpecificityAtSensitivity=dict(sensitivity=1.0),
    )
    configs = find_classes(config_module, prefix)
    for config_class in configs:
        req = deepcopy(required)

        name = re.sub(prefix, '', config_class.__name__)
        fix = lut.get(name, {})
        if fix != {}:
            req.update(fix)

        config = config_class.model_validate(req).model_dump()
        config['name'] = name
        try:
            yield library_module.get(config)
        except Exception as e:
            raise e


class TFConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(
            name='tensorflow',
            device='gpu',
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=False,
            steps_per_execution=1,
            jit_compile=False,
            auto_scale_loss=True,
        )

    def test_framework_validate(self):
        fi_tfconfig.TFFramework.model_validate(self.get_config())

    def test_framework_defaults(self):
        expected = self.get_config()
        result = fi_tfconfig.TFFramework().model_dump()
        self.assertEqual(result, expected)

    def test_optimizer(self):
        req = dict(name='test')
        results = get_class_and_kwargs('TFOpt', fi_tfconfig, fi_tfoptim, req)
        self.assertGreater(len(list(results)), 0)

    def test_loss(self):
        req = dict(name='test', dtype='float16')
        results = get_class_and_kwargs('TFLoss', fi_tfconfig, fi_tfloss, req)
        self.assertGreater(len(list(results)), 0)

    def test_metric(self):
        req = dict(name='test')
        results = get_class_and_kwargs('TFMetric', fi_tfconfig, fi_tfmetric, req)
        self.assertGreater(len(list(results)), 0)
