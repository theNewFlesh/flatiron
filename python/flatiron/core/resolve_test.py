from copy import deepcopy
import unittest

import flatiron.core.resolve as rez
import flatiron.tf.models.dummy as fi_tfdummy
# ------------------------------------------------------------------------------


class PipelineTestBase(unittest.TestCase):
    def get_config(self):
        return dict(
            framework=dict(
                name='tensorflow',
                device='cpu',
            ),
            model=dict(
                shape=[10, 10, 3]
            ),
            optimizer=dict(name='SGD'),
            loss=dict(name='dice_loss'),
            metrics=[dict(name='jaccard'), dict(name='dice')],
            dataset=dict(
                source='/mnt/data/dataset',
                labels=[2],
                label_axis=-1,
            ),
            callbacks=dict(
                project='proj',
                root='/root',
            ),
            train=dict(
                epochs=1,
            ),
            logger=dict(
                slack_url='https://hooks.slack.com/services/fake-service',
                slack_channel='test',
                slack_methods=['load'],
            ),
        )

    def get_simple_config(self):
        return dict(
            framework=dict(
                name='torch'
            ),
            model=dict(
                shape=[10, 10, 3]
            ),
            optimizer=dict(name='SGD'),
            loss=dict(name='MSELoss'),
            metrics=[dict(name='Dice')],
            dataset=dict(
                source='/mnt/data/dataset',
            ),
            callbacks=dict(
                project='proj',
                root='/root',
            ),
            train=dict(),
            logger=dict(),
        )

    def test_generate_config(self):
        result = rez._generate_config()
        self.assertEqual(result['framework']['name'], 'torch')

        result = rez._generate_config(framework='tensorflow')
        self.assertEqual(result['framework']['name'], 'tensorflow')

    def test_resolve_config(self):
        config = self.get_simple_config()
        result = rez.resolve_config(config, fi_tfdummy.DummyConfig)

        self.assertEqual(result['framework']['device'], 'cpu')
        self.assertEqual(result['model']['activation'], 'relu')
        self.assertEqual(result['optimizer']['momentum'], 0)
        self.assertEqual(result['loss']['reduce'], None)
        self.assertEqual(result['dataset']['label_axis'], -1)
        self.assertEqual(result['callbacks']['verbose'], 0)
        self.assertEqual(result['train']['batch_size'], 32)
        self.assertEqual(result['logger']['level'], 'warn')

    def test_resolve_model(self):
        config = self.get_config()
        expected = {'shape': [10, 10, 3]}
        self.assertEqual(config['model'], expected)

        result = rez._resolve_model(config, fi_tfdummy.DummyConfig)
        expected = {'shape': [10, 10, 3], 'activation': 'relu'}
        self.assertEqual(result['model'], expected)

    def test_resolve_pipeline(self):
        config = self.get_config()
        tz = config['logger'].get('timezone', None)
        self.assertIs(tz, None)

        result = rez._resolve_pipeline(config)
        self.assertEqual(result['logger']['timezone'], 'UTC')

        expected = {'shape': [10, 10, 3]}
        self.assertEqual(result['model'], expected)

    def test_resolve_field(self):
        config = self.get_config()
        result = rez._resolve_field(deepcopy(config), 'optimizer')
        res = result['optimizer']
        self.assertEqual(res['name'], 'SGD')
        self.assertIs(res['clipnorm'], None)

        del config['optimizer']
        del result['optimizer']
        self.assertEqual(result, config)

    def test_resolve_field_metrics(self):
        config = self.get_config()
        result = rez._resolve_field(config, 'metrics')['metrics']
        self.assertEqual(result[0]['name'], 'jaccard')
        self.assertEqual(result[1]['name'], 'dice')

    def test_resolve_subconfig(self):
        result = rez._resolve_subconfig(
            dict(name='Huber'),
            'TFLoss',
            True,
            'flatiron.tf.config',
            'flatiron.tf.loss',
        )
        self.assertEqual(result['name'], 'Huber')
        self.assertIs(result['dtype'], None)

    def test_resolve_subconfig_config_module(self):
        result = rez._resolve_subconfig(
            dict(name='jaccard_loss'),
            'TFLoss',
            False,
            'flatiron.tf.loss',
            None,
        )
        self.assertEqual(result['name'], 'jaccard_loss')

    def test_resolve_subconfig_no_prepend(self):
        result = rez._resolve_subconfig(
            dict(name='tensorflow'),
            'TFFramework',
            False,
            'flatiron.tf.config',
            None,
        )
        self.assertEqual(result['name'], 'tensorflow')
        self.assertIs(result['jit_compile'], False)

    def test_resolve_subconfig_custom(self):
        expected = dict(name='jaccard_loss')
        result = rez._resolve_subconfig(
            expected,
            'TFLoss',
            True,
            'flatiron.tf.config',
            'flatiron.tf.loss',
        )
        self.assertEqual(result, expected)
