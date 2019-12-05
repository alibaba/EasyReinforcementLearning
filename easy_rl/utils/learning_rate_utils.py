import tensorflow as tf


class LearningRateStrategy(object):
    def __init__(self, init_lr, strategy_spec):
        self._type = strategy_spec.pop('type', 'exponential_decay')
        self._decay_steps = strategy_spec.pop('decay_steps', 1000)
        self._decay_rate = strategy_spec.pop('decay_rate', 0.9)
        self._kwargs = strategy_spec
        self._init_lr = init_lr

    def __call__(self, global_step):
        if self._type == 'exponential_decay':
            lr = tf.train.exponential_decay(
                learning_rate=self._init_lr,
                global_step=global_step,
                decay_steps=self._decay_steps,
                decay_rate=self._decay_rate,
                **self._kwargs)
        elif self._type == 'polynomial_decay':
            lr = tf.train.polynomial_decay(
                learning_rate=self._init_lr,
                global_step=global_step,
                decay_steps=self._decay_steps,
                **self._kwargs)
        elif self._type == 'natural_exp_decay':
            lr = tf.train.natural_exp_decay(
                learning_rate=self._init_lr,
                global_step=global_step,
                decay_steps=self._decay_steps,
                decay_rate=self._decay_rate**self._kwargs)
        elif self._type == 'inverse_time_decay':
            lr = tf.train.inverse_time_decay(
                learning_rate=self._init_lr,
                global_step=global_step,
                decay_steps=self._decay_steps,
                decay_rate=self._decay_rate,
                **self._kwargs)

        return lr
