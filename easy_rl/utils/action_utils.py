import tensorflow as tf
import numpy as np


class Distribution(object):
    def __init__(self, distribution_params, deterministic_ph, eps_ph=None):
        """Action Distribution of an agent.

        Args:
            distribution_params: the input params of distribution.
            deterministic_ph: placeholder for whether to use deterministic action
            eps_ph: placeholder for epsilon
        """
        self.distribution_params = distribution_params
        self.deterministic_ph = deterministic_ph
        self.eps_ph = eps_ph

    def get_action(self):
        """get an action from the specificed distribution
        """

        return NotImplementedError

    def log_p(self, action):
        """the log-likelihood of the action distribution

        Args:
            action: the selected action
        """

        return NotImplementedError

    def entropy(self):
        """the entropy of the action distribution
        """

        return NotImplementedError

    def kl(self, dist):
        """the KL-divergence between two distribution

        Args:
            dist: the other distribution to calculate the KL-divergence
        """

        return NotImplementedError


class Categorical(Distribution):
    """Categorical distribution for discrete action
    """

    def log_p(self, action):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.distribution_params, labels=action)

    def entropy(self):
        """ entroy = -sum(p*log(p))
        """

        a = self.distribution_params - tf.reduce_max(
            self.distribution_params, axis=1, keep_dims=True)
        ea = tf.exp(a)
        z = tf.reduce_sum(ea, axis=1, keep_dims=True)
        p = ea / z

        return tf.reduce_sum(p * (tf.log(z) - a), axis=1)

    def kl(self, dist):
        """ kl(p||q) = -sum(p*log(q/p))
        """
        a0 = self.distribution_params - tf.reduce_max(
            self.distribution_params, axis=1, keep_dims=True)
        a1 = dist.distribution_params - tf.reduce_max(
            dist.distribution_params, axis=1, keep_dims=True)

        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)

        z0 = tf.reduce_sum(ea0, axis=1, keep_dims=True)
        z1 = tf.reduce_sum(ea1, axis=1, keep_dims=True)

        p0 = ea0 / z0

        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=1)

    def get_action(self):
        params_shape = tf.cast(tf.shape(self.distribution_params), tf.int64)
        batch_size, action_dim = params_shape[0], params_shape[1]

        deterministic_action = tf.argmax(self.distribution_params, axis=1)

        if self.eps_ph is not None:
            random_choice = tf.random_uniform(
                shape=(batch_size, ), minval=0.0, maxval=1.0,
                dtype=tf.float32) < self.eps_ph
            random_valid_action_logits = tf.where(
                tf.equal(self.distribution_params, tf.float32.min),
                tf.ones_like(self.distribution_params) * tf.float32.min,
                tf.ones_like(self.distribution_params))
            uniform_action = tf.squeeze(
                tf.multinomial(random_valid_action_logits, 1), axis=1)
            sample_action = tf.where(random_choice, uniform_action,
                                     deterministic_action)
        else:
            sample_action = tf.squeeze(
                tf.multinomial(self.distribution_params, num_samples=1),
                axis=1)

        output_action = tf.cond(self.deterministic_ph,
                                lambda: deterministic_action,
                                lambda: sample_action)

        return output_action


class DiagGaussian(Distribution):
    """Gaussian distribution for continuous action
    """

    def __init__(self, distribution_params, deterministic_ph):

        assert len(distribution_params) == 2, "Invalid number of variables to parameterize gaussian distribution" \
                                                   "2 is required(mean and log_std), {} supplied".format(len(distribution_params))

        self.mean, self.log_std = distribution_params
        self.std = tf.exp(self.log_std)
        super(DiagGaussian, self).__init__(distribution_params,
                                           deterministic_ph)

    def log_p(self, action):
        """the log-likelihood function of gaussian distribution"""

        return (-0.5 * tf.reduce_sum(
            tf.square((action - self.mean) / self.std), axis=1) -
                0.5 * np.log(2 * np.pi) * tf.to_float(tf.shape(action)[1]) -
                tf.reduce_sum(self.log_std, axis=1))

    def entropy(self):
        return tf.reduce_sum(
            0.5 * self.log_std + 0.5 * np.log(2 * np.pi * np.e), axis=1)

    def kl(self, dist):
        assert isinstance(dist, DiagGaussian), "Invalid type of distribution to calculate KL-divergence, `DiagGaussian` is" \
                                               "expected, but {} got".format(type(dist))

        return tf.reduce_sum(
            dist.log_std - self.log_std +
            (tf.square(self.std) + tf.square(self.mean - dist.mean)) /
            (2.0 * tf.square(dist.std)) - 0.5,
            axis=1)

    def get_action(self):
        deterministic_action = self.mean

        sample_action = self.mean + self.std * tf.random_normal(
            shape=tf.shape(self.mean))

        output_action = tf.cond(self.deterministic_ph,
                                lambda: deterministic_action,
                                lambda: sample_action)

        return output_action


class Identity(Distribution):
    """Identity of the distribution parameters,
    ornstein uhlenbeck noise will be used for exploration"""

    def __init__(self,
                 distribution_params,
                 deterministic_ph,
                 sigma=1.0,
                 theta=0.3,
                 noise_scale=1.0):
        if not isinstance(distribution_params,
                          tf.Tensor) and len(distribution_params) == 2:
            self.mean, self.ornstein_uhlenbeck_state = distribution_params
        else:
            self.mean, self.ornstein_uhlenbeck_state = distribution_params, None
        self._sigma = sigma
        self._theta = theta
        self._noise_scale = noise_scale

        super(Identity, self).__init__(
            distribution_params=distribution_params,
            deterministic_ph=deterministic_ph)

    def get_action(self):
        deterministic_action = self.mean

        if self.ornstein_uhlenbeck_state:
            random_norm = tf.random_normal(
                shape=self.mean.get_shape().as_list()[1:],
                mean=0.0,
                stddev=1.0)
            ou_noise = tf.assign_add(
                self.ornstein_uhlenbeck_state,
                self._theta * (-self.ornstein_uhlenbeck_state) +
                random_norm * self._sigma)
            sample_action = self.mean + ou_noise * self._noise_scale

            output_action = tf.cond(self.deterministic_ph,
                                    lambda: deterministic_action,
                                    lambda: sample_action)
        else:
            output_action = deterministic_action

        return output_action


def get_action_distribution(states_embedding,
                            distribution_type,
                            deterministic_ph,
                            eps_ph=None,
                            **kwargs):
    """ return the specific action distribution
    """
    if distribution_type == "Categorical":
        action_dist = Categorical(
            distribution_params=states_embedding,
            deterministic_ph=deterministic_ph,
            eps_ph=eps_ph)

    elif distribution_type == "DiagGaussian":
        action_dist = DiagGaussian(
            distribution_params=states_embedding,
            deterministic_ph=deterministic_ph)

    elif distribution_type == "Identity":
        action_dist = Identity(
            distribution_params=states_embedding,
            deterministic_ph=deterministic_ph,
            **kwargs)

    else:
        raise NotImplementedError(
            "Ivalid parameter for distribution_type:{}, the available value if one of ``"
            .format(distribution_type))
    return action_dist
