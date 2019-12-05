import tensorflow as tf
from tensorflow import layers
from functools import wraps
import copy


class DefaultFCNetwork(object):
    def __init__(self,
                 action_dim,
                 hidden_size=(
                     64,
                     64,
                 ),
                 activation=tf.nn.relu,
                 need_v=False,
                 input_action=None):
        self._hidden_size = hidden_size
        self._need_v = need_v
        self._action_dim = action_dim
        self._activation = activation
        # construct network to encode obs-action if input_action supplied
        self._input_action = input_action

    def __call__(self, input_obs):
        with tf.variable_scope("default_fc"):
            flatten_inputs = []
            if isinstance(input_obs, dict):
                flatten_inputs.extend([ts for _, ts in input_obs.items()])
            elif isinstance(input_obs, tf.Tensor):
                flatten_inputs = [input_obs]
            else:
                flatten_inputs = input_obs
            if self._input_action is not None:
                flatten_inputs.append(self._input_action)
            flatten_inputs = tf.concat(
                flatten_inputs, axis=1, name='flatten_inputs')

            hi = flatten_inputs
            if len(self._hidden_size) > 0:
                for i, hs in enumerate(self._hidden_size):
                    ho = layers.dense(
                        hi,
                        hs,
                        activation=tf.nn.relu,
                        name="h{}".format(i + 1))
                    hi = ho
            logits = layers.dense(
                hi, units=self._action_dim, activation=None, name="logits")
            if self._need_v:
                logits_to_v = layers.dense(
                    flatten_inputs,
                    units=256,
                    activation=tf.nn.relu,
                    name="logits_to_value")
                v = layers.dense(
                    logits_to_v,
                    units=64,
                    activation=tf.nn.relu,
                    name="value_hidden")
                v = layers.dense(v, units=1, activation=None, name="value")
                v = tf.squeeze(v, [-1])
                return (
                    logits,
                    v,
                )

            return logits


class DefaultConvNetwork(object):
    def __init__(self,
                 action_dim,
                 conv_filters=((16, (8, 8), 4, 'same'),
                               (32, (4, 4), 2, 'same'), (256, (11, 11), 1,
                                                         'valid')),
                 activation=tf.nn.relu,
                 need_v=False,
                 input_action=None):
        self._conv_filters = conv_filters
        self._need_v = need_v
        self._action_dim = action_dim
        self._activation = activation
        # construct network to encode obs-action if input_action supplied
        self._input_action = input_action

    def __call__(self, input_obs):
        with tf.variable_scope("default_conv_net"):
            inputs = input_obs
            for i, (filters_, kernel_size, stride,
                    padding) in enumerate(self._conv_filters):
                inputs = layers.conv2d(
                    inputs=inputs,
                    filters=filters_,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    activation=self._activation,
                    name="conv{}".format(i))
            hidden = layers.flatten(inputs)
            if self._input_action:
                hidden = tf.concat([hidden, self._input_action], axis=1)
            logits = layers.dense(
                hidden, units=self._action_dim, activation=None, name="logits")
            if self._need_v:
                logits_to_v = layers.dense(
                    hidden,
                    units=32,
                    activation=tf.nn.relu,
                    name="logits_to_value")
                v = layers.dense(
                    logits_to_v, units=1, activation=None, name="value")
                v = tf.squeeze(v, [-1])
                return (
                    logits,
                    v,
                )

            return logits


def get_singleton(cls):
    instances = {}

    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
            return instances[cls]
        return instances[cls]

    return get_instance


@get_singleton
class AutoGraphHandler(object):
    """AutoGraphHandler will maintain input/output tensor of block in the building process.
    As a singleton, tensors maintained by AutoGraphHandler can be shared during different
    building process(tensor generated in one call of `build_graph` can be used in another call of `build_graph`)
    """

    def __init__(self,
                 is_training_ph=tf.placeholder_with_default(True, shape=())):
        self._hiddens = {}
        self._outputs = {}
        self._inputs = {}
        self.is_training_ph = is_training_ph

    def update_block_hiddens(self, hiddens):
        for name, ts in hiddens.items():
            if name in self._hiddens:
                raise ValueError(
                    "output tensor of block should be unique, but {} has already exists"
                    .format(name))
            self._hiddens[name] = ts

    def get_hiddens(self, name):
        if name not in self._hiddens:
            raise ValueError(
                "Can not find tensor {}, all available tensor are{}".format(
                    name, self._hiddens.keys()))
        return self._hiddens[name]

    def update_block_inputs(self, inputs):
        for name, ts in inputs.items():
            self._inputs[name] = ts

    def get_final_outputs(self):
        outputs = []
        for name, ts in self._hiddens.items():
            if name not in self._inputs and name not in self._outputs:
                self._outputs[name] = ts
                outputs.append(ts)
        return tuple(outputs)


class Layer(object):

    activation_dict = {
        "relu": tf.nn.relu,
        "tanh": tf.nn.tanh,
        "sigmoid": tf.nn.sigmoid,
        "elu": tf.nn.elu,
        "selu": tf.nn.selu
    }

    initializer_dict = {
        "random_normal_initializer": tf.random_normal_initializer,
        "random_uniform_initializer": tf.random_uniform_initializer,
        "glorot_normal_initializer": tf.glorot_normal_initializer,
        "glorot_uniform_initializer": tf.glorot_uniform_initializer
    }

    def __init__(self, name, layer_conf):
        self._name = layer_conf.pop('name', None) or name
        activation_name = layer_conf.get('activation', None)
        if activation_name:
            layer_conf['activation'] = Layer.activation_dict[activation_name]

        self._kernel_initializer = layer_conf.pop('kernel_initializer', None)
        if isinstance(self._kernel_initializer, str):
            assert self._kernel_initializer in ('random_normal_initializer',
                                                'random_uniform_initializer',
                                                'glorot_normal_initializer',
                                                'glorot_uniform_initializer'), \
                "Invalid value of kernel_initializer, available value is one of " \
                "['random_normal_initializer', 'random_uniform_initializer'," \
                "'glorot_normal_initializer', 'glorot_uniform_initializer']"

            self._kernel_initializer = Layer.initializer_dict[
                self._kernel_initializer]
        elif (isinstance(self._kernel_initializer, int)
              or isinstance(self._kernel_initializer, float)):
            self._kernel_initializer = tf.constant_initializer(
                value=self._kernel_initializer)


class Dense(Layer):
    def __init__(self, name, layer_conf):
        super(Dense, self).__init__(name, layer_conf)
        self._units = layer_conf.pop('units')
        self._layer_conf = layer_conf

        assert self._units is not None, ""

    def __call__(self, inputs):
        return layers.dense(
            inputs=inputs,
            units=self._units,
            kernel_initializer=self._kernel_initializer,
            name=self._name,
            **self._layer_conf)


class Conv2d(Layer):
    def __init__(self, name, layer_conf):
        super(Conv2d, self).__init__(name, layer_conf)
        self._filters = layer_conf.pop('filters')
        self._kernel_size = layer_conf.pop('kernel_size')
        self._layer_conf = layer_conf

        assert self._filters is not None, ""
        assert self._kernel_size is not None, ""

    def __call__(self, inputs):
        return layers.conv2d(
            inputs=inputs,
            filters=self._filters,
            kernel_size=self._kernel_size,
            kernel_initializer=self._kernel_initializer,
            name=self._name,
            **self._layer_conf)


class Pooling2d(Layer):
    def __init__(self, name, layer_conf):
        super(Pooling2d, self).__init__(name, layer_conf)
        self._mode = layer_conf.pop('mode')
        self._mode = self._mode.lower()
        assert self._mode in ['avg', 'max'], "mode of Pooling2d should be one of ['avg', 'max']" \
                                                     "but {} found".format(self._mode)
        self._pool_size = layer_conf.pop('pool_size', None)
        assert self._pool_size is not None, ""
        self._strides = layer_conf.pop('strides', None)
        assert self._strides is not None, ""

        self._layer_conf = layer_conf

    def __call__(self, inputs):
        if self._mode == 'avg':
            return layers.AveragePooling2D(
                pool_size=self._pool_size,
                strides=self._strides,
                **self._layer_conf)
        else:
            return layers.MaxPooling2D(
                pool_size=self._pool_size,
                strides=self._strides,
                **self._layer_conf)


class Embedding(Layer):
    def __init__(self, name, layer_conf):
        super(Embedding, self).__init__(name, layer_conf)
        self._vocab_size = layer_conf.pop('vocab_size')
        self._embed_dim = layer_conf.pop('embed_dim')
        self._padding_id = layer_conf.pop('padding_id', 0)

    def __call__(self, inputs):
        self._params = tf.get_variable(
            name=self._name,
            shape=(self._vocab_size, self._embed_dim),
            dtype=tf.float32,
            initializer=self._kernel_initializer)
        return tf.nn.embedding_lookup(self._params, inputs)


class Reduce(Layer):
    def __init__(self, name, layer_conf):
        super(Reduce, self).__init__(name, layer_conf)
        self._mode = layer_conf.pop('mode')
        self._mode = self._mode.lower()
        assert self._mode in ['mean', 'max', 'min', 'sum', 'prod'], "mode of Reduce should be one of" \
            " ['mean', 'max', 'min', 'sum', 'prod'] but {} found".format(self._mode)
        self._layer_conf = layer_conf

    def __call__(self, inputs):
        if self._mode == "mean":
            return tf.reduce_mean(inputs, **self._layer_conf)
        elif self._mode == "sum":
            return tf.reduce_sum(inputs, **self._layer_conf)
        elif self._mode == 'max':
            return tf.reduce_max(inputs, **self._layer_conf)
        elif self._mode == 'min':
            return tf.reduce_min(inputs, **self._layer_conf)
        else:
            return tf.reduce_prod(inputs, **self._layer_conf)


class BatchNormalization(Layer):
    def __init__(self, name, layer_conf):
        super(BatchNormalization, self).__init__(name, layer_conf)
        self._is_training = layer_conf.pop('training')

    def __call__(self, inputs):
        return layers.batch_normalization(
            inputs, training=self._is_training, name=self._name)


class Dropout(Layer):
    def __init__(self, name, layer_conf):
        super(Dropout, self).__init__(name, layer_conf)
        self._is_training = layer_conf.pop('training')

    def __call__(self, inputs):
        return layers.dropout(inputs, training=self._is_training)


class LayerBuilder(object):

    tf_layers = {
        "dense": Dense,
        "conv2d": Conv2d,
        "pooling2d": Pooling2d,
        "flatten": layers.flatten,
        "embedding": Embedding,
        "batch_normalization": layers.batch_normalization,
        "reduce": Reduce,
        "dropout": Dropout
    }

    def __init__(self, layer_conf, is_training_ph):
        _layer_conf = copy.deepcopy(layer_conf)
        type_ = _layer_conf.pop("type")
        type_ = type_.lower()

        if type_ in ['dropout', 'batch_normalization']:
            layer_conf.update({'training': is_training_ph})

        self._layer = LayerBuilder.tf_layers[type_](type_, _layer_conf)
        self._kwargs = _layer_conf

    def __call__(self, inputs):
        return self._layer(inputs)


def process_inputs(input_layer_conf, graph_handler):
    """process the inputs of the block, if `Inputs` is not defined in first layer of block
    default key `Inputs` will be add to inputs tensor

    Arguments:
        input_layer_conf: the configuration of first layer for each block.
        graph_handler: the handler for graph building.
    """
    input_tensor_names = input_layer_conf.get('inputs')
    process_type = input_layer_conf.get('process_type', None)
    if process_type:
        process_type = process_type.lower()
    assert process_type in [
        None, 'concat', 'add', 'minus', 'multipy', 'divide'
    ]

    if isinstance(input_tensor_names, list):
        input_ts = [
            graph_handler.get_hiddens(ts_name)
            for ts_name in input_tensor_names
        ]
        graph_handler.update_block_inputs(
            {ts_name: ts
             for ts_name, ts in zip(input_tensor_names, input_ts)})

        if process_type is None:
            return input_ts
        elif process_type is 'concat':
            return [tf.concat(input_ts, axis=1)]
        elif process_type is 'add':
            return [tf.add_n(input_ts)]
        elif process_type in ('minus', 'multipy', 'divide'):
            assert len(
                input_ts
            ) == 2, "operator {} need two input tensors with the same shape"
            if process_type == 'minus':
                return [input_ts[0] - input_ts[1]]
            elif process_type == 'multipy':
                return [input_ts[0] * input_ts[1]]
            else:
                return [input_ts[0] / input_ts[1]]
    else:
        ts = graph_handler.get_hiddens(input_tensor_names)
        graph_handler.update_block_inputs({input_tensor_names: ts})
        return [ts]


def process_outputs(outputs, output_layer_conf, graph_handler, block_id):
    """assign the name specified in the configuration to the output tensor of the layer block,
    output tensor will be treated as hidden state to build connection between different blocks.

    Arguments:
        outputs: the output tensor of each block.
        output_layer_conf: the configuration of last layer for each block.
        graph_handler: the handler for graph building.
        block_id: the index of block.
    """
    output_tensor_names = output_layer_conf.get('outputs')
    if not isinstance(output_tensor_names, list):
        output_tensor_names = [output_tensor_names]

    assert len(output_tensor_names) == len(outputs), "number of block output is mismatched," \
                                                     " {} expected, but {} given in block:{}".format(
        len(output_tensor_names), len(outputs), block_id)

    output_dict = {name: ts for name, ts in zip(output_tensor_names, outputs)}
    graph_handler.update_block_hiddens(output_dict)


def build_model(inputs,
                network_spec,
                is_training_ph=tf.placeholder_with_default(True, shape=())):
    """build the computation graph according to configuration of network_spec.
    the purpose of this function is to automatically build computation graph through
    configuration for those who do not want to override the `encode_states` function.

    Arguments:
        inputs: input (dict of) tensor of observation.
        network_spec: network configuration.
        is_training_ph: placeholder of is_training to indicate the training stage
    """

    graph_handler = AutoGraphHandler(is_training_ph)
    if isinstance(inputs, dict):
        graph_handler.update_block_hiddens(inputs)
    else:
        graph_handler.update_block_hiddens({"input_ph": inputs})

    for net_block in network_spec:
        if "inputs" not in net_block[0]:
            # add inputs for first layer with fixed name `input_ph`
            net_block.insert(0, {"inputs": "input_ph"})

    # build graph circulation
    for block_id, net_block in enumerate(network_spec):
        processed_inputs = process_inputs(net_block[0], graph_handler)

        output = []
        with tf.variable_scope(
                name_or_scope="block-{}".format(block_id),
                reuse=tf.AUTO_REUSE):
            for input_ts in processed_inputs:
                for i, layer_conf in enumerate(net_block[1:-1]):
                    if "name" not in layer_conf:
                        assert 'type' in layer_conf, "can not find type in {}, {}".format(
                            layer_conf, net_block[1:])
                        layer_conf[
                            "name"] = layer_conf['type'] + '-{}'.format(i)
                    input_ts = LayerBuilder(
                        layer_conf, graph_handler.is_training_ph)(input_ts)
                output.append(input_ts)

        process_outputs(output, net_block[-1], graph_handler, block_id)

    return graph_handler.get_final_outputs()
