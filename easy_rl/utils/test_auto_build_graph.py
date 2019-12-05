from easy_rl.utils import layer_utils
import tensorflow as tf
import numpy as np

# implementation of build network from configuration
# the following network_spec will build a partially shared model graph for transfer learning
#
#============================================================================
#
#   source_output  target_output
#         |              |
#       layer3        layer4
#           \          /
#           layer2(shared)
#           /          \
#       layer0        layer1
#           \          /
#         input_placeholder
#
#
#   "network_spec" : [
#     [
#       {"inputs": "input_ph"}
#       {"type": "dense", "units": 10, "use_bias" : True, "activation": "relu"}
#       {"outputs": "layer0_output"}
#     ]
#     [
#       {"inputs": "input_ph"}
#       {"type": "dense", "units": 10, "use_bias" : True, "activation": "relu"}
#       {"outputs": "layer1_output"}
#     ]
#     [
#       {"inputs": ["layer0_output", "layer1_output"], "propress_type":None}
#       {"type": "dense", "units": 10, "use_bias" : True, "activation": "relu"}
#       {"outputs": ["layer2_output_0", "layer2_output_1"]}
#     ]
#     [
#       {"inputs": "layer2_output_0"}
#       {"type": "dense", "units": 10, "use_bias" : True, "activation": "relu"}
#       {"outputs": "source_output"}
#     ]
#     [
#       {"inputs": "layer2_output_1"}
#       {"type": "dense", "units": 10, "use_bias" : True, "activation": "relu"}
#       {"outputs": "target_output"}
#     ]
#   ]
#
#=============================================================================

tl_model_conf = {
    "network_spec": [[
        {
            "inputs": "input_ph"
        },
        {
            "type": "dense",
            "units": 10,
            "use_bias": False,
            "activation": "relu",
            "kernel_initializer": 1.0
        },
        {
            "outputs": "layer0_output"
        },
    ], [
        {
            "inputs": "input_ph"
        },
        {
            "type": "dense",
            "units": 10,
            "use_bias": False,
            "activation": "relu",
            "kernel_initializer": 1.0
        },
        {
            "outputs": "layer1_output"
        },
    ], [
        {
            "inputs": ["layer0_output", "layer1_output"],
            "process_type": None
        },
        {
            "type": "dense",
            "units": 10,
            "use_bias": True,
            "activation": "relu"
        },
        {
            "outputs": ["layer2_output_0", "layer2_output_1"]
        },
    ], [
        {
            "inputs": "layer2_output_0"
        },
        {
            "type": "dense",
            "units": 1,
            "use_bias": False,
            "activation": None,
            "kernel_initializer": -1.0
        },
        {
            "outputs": "source_output"
        },
    ], [{
        "inputs": "layer2_output_1"
    }, {
        "type": "dense",
        "units": 1,
        "use_bias": False,
        "activation": None,
        "kernel_initializer": 1.0
    }, {
        "outputs": "target_output"
    }]]
}

data_input = tf.placeholder(dtype=tf.float32, shape=(None, 4))
is_training = tf.placeholder_with_default(True, shape=())


def build_model(input_ph, is_training_ph, config):

    outputs = layer_utils.build_model(
        inputs=input_ph,
        network_spec=config.get('network_spec'),
        is_training_ph=is_training_ph)
    return outputs


source_output, target_output = build_model(data_input, is_training,
                                           tl_model_conf)

# get graph_handler to obtain the intermediate output tensor
graph_handler = layer_utils.AutoGraphHandler()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

fetches = [
    source_output, target_output,
    graph_handler.get_hiddens('layer2_output_0'),
    graph_handler.get_hiddens('layer2_output_1')
]
res = sess.run(
    fetches=fetches, feed_dict={data_input: np.random.random((10, 4))})

specific_output0, specific_output1, shared_output0, shared_output1 = res

assert np.sum(np.abs(shared_output0 - shared_output1)) == 0, "the output from shared network should be equal" \
                                                             "in the case that all the variables initialized" \
                                                             "with the same initializer"
assert np.sum(
    np.abs(specific_output0 -
           specific_output1)) > 0, "the final output should be different"
