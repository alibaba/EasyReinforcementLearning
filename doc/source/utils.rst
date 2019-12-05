Utils Module Reference
======================

General Utilities
----------------------

.. autofunction:: easy_rl.utils.utils.discount
.. autofunction:: easy_rl.utils.utils.unflatten_tensor
.. autofunction:: easy_rl.utils.utils.prod
.. autofunction:: easy_rl.utils.utils.n_step_adjustment
.. autofunction:: easy_rl.utils.utils.compute_targets


Layer Utilities
---------------

.. autoclass:: easy_rl.utils.layer_utils.DefaultFCNetwork
.. autoclass:: easy_rl.utils.layer_utils.DefaultConvNetwork
.. autoclass:: easy_rl.utils.layer_utils.AutoGraphHandler
.. autoclass:: easy_rl.utils.layer_utils.Layer
.. autoclass:: easy_rl.utils.layer_utils.Dense
.. autoclass:: easy_rl.utils.layer_utils.Conv2d
.. autoclass:: easy_rl.utils.layer_utils.Pooling2d
.. autoclass:: easy_rl.utils.layer_utils.Embedding
.. autoclass:: easy_rl.utils.layer_utils.Reduce
.. autoclass:: easy_rl.utils.layer_utils.BatchNormalization
.. autoclass:: easy_rl.utils.layer_utils.Dropout
.. autoclass:: easy_rl.utils.layer_utils.LayerBuilder
.. autofunction:: easy_rl.utils.layer_utils.process_inputs
.. autofunction:: easy_rl.utils.layer_utils.process_outputs
.. autofunction:: easy_rl.utils.layer_utils.build_model

Action Utilities
----------------

.. autoclass:: easy_rl.utils.action_utils.Distribution
.. autoclass:: easy_rl.utils.action_utils.Categorical
.. autoclass:: easy_rl.utils.action_utils.DiagGaussian
.. autoclass:: easy_rl.utils.action_utils.Identity
.. autofunction:: easy_rl.utils.action_utils.get_action_distribution

Learning Rate Utilities
-----------------------

.. autoclass:: easy_rl.utils.learning_rate_utils.LearningRateStrategy
