# Copyright (c) 2019 Alibaba Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

__version__ = '0.0.1'

import logging


def _setup_logger():
    """Set up the root logger of this module.
    Any submodule's logger would regard this logger as their ancestor.
    We provided a stream handler which by-default prints messages into stderr.
    Users can add more handlers for this logger in their entry script.
    See demo/use_logger.py for more details.
    """
    logger = logging.getLogger("easy_rl")
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s\t%(levelname)s %(filename)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


_setup_logger()
