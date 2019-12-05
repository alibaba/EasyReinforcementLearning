# Copyright 2017 reinforce.io. All Rights Reserved.
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

import os

from setuptools import setup, find_packages

install_requires = ['numpy', 'six', 'scipy', 'pillow', 'pytest']

setup_requires = ['numpy', 'recommonmark']

extras_require = {
    'tf': ['tensorflow>=1.3.0'],
    'tf_gpu': ['tensorflow-gpu>=1.3.0'],
    'gym': ['gym>=0.7.4'],
    'universe': ['universe>=0.21.3'],
    'mazeexp': ['mazeexp>=0.0.1'],
    'yapf': ['yapf==0.23.0'],
    'flake8': ['flake8==3.7.7', 'flake8-quotes==2.1.0']
}

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name='easy_rl',
    version=
    '0.0.1',  # please remember to edit easy_rl/__init__.py in response, once updating the version
    description='A Reinforcement Learning Package',
    url='https://github.com/alibaba/EasyRL',
    author='Alibaba PAI Innovative Algorithm Group',
    author_email='ContactPAI',
    packages=[
        package for package in find_packages()
        if package.startswith('easy_rl')
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False)
