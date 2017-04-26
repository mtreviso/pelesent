# -*- coding: utf-8 -*-
"""
pelesent
~~~~~~~~~~~~~~~~~~~

Sentiment analysis from pelenudos to pelenudos

:copyright: (c) 2017 by Marcos Treviso
:licence: MIT, see LICENSE for more details
"""
from __future__ import absolute_import, unicode_literals
import logging

import theano

theano.config.floatX = 'float32'  # XXX: has to come before loading anything related to Theano or Keras


# Generate your own AsciiArt at:
# patorjk.com/software/taag/#f=Calvin%20S&t=PeleSent
__banner__ = r"""
╔═╗┌─┐┬  ┌─┐╔═╗┌─┐┌┐┌┌┬┐
╠═╝├┤ │  ├┤ ╚═╗├┤ │││ │ 
╩  └─┘┴─┘└─┘╚═╝└─┘┘└┘ ┴ 
"""

__prog__ = 'pelesent'
__title__ = 'PeleSent'
__summary__ = 'Sentiment analysis from pelenudos to pelenudos.'
__uri__ = 'https://www.github.com/mtreviso/pelesent'

__version__ = '0.0.1-alpha'

__author__ = 'Marcos Treviso'
__email__ = 'marcostreviso@usp.br'

__license__ = 'MIT'
__copyright__ = 'Copyright 2017 Marcos Treviso'

# the user should dictate what happens when a logging event occurs
logging.getLogger(__name__).addHandler(logging.NullHandler())
