#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __init__.py
"""
Testing for dphtools.

Copyright (c) 2021, David Hoffman
"""

import unittest


class InitializationTests(unittest.TestCase):
    """Ensure basic testing works."""

    def test_initialization(self):
        """Check the test suite runs by affirming 2+2=4."""
        self.assertEqual(2 + 2, 4)

    def test_import(self):
        """Ensure the test suite can import our module."""
        try:
            import dphtools
        except ImportError:
            self.fail("Was not able to import the dphutils package")
