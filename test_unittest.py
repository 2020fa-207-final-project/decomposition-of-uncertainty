"""Tests for `utils` package."""
from unittest import TestCase, main
from utils.test import add_two_numbers

class exampleTests(TestCase):
    def test_basic(self):
        self.assertEqual(1, 1)

    def test_basic2(self):
        assert isinstance(1, int)

class testTests(TestCase):
    def test_addition(self):
        self.assertEqual(10, add_two_numbers(5,5))

if __name__ == "__main__":
    main()
