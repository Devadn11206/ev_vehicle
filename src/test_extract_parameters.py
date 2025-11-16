import unittest
import sys, os

# Ensure local directory (contains chatbot.py) is on path
sys.path.append(os.path.dirname(__file__))
from chatbot import extract_parameters

class TestExtractParameters(unittest.TestCase):
    def test_basic_percent(self):
        params = extract_parameters("Estimate range 70% 40kWh at 60km/h 25C")
        self.assertEqual(params['battery_percent'], 70.0)
        self.assertEqual(params['battery_capacity'], 40.0)
        self.assertEqual(params['speed'], 60.0)
        self.assertEqual(params['temp'], 25.0)

    def test_mph_conversion(self):
        params = extract_parameters("Range at 50mph 80% battery 75kWh 10C")
        self.assertAlmostEqual(params['speed'], 50 * 1.60934, places=2)

    def test_fahrenheit_conversion(self):
        params = extract_parameters("Predict range 60% 50kWh 55mph 68F")
        # 68F -> 20C
        self.assertAlmostEqual(params['temp'], 20.0, places=2)

    def test_missing_values(self):
        params = extract_parameters("How far can I go?")
        self.assertFalse('battery_percent' in params)
        self.assertFalse('battery_capacity' in params)
        self.assertFalse('speed' in params)
        self.assertFalse('temp' in params)

    def test_upper_bounds(self):
        params = extract_parameters("Battery 100% Capacity 120kWh speed 120km/h temp -5C")
        self.assertEqual(params['battery_percent'], 100.0)
        self.assertEqual(params['battery_capacity'], 120.0)
        self.assertEqual(params['speed'], 120.0)
        self.assertEqual(params['temp'], -5.0)

if __name__ == '__main__':
    unittest.main()
