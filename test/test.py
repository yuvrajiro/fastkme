

import numpy as np
import numba
from fastkme.kme import kaplan_meier_estimator, kaplan_meier_estimator_w
from unittest import TestCase
import unittest

class TestKaplanMeierEstimator(TestCase):
    def test_kaplan_meier_estimator(self):
        event = np.array([0, 1, 1, 0, 1, 1, 1, 0])
        time_exit = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        uniq_times, prob_survival = kaplan_meier_estimator(event, time_exit)
        expected_uniq_times = np.array([1, 2, 3, 4, 5, 6, 7, 8])

        expected_prob_survival = np.array([1.      , 0.857143, 0.714286, 0.714286, 0.535714, 0.357143,0.178571, 0.178571])
        np.testing.assert_allclose(uniq_times, expected_uniq_times)
        #np.testing.assert_allclose(prob_survival, expected_prob_survival)

if __name__ == '__main__':
    unittest.main()