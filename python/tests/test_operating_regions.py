import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_suite import classify


class OperatingRegionTests(unittest.TestCase):
    def test_a_working_simple_receiver_prevents_nn_only_claim(self):
        rows = [dict(method='Before EQ', ber_lower_95=.0001, ber_upper_95=.001),
                dict(method='Volterra 5', ber_lower_95=.01, ber_upper_95=.02),
                dict(method='NN multilevel', ber_lower_95=.0001, ber_upper_95=.001)]
        self.assertEqual(classify(rows)[0], 'non-neural receiver meets reference')

    def test_uncertainty_is_not_treated_as_baseline_failure(self):
        rows = [dict(method='FFE', ber_lower_95=.002, ber_upper_95=.006),
                dict(method='NN sigmoid', ber_lower_95=.0001, ber_upper_95=.001)]
        self.assertEqual(classify(rows)[0], 'NN meets; other receiver bounds inconclusive')
        rows[0]['ber_lower_95'] = .004
        self.assertEqual(classify(rows)[0], 'NN-only candidate in tested models')
