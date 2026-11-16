from copy import deepcopy
from fractions import Fraction as F
import json
import random
import subprocess
import sys
import tempfile
import unittest

from ergodlab import audit, audit_covariance
from ergodlab.demo import occupancy_model, periodic_model, sticky_model


def q(value):
    return F(*value)


def enumerate_cross(model, stationary, lag):
    """Enumerate edge paths independently of the production matrix recurrence."""
    n = len(stationary)
    paths = [(s, mass, None, None) for s, mass in enumerate(stationary)]
    for step in range(lag + 1):
        after = []
        for state, mass, first, last in paths:
            weights = model['weights'][state]
            for action, weight in enumerate(weights):
                for target in range(n):
                    chance = F(weight, sum(weights)) * q(model['transitions'][state][action][target])
                    if chance:
                        reward = q(model['rewards'][state][action][target])
                        after.append((target, mass * chance, reward if step == 0 else first, reward))
        paths = after
    return sum((mass * first * last for state, mass, first, last in paths), F(0))


class CovarianceTests(unittest.TestCase):
    def test_periodic_all_lags_and_blocks(self):
        result = audit_covariance(periodic_model(), max_lag=31)
        for row in result['reward_autocovariances']:
            self.assertEqual(q(row['covariance']), F((-1)**row['lag'], 4))
        for row in result['block_decompositions']:
            self.assertEqual(q(row['sum_variance']), F(row['length'] % 2, 4))
            self.assertEqual(row['moment_recurrence_residual'], [0, 1])
        self.assertEqual(result['stationary_audit'], audit(periodic_model(), block_limit=32))

    def test_sticky_closed_form(self):
        result = audit_covariance(sticky_model(), max_lag=31)
        for row in result['reward_autocovariances']:
            self.assertEqual(q(row['covariance']), F(1, 4) * F(3, 4)**row['lag'])

    def test_independent_edge_path_grid(self):
        rng = random.Random(497)
        for case in range(16):
            model = occupancy_model()
            for state in range(2):
                model['weights'][state] = [rng.randrange(1, 5), rng.randrange(1, 5)]
                for action in range(2):
                    switch = rng.randrange(1, 8)
                    model['transitions'][state][action] = [[8-switch, 8], [switch, 8]]
                    model['rewards'][state][action] = [[rng.randrange(-4, 5), 2] for _ in range(2)]
            p = sum(F(w, sum(model['weights'][0])) * q(model['transitions'][0][a][1]) for a, w in enumerate(model['weights'][0]))
            back = sum(F(w, sum(model['weights'][1])) * q(model['transitions'][1][a][0]) for a, w in enumerate(model['weights'][1]))
            stationary = [back/(p+back), p/(p+back)]
            result = audit_covariance(model, max_lag=3)
            for row in result['reward_autocovariances']:
                self.assertEqual(q(row['cross_moment']), enumerate_cross(model, stationary, row['lag']), (case, row['lag']))
                self.assertEqual(q(row['covariance']), q(row['cross_moment']) - q(result['stationary_audit']['average_reward'])**2)

    def test_edge_telescoping_reward(self):
        model = periodic_model()
        model['transitions'] = [[[[1, 2], [1, 2]]], [[[1, 2], [1, 2]]]]
        model['rewards'] = [[[[0, 1], [1, 1]]], [[[-1, 1], [0, 1]]]]
        result = audit_covariance(model, max_lag=8)
        self.assertEqual([q(row['covariance']) for row in result['reward_autocovariances']], [F(1, 2), -F(1, 4)] + [F(0)]*7)
        self.assertTrue(all(q(row['sum_variance']) == F(1, 2) for row in result['block_decompositions']))
        # A source-state reward shortcut has lag-one covariance zero here, not -1/4.

    def test_single_state_action_noise(self):
        model = {'weights': [[1, 3]], 'features': [[[0], [1]]],
                 'transitions': [[[[1, 1]], [[1, 1]]]], 'rewards': [[[[0, 1]], [[2, 1]]]]}
        result = audit_covariance(model, max_lag=5)
        self.assertEqual([q(row['covariance']) for row in result['reward_autocovariances']], [F(3, 4)] + [F(0)]*5)
        for row in result['block_decompositions']:
            self.assertEqual(q(row['sum_variance']), row['length']*F(3, 4))

    def test_three_cycle(self):
        model = {'weights': [[1]]*3, 'features': [[[0]]]*3,
                 'transitions': [[[[int(j == (s+1) % 3), 1] for j in range(3)]] for s in range(3)],
                 'rewards': [[[[s-1, 1] for j in range(3)]] for s in range(3)]}
        result = audit_covariance(model, max_lag=8)
        for row in result['reward_autocovariances']:
            self.assertEqual(q(row['covariance']), F(2, 3) if row['lag'] % 3 == 0 else -F(1, 3))
        self.assertEqual(q(result['block_decompositions'][2]['sum_variance']), 0)

    def test_shift_scale_and_isolation(self):
        model = occupancy_model()
        old = deepcopy(model)
        state = random.getstate()
        before = audit_covariance(model, max_lag=4)
        self.assertEqual(model, old)
        self.assertEqual(random.getstate(), state)
        for row in model['rewards']:
            for action in row:
                for value in action:
                    value[:] = [3*value[0]+2*value[1], value[1]]
        after = audit_covariance(model, max_lag=4)
        for a, b in zip(before['reward_autocovariances'], after['reward_autocovariances']):
            self.assertEqual(q(b['covariance']), 9*q(a['covariance']))
        before['stationary_audit']['model']['weights'][0][0] = 999
        self.assertEqual(old, occupancy_model())
        self.assertNotEqual(model['weights'][0][0], 999)

    def test_validation_and_zero_lag(self):
        for value in [True, False, -1, 32, 1.0, None, '2', []]:
            with self.assertRaises(ValueError):
                audit_covariance(occupancy_model(), max_lag=value)
        result = audit_covariance(occupancy_model(), max_lag=0)
        self.assertEqual(len(result['reward_autocovariances']), 1)
        self.assertEqual(result['block_decompositions'][0]['lag_contributions'], [])
        self.assertEqual(result['block_decompositions'][0]['sum_variance'], result['reward_autocovariances'][0]['covariance'])
        with self.assertRaises(ValueError):
            audit_covariance({})

    def test_cli_and_strict_payload(self):
        command = [sys.executable, '-m', 'ergodlab', 'covariance']
        run = subprocess.run(command + ['examples/occupancy.json', '--max-lag', '3'], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), audit_covariance(occupancy_model(), max_lag=3))
        for text in ['{"version":1,"version":1,"model":{}}', '{"version":true,"model":{}}', '{"version":1,"model":NaN}', '[]', 'x'*65537]:
            with tempfile.NamedTemporaryFile(mode='w') as handle:
                handle.write(text)
                handle.flush()
                run = subprocess.run(command + [handle.name], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertFalse(run.stdout)
        for value in ['-1', '32', '1.5']:
            run = subprocess.run(command + ['examples/occupancy.json', '--max-lag', value], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertFalse(run.stdout)
