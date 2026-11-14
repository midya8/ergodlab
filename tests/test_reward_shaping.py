"""Exact telescoping identities, not a second resolvent implementation."""
from copy import deepcopy
from fractions import Fraction as F
import unittest

from ergodlab import audit, audit_discounted
from ergodlab.demo import periodic_model
from ergodlab.discount_demo import curl_model


def fraction(pair):
    return F(*pair)


def shaped(model, beta, potential):
    result = deepcopy(model)
    for state, actions in enumerate(result['rewards']):
        for rewards in actions:
            for target, reward in enumerate(rewards):
                value = fraction(reward) + beta * potential[target] - potential[state]
                rewards[target] = [value.numerator, value.denominator]
    return result


class RewardShaping(unittest.TestCase):
    def test_discounted_telescoping_preserves_true_gradient_not_proxy(self):
        potential = [F(-1, 2), F(3, 4)]
        changed_proxy = False
        for model in [curl_model(), periodic_model()]:
            for beta in [F(0), F(1, 2), F(3, 4), F(1)]:
                pair = [[beta.numerator, beta.denominator]]
                transformed = shaped(model, beta, potential)
                for start in range(2):
                    before = audit_discounted(model, betas=pair, initial_state=start)
                    after = audit_discounted(transformed, betas=pair, initial_state=start)
                    old, new = before['discounts'][0], after['discounts'][0]
                    self.assertEqual(new['discounted_occupancy'], old['discounted_occupancy'])
                    # Sum beta**t * (beta*Phi(S[t+1])-Phi(S[t]))
                    # telescopes to -Phi(S[0]); the normalized objective
                    # multiplies this by 1-beta. The limit at one is zero.
                    self.assertEqual(fraction(new['fixed_start_normalized_objective']),
                        fraction(old['fixed_start_normalized_objective']) - (1-beta)*potential[start])
                    self.assertEqual(new['fixed_start_objective_gradient'],
                                     old['fixed_start_objective_gradient'])
                    base = before['stationary_audit']
                    mu = list(map(fraction, base['stationary']))
                    mean_potential = sum(x*y for x, y in zip(mu, potential))
                    # A shared offset in both reports would evade the
                    # transformation identity; check the gauge separately.
                    for report in [old, new]:
                        self.assertEqual(sum(weight*fraction(value) for weight, value
                            in zip(mu, report['centered_resolvent'])), 0)
                    self.assertEqual(list(map(fraction, new['centered_resolvent'])),
                        [fraction(value)-phi+mean_potential
                         for value, phi in zip(old['centered_resolvent'], potential)])
                    correction = [(beta-1)*sum(mu[s]*fraction(matrix[s][t])*potential[t]
                        for s in range(2) for t in range(2))
                        for matrix in base['transition_derivatives']]
                    self.assertEqual(list(map(fraction, new['stationary_gradient_proxy'])),
                        [fraction(value)+delta for value, delta in
                         zip(old['stationary_gradient_proxy'], correction)])
                    changed_proxy |= new['stationary_gradient_proxy'] != old['stationary_gradient_proxy']
        self.assertTrue(changed_proxy, 'the finite-beta proxy is not shaping-invariant')

    def test_average_telescoping_preserves_variance_and_gradient(self):
        potential = [F(-1, 2), F(3, 4)]
        for model in [curl_model(), periodic_model()]:
            before = audit(model, block_limit=4)
            after = audit(shaped(model, F(1), potential), block_limit=4)
            self.assertEqual(after['average_reward'], before['average_reward'])
            self.assertEqual(after['gradient']['total'], before['gradient']['total'])
            # The accumulated reward changes by a bounded endpoint term.
            # Its asymptotic variance per step is unchanged, unlike the
            # finite-block variance, which is intentionally not equated.
            self.assertEqual(after['variance']['asymptotic_per_step'],
                             before['variance']['asymptotic_per_step'])


if __name__ == '__main__':
    unittest.main()
