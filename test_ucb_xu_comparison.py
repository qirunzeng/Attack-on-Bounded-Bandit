"""Independent sequential checks of the common anytime UCB comparison."""
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

import bandit
import paper_runner
import ucb_xu_comparison as comparison
import xu_budget


def reference(clean, n, arrays, T, seed, phase1=0, phase2=0, intervene=False,
              exploration_parameter=1/3, fixed_terminal_clock=False):
    K = len(arrays)
    counts = np.zeros(K,dtype=int) if intervene else np.full(K,5)+n
    totals = np.zeros(K) if intervene else clean.copy()
    if not intervene:
        totals[-1] += n[-1]
    online, stages = np.zeros(K,dtype=int), np.zeros((3,K),dtype=int)
    start, modified, magnitude = int(counts.sum()), 0, 0.
    rng = np.random.default_rng(seed)
    for step in range(T-start):
        missing = np.flatnonzero(counts==0)
        time = T if fixed_terminal_clock else start+step+1
        arm = int(missing[0]) if len(missing) else int(np.argmax(
            totals/counts + 3*exploration_parameter*np.sqrt(np.log(time)/counts)))
        raw = arrays[arm][rng.integers(len(arrays[arm]))]
        stage = 0 if intervene and step<phase1 else 1 if intervene and step<phase1+phase2 else 2
        observed = 0. if stage==0 else float(arm==K-1) if stage==1 else raw
        modified += raw!=observed; magnitude += abs(raw-observed)
        counts[arm] += 1; totals[arm] += observed; online[arm] += 1; stages[stage,arm] += 1
    return online,modified,magnitude,*stages


class UCBXuComparisonTests(unittest.TestCase):
    arrays = [np.array([0.,1.,1.]),np.array([0.,.3,1.]),np.array([.2])]

    def test_both_attacks_match_independent_anytime_reference(self):
        for intervene,n in [(False,np.array([3,7,11])),(True,np.zeros(3,dtype=int))]:
            clean = np.zeros(3) if intervene else np.array([4.,3.,1.])
            for sigma in [1/3,.5]:
                for seed in [0,42,203]:
                    phases = dict(phase1=12,phase2=17) if intervene else {}
                    got = comparison.simulate(clean,n,self.arrays,300,seed,intervene=intervene,
                                               exploration_parameter=sigma,**phases)
                    expected = reference(clean,n,self.arrays,300,seed,intervene=intervene,
                                         exploration_parameter=sigma,**phases)
                    for a,b in zip(got,expected):
                        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)

    def test_index_initialization_and_full_internal_clock(self):
        for horizon in [1,2,3]:
            got = comparison.simulate(np.zeros(3),np.zeros(3,dtype=int),self.arrays,
                                      horizon,42,intervene=True)
            np.testing.assert_array_equal(got[0],[int(i<horizon) for i in range(3)])
        # The initial warm-start counts determine t; log(T) is a different learner.
        n=np.array([10,20]);clean=np.array([5.,1.]);arrays=[np.ones(1),np.array([.2])]
        got=comparison.simulate(clean,n,arrays,100,42)
        expected=reference(clean,n,arrays,100,42)
        fixed=reference(clean,n,arrays,100,42,fixed_terminal_clock=True)
        for a,b in zip(got,expected):np.testing.assert_allclose(a,b)
        self.assertFalse(np.array_equal(got[0],fixed[0]))
        self.assertEqual(int(got[0].sum()),60)

    def test_allocation_parameter_and_module_state_are_isolated(self):
        clean=(3.,2.,1.)
        previous={name:getattr(bandit,name) for name in
                  ['K','target_arm','sigma','delta','N0_i','r_l','r_u','R']}
        old_cache=paper_runner.allocation.cache_info()
        one=comparison.allocation(clean,100_000,.2,1/3)
        two=comparison.allocation(clean,100_000,.2,.5)
        self.assertNotEqual(one[0],two[0])
        self.assertEqual({name:getattr(bandit,name) for name in previous},previous)
        self.assertEqual(paper_runner.allocation.cache_info(),old_cache)
        with comparison.allocation_settings(3,1/3):
            self.assertEqual(bandit.sigma,1/3)
            bandit._check_ucb_cutoffs(np.array(clean)/5,100_000,np.array(one[0]),one[1],one[2])
        self.assertEqual({name:getattr(bandit,name) for name in previous},previous)
        with patch.object(bandit,'_ucb_counts_for_T',side_effect=RuntimeError('check restoration')):
            with self.assertRaisesRegex(RuntimeError,'check restoration'):
                comparison.allocation((3.,2.,1.),100_001,.2,1/3)
        self.assertEqual({name:getattr(bandit,name) for name in previous},previous)

    def test_zero_gap_infeasible_budget_and_fifty_repeat_grid(self):
        grid=comparison.configuration_grid()
        self.assertEqual(comparison.REPEATS,50)
        self.assertEqual(len(grid)*comparison.REPEATS*len(comparison.METHODS),1400)
        budgets=[xu_budget.budget('UCB',10,1_000_000,c['Delta_K']) for c in grid]
        self.assertIsNone(budgets[0]['total'])
        self.assertTrue(any(b['total'] is not None and not b['feasible'] for b in budgets))
        self.assertTrue(any(b['feasible'] for b in budgets))
        metadata=dict(selected_reward_arrays=self.arrays)
        config=dict(multiplier=0.,Delta_K=0.)
        ours,xu=comparison.run_case(metadata,0,config,100_000)
        self.assertEqual(ours['status'],'simulated')
        self.assertEqual(xu['status'],'infeasible')
        self.assertEqual(xu['cost'],'')
        for key in ['online_counts_json','target_ratio','online_ratio','post_attack_target_ratio','trajectory_seconds']:
            self.assertEqual(xu[key],'')

    def test_rows_record_actual_stages_and_each_ratio(self):
        metadata=dict(selected_reward_arrays=self.arrays)
        rows=comparison.run_case(metadata,2,dict(multiplier=1.,Delta_K=.2),100_000)
        for row in rows:
            self.assertEqual(row['status'],'simulated')
            online=np.array(json.loads(row['online_counts_json']))
            n=np.array(json.loads(row['allocation_json']))
            stages=np.array([json.loads(row[k]) for k in
                             ['phase1_counts_json','phase2_counts_json','post_attack_counts_json']])
            np.testing.assert_array_equal(stages.sum(axis=0),online)
            self.assertEqual(row['H']+row['T0'],row['T'])
            self.assertEqual(row['target_ratio'],(row['N0_i']+n[-1]+online[-1])/row['T'])
            self.assertEqual(row['online_ratio'],online[-1]/row['H'])
            self.assertEqual(row['post_attack_target_ratio'],stages[2,-1]/row['post_attack_H'])
            self.assertEqual(list(stages.sum(axis=1)),[row['C1'],row['C2'],row['post_attack_H']])
            if row['method']=='Xu2021':
                self.assertEqual(row['cost'],row['C1']+row['C2'])
                np.testing.assert_array_equal(stages[0],np.full(3,row['C1']//3))
                np.testing.assert_array_equal(stages[1],[1,1,row['C2']-2])
                np.testing.assert_array_equal(stages[2],[0,0,row['post_attack_H']])
            else:
                self.assertEqual(row['cost'],sum(n))


if __name__=='__main__':
    unittest.main()
