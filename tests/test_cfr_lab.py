import random
import unittest

from cfr_lab.core import Kuhn, Leduc, Solver, evaluate
from cfr_lab.river import experiment_action
from cfr_lab.run import river_fixture


class CFRTests(unittest.TestCase):
    def test_kuhn_value_and_exploitability(self):
        for sampled,steps in ((False,1600),(True,10000)):
            game = Kuhn()
            solver = Solver(13)
            solver.train(game,steps,sampled)
            result = evaluate(game,solver.average())
            self.assertAlmostEqual(result["value_p0"],-1/18,delta=.02)
            self.assertLess(result["exploitability"],.025)

    def test_leduc_cards_and_external_sampling(self):
        game = Leduc()
        self.assertAlmostEqual(sum(p for _,p in game.deals),1)
        self.assertEqual(game.info((0,0,1),(),0),game.info((0,1,2),(),0))
        self.assertNotEqual(game.info((0,0,1),("check","check"),0),
                            game.info((0,1,2),("check","check"),0))
        solver = Solver(13)
        solver.train(game,12000,external=True)
        self.assertLess(evaluate(game,solver.average())["exploitability"],.35)

    def test_river_uses_joint_blockers_and_chip_utilities(self):
        g = river_fixture(0)
        self.assertAlmostEqual(sum(p for _,p in g.deals),1)
        for _ in range(40):
            h0,h1 = g.sample(random.Random(_))
            self.assertEqual(len(set(g.board+h0+h1)),9)
        d = g.deals[0][0]
        b = g.openers[0]
        self.assertEqual(g.node(d,(b,"fold"))[2],g.pot/2)
        self.assertEqual(abs(g.node(d,(b,"call"))[2]),
                         0 if g.rank[d[0]]==g.rank[d[1]] else g.pot/2+b)
        solver = Solver(19)
        solver.train(g,1500,external=True)
        self.assertLess(evaluate(g,solver.average())["exploitability"],
                        evaluate(g,{})["exploitability"])

    def test_unsupported_state_falls_back(self):
        state = {"phase":"FLOP","players":[]}
        self.assertEqual(experiment_action(state,"hero",None,random.Random(1),
                                            lambda:("check",0)),("check",0))
