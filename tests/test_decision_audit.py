import unittest

from ev import _expectation, preflop_change


class PreflopPayoutTests(unittest.TestCase):
    def test_folded_chips_stay_in_the_correct_layer(self):
        chips = {"hero":20,"big":100,"folded":100}
        ranks = {"hero":(8,),"big":(1,)}
        self.assertEqual(preflop_change(chips,ranks,"hero",20,list(chips)),40)

    def test_unmatched_shove_is_returned(self):
        players = [{"id":"short","bet":20}]
        original = {"hero":10,"short":20}
        order = ["short","hero"]
        args = (30,"hero",players,[0],{"short":100},10,1000,(original,order))
        self.assertEqual(_expectation([{"hero":(8,),"short":(1,)}],*args),110)
        self.assertEqual(_expectation([{"hero":(1,),"short":(8,)}],*args),-90)

    def test_five_equal_callers(self):
        players = [{"id":str(i),"bet":0} for i in range(5)]
        ids = [p["id"] for p in players]
        original = {pid:0 for pid in ["hero",*ids]}
        ranks = {"hero":(8,),**{pid:(1,) for pid in ids}}
        finals = {pid:1000 for pid in ids}
        value = _expectation([ranks],0,"hero",players,[0]*5,finals,0,1000,
                             (original,list(original)))
        self.assertEqual(value,5000)
