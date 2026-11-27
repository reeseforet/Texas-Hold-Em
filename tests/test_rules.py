"""Rules, range updates, and a short official-engine match."""

import sys
import unittest
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1].parent / "ai-poker-server"
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine import PokerEngine

from ev import pot_share
from ev import fold_product, showdown_change, sized_fold_probability
from match import AdaptiveBot, ThresholdBot, _build_deck, load_params, play_match
from opponent_model import FOLD_A, FOLD_B, RangeModel, hand_class


class EngineTests(unittest.TestCase):
    def test_chips_are_conserved_across_hands(self):
        def actor_factory(player_id):
            params = load_params("aggressor")
            return lambda: ThresholdBot(player_id, params, trials=8)

        names = ["A", "B", "C", "D"]
        deltas = play_match([(name, actor_factory(name)) for name in names], hands=6, seed=3)
        self.assertEqual(sum(deltas.values()), 0)

    def test_illegal_raise_is_rejected_and_side_pot_pays_the_right_players(self):
        engine = PokerEngine(starting_stack=1000, small_blind=10, big_blind=20)
        for name in ("A", "B", "C"):
            engine.add_player(name, name)
        engine.start_hand()
        # Whoever is not the blind posts a legal min-raise, then a too-small raise fails.
        actor = engine.current_turn
        short = engine.place_action(actor, "raise", engine.current_bet + 1)
        self.assertFalse(short["ok"])
        legal = engine.place_action(actor, "raise", engine.current_bet + engine.minimum_raise)
        self.assertTrue(legal["ok"])
        # Finish by folding everyone else so the pot is awarded and chips still add up.
        guard = 0
        while engine.phase not in ("WAITING", "SHOWDOWN") and guard < 30:
            guard += 1
            seat = engine.current_turn
            state = engine.snapshot(seat)
            me = next(player for player in state["players"] if player["id"] == seat)
            if state["current_bet"] > me["bet"]:
                engine.place_action(seat, "fold", 0)
            else:
                engine.place_action(seat, "check", 0)
        if engine.phase == "SHOWDOWN":
            engine.finalize_showdown()
        self.assertEqual(sum(player.stack for player in engine.players.values()), 3000)

    def test_all_in_side_pot_does_not_create_chips(self):
        engine = PokerEngine(starting_stack=100, small_blind=10, big_blind=20)
        for name in ("A", "B", "C"):
            engine.add_player(name, name)
        engine.players["A"].stack = 40
        engine.players["B"].stack = 80
        engine.players["C"].stack = 200
        engine.start_hand()
        guard = 0
        while engine.phase not in ("WAITING", "SHOWDOWN") and guard < 40:
            guard += 1
            seat = engine.current_turn
            state = engine.snapshot(seat)
            me = next(player for player in state["players"] if player["id"] == seat)
            ceiling = me["bet"] + me["stack"]
            target = state["current_bet"] + state["minimum_raise"]
            if ceiling > state["current_bet"] and me["stack"] > 0 and target <= ceiling:
                result = engine.place_action(seat, "raise", ceiling)
                if not result["ok"]:
                    engine.place_action(seat, "call", 0)
            elif state["current_bet"] > me["bet"]:
                engine.place_action(seat, "call", 0)
            else:
                engine.place_action(seat, "check", 0)
        if engine.phase == "SHOWDOWN":
            engine.finalize_showdown()
        self.assertEqual(sum(player.stack for player in engine.players.values()), 320)


class RangeTests(unittest.TestCase):
    def test_repeated_raises_make_aces_more_likely_than_junk(self):
        model = RangeModel()
        before_aces = model.combo_weight("villain", (14, "s"), (14, "h"))
        before_junk = model.combo_weight("villain", (7, "c"), (2, "d"))
        self.assertGreater(before_aces, before_junk)
        for _ in range(10):
            model.update_action("villain", "raise", True)
        aces = model.combo_weight("villain", (14, "s"), (14, "h"))
        junk = model.combo_weight("villain", (7, "c"), (2, "d"))
        self.assertGreater(aces / junk, before_aces / before_junk)
        self.assertLess(max(model.class_probabilities("villain").values()), 0.08)

    def test_fold_to_bet_ignores_checks(self):
        model = RangeModel()
        for _ in range(6):
            model.update_action("villain", "check", False)
        fold_rate_before = model.fold_to_bet("villain")
        for _ in range(6):
            model.update_action("villain", "fold", True)
        self.assertGreater(model.fold_to_bet("villain"), fold_rate_before)

    def test_a_raise_facing_a_bet_counts_as_not_folding(self):
        model = RangeModel()
        model.update_action("villain", "raise", True)
        model.update_action("villain", "fold", True)
        model.update_action("villain", "call", True)
        # Prior folds plus one fold, and prior continues plus one call and one raise.
        self.assertAlmostEqual(model.fold_to_bet("villain"), (FOLD_A + 1) / (FOLD_A + 1 + FOLD_B + 2))

    def test_an_opening_bet_is_not_a_fold_opportunity(self):
        model = RangeModel()
        before = model.fold_to_bet("villain")
        model.update_action("villain", "raise", False)
        self.assertAlmostEqual(model.fold_to_bet("villain"), before)

    def test_class_probabilities_sum_to_one(self):
        model = RangeModel()
        for _ in range(6):
            model.update_action("villain", "raise", True)
        total = sum(model.class_probabilities("villain").values())
        self.assertAlmostEqual(total, 1.0)

    def test_duplicate_state_is_counted_once(self):
        model = RangeModel()
        state = {
            "hand_number": 3,
            "phase": "PREFLOP",
            "current_bet": 60,
            "players": [],
            "last_action": {"type": "raise", "player": "villain", "amount": 40, "target": 60},
        }
        model.observe_state({"hand_number": 3, "phase": "PREFLOP", "current_bet": 20, "players": [], "last_action": {"type": "deal"}}, "hero")
        model.observe_state(state, "hero")
        model.observe_state(state, "hero")
        self.assertEqual(model._player("villain")["actions"], 1)

    def test_showdown_class_is_recorded(self):
        model = RangeModel()
        model.reveal("villain", [(14, "s"), (13, "s")])
        self.assertEqual(model._player("villain")["shown"][hand_class((14, "s"), (13, "s"))], 1)


class EquityTests(unittest.TestCase):
    def test_aces_are_ahead_of_one_random_hand(self):
        share, trials = pot_share(([(14, "s"), (14, "h")], []), ["random"], None, trials=80)
        self.assertGreater(trials, 20)
        self.assertGreater(share, 0.75)


class ValueTests(unittest.TestCase):
    def test_multiway_fold_chance_is_a_product(self):
        model = RangeModel()
        ids = ["a", "b", "c", "d"]
        for player_id in ids:
            for _ in range(8):
                model.update_action(player_id, "fold", True)
        rate = sized_fold_probability(model.fold_to_bet("a"), 1.0)
        self.assertAlmostEqual(fold_product(model, ids, 1.0), rate ** 4)
        stranger = fold_product(model, ["new"], 1.0)
        self.assertGreater(stranger, 0.05)
        self.assertLess(stranger, 0.50)

    def test_showdown_pays_a_short_call_and_returns_an_unmatched_raise(self):
        hero_wins = {"hero": (8, 14), "short": (1, 2)}
        # Pot 100 is already in. Each player adds 40. Hero wins all of it.
        self.assertAlmostEqual(
            showdown_change(100, "hero", 0, {"hero": 40, "short": 40}, hero_wins, 40),
            140,
        )
        hero_loses = {"hero": (1, 2), "short": (8, 14)}
        self.assertAlmostEqual(
            showdown_change(100, "hero", 0, {"hero": 40, "short": 40}, hero_loses, 40),
            -40,
        )
        # Raise is returned. Hero's 20 was already in a pot of 90, and nobody calls.
        self.assertAlmostEqual(
            showdown_change(90, "hero", 20, {"hero": 20}, {"hero": (8, 14)}, 0),
            90,
        )

    def test_population_fold_rate_is_between_a_call_and_a_fold(self):
        model = RangeModel()
        self.assertAlmostEqual(model.fold_to_bet("stranger"), 0.27)
        self.assertLess(sized_fold_probability(0.27, 8), 0.63)
        self.assertGreater(sized_fold_probability(0.27, 0.2), 0.05)


class DecisionTests(unittest.TestCase):
    def _state(self, hero_cards, pot, current, hero_bet, hero_stack, opponent_stack, board):
        return {
            "phase": "FLOP" if board else "PREFLOP",
            "pot": pot,
            "current_bet": current,
            "minimum_raise": 20,
            "big_blind": 20,
            "community_cards": board,
            "players": [
                {"id": "hero", "bet": hero_bet, "stack": hero_stack, "folded": False, "hole_cards": hero_cards},
                {"id": "villain", "bet": current, "stack": opponent_stack, "folded": False, "hole_cards": []},
            ],
        }

    def test_aces_call_a_small_bet_and_junk_folds_a_large_one(self):
        import random
        from ev import choose
        aces = [
            {"rank": 14, "suit": "spades", "label": "AS"},
            {"rank": 14, "suit": "hearts", "label": "AH"},
        ]
        junk = [
            {"rank": 7, "suit": "clubs", "label": "7C"},
            {"rank": 2, "suit": "diamonds", "label": "2D"},
        ]
        call = self._state(aces, 200, 20, 0, 900, 900, [])
        action, _amount, _info = choose(call, RangeModel(), "hero", trials=80, rng=random.Random(1))
        self.assertNotEqual(action, "fold")
        fold = self._state(junk, 40, 200, 0, 900, 900, [])
        action, _amount, _info = choose(fold, RangeModel(), "hero", trials=80, rng=random.Random(1))
        self.assertEqual(action, "fold")

    def test_air_does_not_shove_and_the_all_in_is_worse_than_a_check(self):
        import random
        from ev import choose
        junk = [
            {"rank": 7, "suit": "clubs", "label": "7C"},
            {"rank": 2, "suit": "diamonds", "label": "2D"},
        ]
        board = [
            {"rank": 14, "suit": "hearts", "label": "AH"},
            {"rank": 13, "suit": "spades", "label": "KS"},
            {"rank": 8, "suit": "diamonds", "label": "8D"},
        ]
        state = self._state(junk, 80, 0, 0, 900, 900, board)
        state["players"][1]["bet"] = 0
        action, amount, info = choose(state, RangeModel(), "hero", trials=120, rng=random.Random(4))
        self.assertNotEqual((action, amount), ("raise", 900))
        by_amount = {chips: ev for _name, chips, ev in info["options"]}
        self.assertLess(by_amount[900], by_amount[0])

    def test_same_seed_deals_the_same_deck(self):
        import random
        first = [str(card) for card in _build_deck(random.Random(4))]
        second = [str(card) for card in _build_deck(random.Random(4))]
        self.assertEqual(first, second)

    def test_a_chosen_raise_is_legal(self):
        engine = PokerEngine()
        engine.add_player("hero", "hero")
        engine.add_player("villain", "villain")
        engine.start_hand()
        seat = engine.current_turn
        state = engine.snapshot(seat)
        action, amount = AdaptiveBot(seat, trials=20)(state)
        result = engine.place_action(seat, action, int(amount))
        self.assertTrue(result["ok"], result)


class PrivacyTests(unittest.TestCase):
    def test_snapshot_hides_opponent_cards_and_the_bot_still_acts(self):
        import sys
        from pathlib import Path
        server = Path(__file__).resolve().parents[1].parent / "ai-poker-server"
        sys.path.insert(0, str(server))
        from app.engine import PokerEngine
        from match import AdaptiveBot

        engine = PokerEngine()
        engine.add_player("Hero", "Hero")
        engine.add_player("Villain", "Villain")
        engine.start_hand()
        seat = engine.current_turn
        state = engine.snapshot(seat)
        for player in state["players"]:
            if player["id"] == seat:
                continue
            self.assertTrue(all(card["label"] == "??" for card in player["hole_cards"]))
        action, amount = AdaptiveBot(seat, trials=10)(state)
        self.assertIn(action, {"fold", "check", "call", "raise"})
        self.assertIsInstance(amount, int)


class MatchTests(unittest.TestCase):
    def test_adaptive_match_conserves_chips(self):
        builders = []
        for name in ("Adaptive", "Aggro", "Rock"):
            if name == "Adaptive":
                builders.append((name, lambda: AdaptiveBot("Adaptive", trials=20)))
            else:
                params = load_params("aggressor" if name == "Aggro" else "evolved")
                builders.append((name, lambda name=name, params=params: ThresholdBot(name, params, trials=6)))
        deltas = play_match(builders, hands=2, seed=1)
        self.assertEqual(sum(deltas.values()), 0)


if __name__ == "__main__":
    unittest.main()
