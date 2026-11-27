
from betting_tracker import BettingTracker



def make_state(hand, phase, bets, big_blind=20, last_action=None):
    return {
        "hand_number": hand,
        "phase": phase,
        "big_blind": big_blind,
        "players": [{"bet": bet} for bet in bets],
        "last_action": last_action
    }



def make_raise_action(player, target, amount):
    return {
        "type": "raise",
        "player": player,
        "target": target,
        "amount": amount
    }

def test_initial_blinds():
    tracker = BettingTracker()

    result = tracker.update(
        make_state(1, "PREFLOP", [10, 20])
    )

    assert result["current_bet"] == 20
    assert result["minimum_raise"] == 20



def test_full_raise():
    tracker = BettingTracker()

    tracker.update(make_state(1, "PREFLOP", [10, 20]))

    result = tracker.update(
        make_state(
            1,
            "PREFLOP",
            [60, 20],
            last_action=make_raise_action("demo-bot-1", 60, 50)
        )
    )

    assert result["minimum_raise"] == 40
    assert result["raise_info_reliable"] is True




def test_second_raise():
    tracker = BettingTracker()

    # Initial blinds: 10/20
    tracker.update(
        make_state(1, "PREFLOP", [10, 20])
    )

    # First full raise: 20 -> 60 (raise size 40)
    tracker.update(
        make_state(
            1,
            "PREFLOP",
            [60, 20],
            last_action=make_raise_action("demo-bot-1", 60, 50)
        )
    )

    # Second full raise: 60 -> 120 (raise size 60)
    result = tracker.update(
        make_state(
            1,
            "PREFLOP",
            [60, 120],
            last_action=make_raise_action("demo-bot-2", 120, 100)
        )
    )

    assert result["current_bet"] == 120
    assert result["minimum_raise"] == 60
    assert result["raise_info_reliable"] is True



def test_new_round_resets():
    tracker = BettingTracker()

    tracker.update(make_state(1, "PREFLOP", [10, 20]))
    tracker.update(make_state(1, "PREFLOP", [60, 20]))

    result = tracker.update(
        make_state(1, "FLOP", [0, 0])
    )

    assert result["minimum_raise"] == 20
    assert result["current_bet"] == 0



# A short all-in should not increase the minimum raise

def test_short_all_in():
    tracker = BettingTracker()

    # Initial blinds: 10/20
    tracker.update(
        make_state(1, "PREFLOP", [10, 20, 0])
    )

    # Full raise to 60
    tracker.update(
        make_state(
            1,
            "PREFLOP",
            [60, 20, 0],
            last_action=make_raise_action("demo-bot-1", 60, 50)
        )
    )

    # Short all-in raise to 80
    result = tracker.update(
        make_state(
            1,
            "PREFLOP",
            [60, 20, 80],
            last_action=make_raise_action("demo-bot-3", 80, 80)
        )
    )

    assert result["current_bet"] == 80
    assert result["minimum_raise"] == 40
    assert result["raise_info_reliable"] is True




def test_short_all_in_blocks_reraise():
    tracker = BettingTracker()

    # Initial blinds
    tracker.update(
        make_state(1, "PREFLOP", [10, 20, 0])
    )

    # Our bot raises to 60
    tracker.update(
        make_state(
            1,
            "PREFLOP",
            [60, 20, 0],
            last_action=make_raise_action("groupM-poker-bot", 60, 50)
        )
    )

    # Record our successfully accepted raise
    tracker.record_our_action("raise")

    # Opponent makes a short all-in raise to 80
    result = tracker.update(
        make_state(
            1,
            "PREFLOP",
            [60, 20, 80],
            last_action=make_raise_action("demo-bot-3", 80, 80)
        )
    )

    assert result["minimum_raise"] == 40
    assert result["can_raise"] is False
    assert tracker.acted_since_full_raise is True



def test_new_round_unlocks_raising():
    tracker = BettingTracker()

    # Initial blinds
    tracker.update(
        make_state(1, "PREFLOP", [10, 20, 0])
    )

    # Our bot raises to 60
    tracker.update(
        make_state(
            1,
            "PREFLOP",
            [60, 20, 0],
            last_action=make_raise_action("groupM-poker-bot", 60, 50)
        )
    )

    tracker.record_our_action("raise")

    # Opponent makes a short all-in to 80
    result = tracker.update(
        make_state(
            1,
            "PREFLOP",
            [60, 20, 80],
            last_action=make_raise_action("demo-bot-3", 80, 80)
        )
    )

    assert result["can_raise"] is False

    # The game moves to the flop
    result = tracker.update(
        make_state(1, "FLOP", [0, 0, 0])
    )

    assert result["can_raise"] is True
    assert tracker.acted_since_full_raise is False
    assert result["minimum_raise"] == 20


def test_missing_raise_information():
    tracker = BettingTracker()

    tracker.update(make_state(1, "PREFLOP", [10, 20]))

    # Bet increases, but no raise action is supplied
    result = tracker.update(
        make_state(1, "PREFLOP", [60, 20])
    )

    assert result["raise_info_reliable"] is False


def test_raise_not_yet_verified():
    tracker = BettingTracker()

    result = tracker.update(
        make_state(1, "PREFLOP", [10, 20])
    )

    assert result["raise_verified"] is False

def test_detect_raise_action():
    tracker = BettingTracker()

    state = make_state(1, "PREFLOP", [60, 20])
    state["last_action"] = {
        "type": "raise",
        "player": "demo-bot-1",
        "amount": 40,
        "target": 60
    }

    assert tracker.is_raise_action(state) is True


def test_detect_non_raise_action():
    tracker = BettingTracker()

    state = make_state(1, "PREFLOP", [10, 20])
    state["last_action"] = {
        "type": "deal"
    }

    assert tracker.is_raise_action(state) is False


def test_unexplained_bet_increase():
    tracker = BettingTracker()

    # Start with the blinds at 10/20
    tracker.update(make_state(1, "PREFLOP", [10, 20]))

    # Highest bet jumps to 100, but the latest action is a call
    state = make_state(1, "PREFLOP", [100, 100])
    state["last_action"] = {
        "type": "call",
        "player": "demo-bot-2",
        "amount": 80
    }

    result = tracker.update(state)

    # We cannot trust this betting information
    assert result["raise_info_reliable"] is False

    # Don't incorrectly change the minimum raise estimate
    assert result["minimum_raise"] == 20

    # Raising must remain unverified
    assert result["raise_verified"] is False


def test_record_confirmed_fold():
    tracker = BettingTracker()

    tracker.record_our_action("fold")

    assert tracker.acted_since_full_raise is True



def test_duplicate_raise_snapshot():
    tracker = BettingTracker()

    # Initial blinds
    tracker.update(make_state(1, "PREFLOP", [10, 20]))

    # Player raises from 20 to 60
    raise_state = make_state(
        1,
        "PREFLOP",
        [60, 20],
        last_action=make_raise_action("demo-bot-1", 60, 50)
    )

    first_result = tracker.update(raise_state)

    # Same snapshot arrives again
    second_result = tracker.update(raise_state)

    assert first_result["minimum_raise"] == 40
    assert second_result["minimum_raise"] == 40
    assert second_result["current_bet"] == 60
    assert second_result["raise_info_reliable"] is True

def test_missed_intermediate_raise():
    tracker = BettingTracker()

    # Initial blinds
    tracker.update(make_state(1, "PREFLOP", [10, 20, 0]))

    # We miss a raise to 60.
    # The latest action is a raise from 60 to 120.
    state = make_state(
        1,
        "PREFLOP",
        [60, 120, 0],
        last_action=make_raise_action("demo-bot-2", 120, 100)
    )

    result = tracker.update(state)

    # We should recognize that our information is incomplete
    assert result["raise_info_reliable"] is False

    # Do not trust a minimum calculated from the combined increase
    assert result["minimum_raise"] == 20

    # Raising must remain unverified
    assert result["raise_verified"] is False
