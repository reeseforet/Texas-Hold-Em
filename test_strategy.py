from strategy import decide_action, choose_strategy, calculate_raise_amount, should_raise
from unittest.mock import patch
from strategy import decide_action, choose_strategy

# Test checking when there is no bet to call
def test_check():
    assert decide_action(100, 0, 0.10) == "check"


# Test calling when the expected value is positive
def test_positive_ev_call():
    assert decide_action(100, 20, 0.60) == "call"


# Test folding when the expected value is negative
def test_negative_ev_fold():
    assert decide_action(100, 50, 0.10) == "fold"


# Test folding when calling has exactly zero expected value
def test_zero_ev():
    assert decide_action(100, 50, 1 / 3) == "fold"


# Test checking even with very low equity
def test_free_check():
    assert decide_action(60, 0, 0.01) == "check"



# Test that the strategy calls when Monte Carlo estimates high equity
def test_strategy_positive_ev():
    # Replace Monte Carlo's result with a fixed 60% equity
    with patch("strategy.monte_carlo_equity", return_value=0.60):
        action = choose_strategy(
            my_cards=[],
            community_cards=[],
            pot=100,
            to_call=20,
            num_opponents=3
        )

    assert action == "call"


# Test that the strategy folds when Monte Carlo estimates low equity
def test_strategy_negative_ev():
    # Replace Monte Carlo's result with a fixed 10% equity
    with patch("strategy.monte_carlo_equity", return_value=0.10):
        action = choose_strategy(
            my_cards=[],
            community_cards=[],
            pot=100,
            to_call=50,
            num_opponents=3
        )

    assert action == "fold"


# Test that the strategy checks when there is no bet to call
def test_strategy_free_check():
    # Even with low equity, checking costs nothing
    with patch("strategy.monte_carlo_equity", return_value=0.01):
        action = choose_strategy(
            my_cards=[],
            community_cards=[],
            pot=60,
            to_call=0,
            num_opponents=3
        )

    assert action == "check"


# Test a normal raise when we have enough chips
def test_normal_raise():
    amount = calculate_raise_amount(
        my_bet=0,
        my_stack=1000,
        highest_bet=20,
        big_blind=20
    )

    assert amount == 60


# Test an all-in raise when our stack is limited
def test_all_in_raise():
    amount = calculate_raise_amount(
        my_bet=20,
        my_stack=45,
        highest_bet=40,
        big_blind=20
    )

    assert amount == 65


# Test when we don't have enough chips to raise
def test_cannot_raise():
    amount = calculate_raise_amount(
        my_bet=20,
        my_stack=10,
        highest_bet=40,
        big_blind=20
    )

    assert amount is None


# Test that the raise never exceeds our available chips
def test_raise_stack_limit():
    amount = calculate_raise_amount(
        my_bet=10,
        my_stack=100,
        highest_bet=80,
        big_blind=20
    )

    assert amount <= 110


# Test using a larger minimum raise
def test_custom_minimum_raise():
    amount = calculate_raise_amount(
        my_bet=20,
        my_stack=1000,
        highest_bet=100,
        big_blind=20,
        minimum_raise=60
    )

    assert amount == 220


# Test raising with a very strong hand
def test_strong_hand_raise():
    assert should_raise(0.90) is True


# Test not raising with a weaker hand
def test_weak_hand_no_raise():
    assert should_raise(0.40) is False


# Test raising exactly at our threshold
def test_raise_threshold():
    assert should_raise(0.75) is True


# Test a custom raising threshold
def test_custom_raise_threshold():
    assert should_raise(0.65, threshold=0.60) is True



# Test that raising is blocked when the rules don't allow it
def test_raise_not_allowed():
    amount = calculate_raise_amount(
        my_bet=20,
        my_stack=1000,
        highest_bet=100,
        big_blind=20,
        minimum_raise=60,
        can_raise=False
    )

    assert amount is None
