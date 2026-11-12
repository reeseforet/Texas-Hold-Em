
# Import the functions we want to test
from hand_evaluator import evaluate_hand, hand_score, best_hand


# Make a card using its rank and suit
def card(rank, suit):
    return {"rank": rank, "suit": suit}


# Test that each poker hand is identified correctly
def test_hand_categories():
    hands = {
        "high_card": [card(14, "spades"), card(11, "hearts"),
                      card(9, "clubs"), card(6, "diamonds"),
                      card(3, "spades")],

        "one_pair": [card(14, "spades"), card(14, "hearts"),
                     card(9, "clubs"), card(6, "diamonds"),
                     card(3, "spades")],

        "two_pair": [card(14, "spades"), card(14, "hearts"),
                     card(9, "clubs"), card(9, "diamonds"),
                     card(3, "spades")],

        "three_of_a_kind": [card(14, "spades"), card(14, "hearts"),
                            card(14, "clubs"), card(9, "diamonds"),
                            card(3, "spades")],

        "straight": [card(5, "spades"), card(6, "hearts"),
                     card(7, "clubs"), card(8, "diamonds"),
                     card(9, "spades")],

        "flush": [card(14, "spades"), card(11, "spades"),
                  card(9, "spades"), card(6, "spades"),
                  card(3, "spades")],

        "full_house": [card(13, "spades"), card(13, "hearts"),
                       card(13, "clubs"), card(7, "diamonds"),
                       card(7, "spades")],

        "four_of_a_kind": [card(13, "spades"), card(13, "hearts"),
                           card(13, "clubs"), card(13, "diamonds"),
                           card(7, "spades")],

        "straight_flush": [card(10, "hearts"), card(11, "hearts"),
                           card(12, "hearts"), card(13, "hearts"),
                           card(14, "hearts")]
    }

    for expected, cards in hands.items():
        actual = evaluate_hand(cards)
        assert actual == expected, f"Expected {expected}, got {actual}"


# Test an Ace-low straight
def test_ace_low_straight():
    cards = [card(14, "spades"), card(2, "hearts"),
             card(3, "clubs"), card(4, "diamonds"),
             card(5, "spades")]

    assert hand_score(cards) == (4, 5)


# Test comparing hands with different kickers
def test_kickers():
    aces_with_king = [card(14, "spades"), card(14, "hearts"),
                      card(13, "clubs"), card(9, "diamonds"),
                      card(3, "spades")]

    aces_with_queen = [card(14, "spades"), card(14, "hearts"),
                       card(12, "clubs"), card(9, "diamonds"),
                       card(3, "spades")]

    assert hand_score(aces_with_king) > hand_score(aces_with_queen)


# Test that identical hand strengths result in a tie
def test_tie():
    hand1 = [card(14, "spades"), card(14, "hearts"),
             card(13, "clubs"), card(9, "diamonds"),
             card(3, "spades")]

    hand2 = [card(14, "clubs"), card(14, "diamonds"),
             card(13, "hearts"), card(9, "spades"),
             card(3, "clubs")]

    assert hand_score(hand1) == hand_score(hand2)


# Test selecting the best five cards from seven
def test_best_hand():
    cards = [card(14, "spades"), card(14, "hearts"),
             card(13, "clubs"), card(13, "diamonds"),
             card(13, "spades"), card(7, "hearts"),
             card(2, "clubs")]

    assert best_hand(cards) == (6, 13, 14)
