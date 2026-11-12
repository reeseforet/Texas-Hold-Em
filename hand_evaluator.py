from itertools import combinations

# Rank the different poker hands from weakest to strongest
HAND_RANKS = {
    "high_card": 0,
    "one_pair": 1,
    "two_pair": 2,
    "three_of_a_kind": 3,
    "straight": 4,
    "flush": 5,
    "full_house": 6,
    "four_of_a_kind": 7,
    "straight_flush": 8
}


# Return the numerical ranking of a poker hand
def get_hand_rank(hand_type):
    return HAND_RANKS[hand_type]

# Count how many cards have each rank
def count_ranks(cards):
    counts = {}

    # Go through each card in our hand
    for card in cards:
        rank = card["rank"]

        # Add one to the count for this rank
        if rank in counts:
            counts[rank] += 1
        else:
            counts[rank] = 1

    return counts


# Identify poker hands based on matching card ranks
def check_matching_cards(cards):
    counts = count_ranks(cards)

    # Get the number of times each rank appears
    matches = list(counts.values())

    # Check for four cards of the same rank
    if 4 in matches:
        return "four_of_a_kind"

    # Check for three cards and two cards of another rank
    elif 3 in matches and 2 in matches:
        return "full_house"

    # Check for three cards of the same rank
    elif 3 in matches:
        return "three_of_a_kind"

    # Check if we have two different pairs
    elif matches.count(2) == 2:
        return "two_pair"

    # Check if we have one pair
    elif 2 in matches:
        return "one_pair"

    # If there are no matching ranks
    else:
        return "high_card"




# Check whether all five cards have the same suit
def check_flush(cards):
    # Get the suit of the first card
    first_suit = cards[0]["suit"]

    # Check if every card has the same suit
    for card in cards:
        if card["suit"] != first_suit:
            return False

    return True


# Check if five cards form a straight
def check_straight(cards):
    # Get the card ranks and sort them from lowest to highest
    ranks = sorted(card["rank"] for card in cards)

    # A straight cannot have repeated ranks
    if len(set(ranks)) != 5:
        return False

    # Check if the ranks are five consecutive numbers
    if ranks[-1] - ranks[0] == 4:
        return True

    # Special case: Ace can count as 1 in A-2-3-4-5
    if ranks == [2, 3, 4, 5, 14]:
        return True

    return False



# Determine the type of poker hand from five cards
def evaluate_hand(cards):
    # Check if the hand is a flush or straight
    is_flush = check_flush(cards)
    is_straight = check_straight(cards)

    # A straight flush is the strongest hand category
    if is_flush and is_straight:
        return "straight_flush"

    # Get the hand type based on matching card ranks
    matching_hand = check_matching_cards(cards)

    # Check for four of a kind or full house
    if matching_hand == "four_of_a_kind":
        return "four_of_a_kind"

    if matching_hand == "full_house":
        return "full_house"

    # A flush beats a straight and three of a kind
    if is_flush:
        return "flush"

    if is_straight:
        return "straight"

    # Otherwise, return the matching hand type
    return matching_hand





# Find the strongest five-card hand out of seven cards
def best_hand(cards):
    best_score = None

    # Try every possible combination of five cards
    for hand in combinations(cards, 5):
        score = hand_score(hand)

        # Keep the strongest complete hand score
        if best_score is None or score > best_score:
            best_score = score

    return best_score





# Compare two hands that have one pair
def compare_one_pair(cards):
    counts = count_ranks(cards)

    # Find the rank of the pair
    pair_rank = next(rank for rank, count in counts.items() if count == 2)

    # Find the other three cards and sort them highest first
    kickers = sorted(
        (rank for rank, count in counts.items() if count == 1),
        reverse=True
    )

    # Return the pair rank followed by the kickers
    return (pair_rank, *kickers)


# Return the hand category and cards needed to break ties
def hand_score(cards):
    hand_type = evaluate_hand(cards)
    counts = count_ranks(cards)

    # Sort card ranks from highest to lowest
    ranks = sorted((card["rank"] for card in cards), reverse=True)

    # High card: compare all five cards in order
    if hand_type == "high_card":
        return (0, *ranks)

    # One pair: compare pair rank, then kickers
    if hand_type == "one_pair":
        return (1, *compare_one_pair(cards))

    # Two pair: compare highest pair, lower pair, then kicker
    if hand_type == "two_pair":
        pairs = sorted(
            (rank for rank, count in counts.items() if count == 2),
            reverse=True
        )
        kicker = next(rank for rank, count in counts.items() if count == 1)
        return (2, *pairs, kicker)

    # Three of a kind: compare the three matching cards, then kickers
    if hand_type == "three_of_a_kind":
        three = next(rank for rank, count in counts.items() if count == 3)
        kickers = sorted(
            (rank for rank, count in counts.items() if count == 1),
            reverse=True
        )
        return (3, three, *kickers)

    # Straight: compare the highest card in the straight
    if hand_type == "straight":
        high = 5 if sorted(ranks) == [2, 3, 4, 5, 14] else max(ranks)
        return (4, high)

    # Flush: compare all five cards from highest to lowest
    if hand_type == "flush":
        return (5, *ranks)

    # Full house: compare three of a kind first, then the pair
    if hand_type == "full_house":
        three = next(rank for rank, count in counts.items() if count == 3)
        pair = next(rank for rank, count in counts.items() if count == 2)
        return (6, three, pair)

    # Four of a kind: compare four matching cards, then kicker
    if hand_type == "four_of_a_kind":
        four = next(rank for rank, count in counts.items() if count == 4)
        kicker = next(rank for rank, count in counts.items() if count == 1)
        return (7, four, kicker)

    # Straight flush: compare highest card
    if hand_type == "straight_flush":
        high = 5 if sorted(ranks) == [2, 3, 4, 5, 14] else max(ranks)
        return (8, high)


# Test our rank counting function
if __name__ == "__main__":
    test_cards = [
        {"rank": 14, "suit": "spades"},
        {"rank": 14, "suit": "hearts"},
        {"rank": 7, "suit": "clubs"},
        {"rank": 7, "suit": "diamonds"},
        {"rank": 13, "suit": "spades"}
    ]

    print(count_ranks(test_cards))
    print(check_matching_cards(test_cards))
        # Test a flush with five spades
    flush_cards = [
        {"rank": 14, "suit": "spades"},
        {"rank": 11, "suit": "spades"},
        {"rank": 9, "suit": "spades"},
        {"rank": 6, "suit": "spades"},
        {"rank": 3, "suit": "spades"}
    ]

    print("Flush test:", check_flush(flush_cards))
    print("Original hand flush:", check_flush(test_cards))

    # Test a regular straight
    straight_cards = [
        {"rank": 5, "suit": "spades"},
        {"rank": 6, "suit": "hearts"},
        {"rank": 7, "suit": "clubs"},
        {"rank": 8, "suit": "diamonds"},
        {"rank": 9, "suit": "spades"}
    ]

    # Test a straight where Ace counts as 1
    ace_low_cards = [
        {"rank": 14, "suit": "spades"},
        {"rank": 2, "suit": "hearts"},
        {"rank": 3, "suit": "clubs"},
        {"rank": 4, "suit": "diamonds"},
        {"rank": 5, "suit": "spades"}
    ]

    print("Straight test:", check_straight(straight_cards))
    print("Ace-low straight test:", check_straight(ace_low_cards))
    print("Original hand straight:", check_straight(test_cards))

    # Test our complete hand evaluation function
    print("Two pair evaluation:", evaluate_hand(test_cards))
    print("Flush evaluation:", evaluate_hand(flush_cards))
    print("Straight evaluation:", evaluate_hand(straight_cards))

    # Test a straight flush
    straight_flush_cards = [
        {"rank": 10, "suit": "hearts"},
        {"rank": 11, "suit": "hearts"},
        {"rank": 12, "suit": "hearts"},
        {"rank": 13, "suit": "hearts"},
        {"rank": 14, "suit": "hearts"}
    ]

    print("Straight flush evaluation:", evaluate_hand(straight_flush_cards))

    # Test finding the best hand out of seven cards
    seven_cards = [
        {"rank": 14, "suit": "spades"},
        {"rank": 14, "suit": "hearts"},
        {"rank": 13, "suit": "clubs"},
        {"rank": 13, "suit": "diamonds"},
        {"rank": 13, "suit": "spades"},
        {"rank": 7, "suit": "hearts"},
        {"rank": 2, "suit": "clubs"}
    ]

    print("Best hand score:", best_hand(seven_cards))   
    # Test comparing two different pairs
    pair_aces = [
        {"rank": 14, "suit": "spades"},
        {"rank": 14, "suit": "hearts"},
        {"rank": 13, "suit": "clubs"},
        {"rank": 9, "suit": "diamonds"},
        {"rank": 3, "suit": "spades"}
    ]

    pair_kings = [
        {"rank": 13, "suit": "spades"},
        {"rank": 13, "suit": "hearts"},
        {"rank": 14, "suit": "clubs"},
        {"rank": 9, "suit": "diamonds"},
        {"rank": 3, "suit": "spades"}
    ]

    print("Pair of Aces:", compare_one_pair(pair_aces))
    print("Pair of Kings:", compare_one_pair(pair_kings))
    print("Aces win:", compare_one_pair(pair_aces) > compare_one_pair(pair_kings))

    # Test scoring different poker hands
    print("Pair of Aces score:", hand_score(pair_aces))
    print("Pair of Kings score:", hand_score(pair_kings))
    print("Straight score:", hand_score(straight_cards))
    print("Flush score:", hand_score(flush_cards))

    # Check that a flush beats a straight
    print("Flush beats straight:",
          hand_score(flush_cards) > hand_score(straight_cards))

    # Check that a pair of Aces beats a pair of Kings
    print("Aces beat Kings:",
          hand_score(pair_aces) > hand_score(pair_kings))



