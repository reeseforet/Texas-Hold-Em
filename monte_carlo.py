
import random
from hand_evaluator import best_hand


# Create a standard deck of 52 playing cards
def create_deck():
    suits = ["hearts", "diamonds", "clubs", "spades"]
    deck = []

    # Add cards ranked 2 through Ace for each suit
    for suit in suits:
        for rank in range(2, 15):
            deck.append({
                "rank": rank,
                "suit": suit
            })

    return deck


# Remove cards we already know about from the deck
def remove_known_cards(deck, known_cards):
    remaining_deck = []

    # Go through every card in the deck
    for card in deck:
        # Check if this card matches one of our known cards
        if card not in known_cards:
            remaining_deck.append(card)

    return remaining_deck


# Simulate opponent hands and the remaining community cards
def simulate_hand(my_cards, community_cards, num_opponents=1):
    # Create a deck without our cards or known community cards
    deck = create_deck()
    known_cards = my_cards + community_cards
    remaining_deck = remove_known_cards(deck, known_cards)

    # Figure out how many community cards are still needed
    missing_cards = 5 - len(community_cards)

    # Each opponent needs two cards
    opponent_card_count = num_opponents * 2
    total_cards = opponent_card_count + missing_cards

    # Draw all the cards at once to prevent duplicates
    drawn_cards = random.sample(remaining_deck, total_cards)

    # Give each opponent two cards
    opponent_hands = []

    for i in range(num_opponents):
        start = i * 2
        opponent_hands.append(drawn_cards[start:start + 2])

    # Use the remaining drawn cards to finish the community cards
    final_community = community_cards + drawn_cards[opponent_card_count:]

    return opponent_hands, final_community


# Simulate one game against multiple opponents
def simulate_game(my_cards, community_cards, num_opponents=1):
    # Deal cards to opponents and complete the community cards
    opponent_hands, final_community = simulate_hand(
        my_cards, community_cards, num_opponents
    )

    # Find our bot's best possible hand
    my_hand = my_cards + final_community
    my_score = best_hand(my_hand)

    # Find the best possible hand for each opponent
    opponent_scores = []

    for opponent in opponent_hands:
        opponent_hand = opponent + final_community
        opponent_scores.append(best_hand(opponent_hand))

    # Find the strongest opponent's hand
    highest_opponent_score = max(opponent_scores)

    # Compare our hand against the strongest opponent
    if my_score > highest_opponent_score:
        return "win"
    elif my_score < highest_opponent_score:
        return "loss"
    else:
        return "tie"


# Simulate a game and calculate our share of the pot
def simulate_equity(my_cards, community_cards, num_opponents=1):
    # Deal cards to opponents and finish the community cards
    opponent_hands, final_community = simulate_hand(
        my_cards, community_cards, num_opponents
    )

    # Find our best possible poker hand
    my_score = best_hand(my_cards + final_community)

    # Find the best possible hand for each opponent
    opponent_scores = []

    for opponent in opponent_hands:
        score = best_hand(opponent + final_community)
        opponent_scores.append(score)

    # If anyone has a better hand, we receive nothing
    if any(score > my_score for score in opponent_scores):
        return 0.0

    # Count how many opponents tie with our hand
    tied_opponents = sum(
        score == my_score for score in opponent_scores
    )

    # Split the pot equally between all tied winners
    return 1.0 / (tied_opponents + 1)


# Estimate our average equity across multiple simulated games
def monte_carlo_equity(my_cards, community_cards, simulations=1000,
                       num_opponents=1):
    total_equity = 0

    # Run the requested number of simulations
    for _ in range(simulations):
        equity = simulate_equity(
            my_cards, community_cards, num_opponents
        )

        # Add our share of the pot from each simulated game
        total_equity += equity

    # Calculate our average expected share of the pot
    return total_equity / simulations


# Estimate our chances of winning by simulating multiple games
def monte_carlo_simulation(my_cards, community_cards, simulations=1000,
                           num_opponents=1):
    wins = 0
    losses = 0
    ties = 0

    # Run the requested number of simulated games
    for _ in range(simulations):
        result = simulate_game(my_cards, community_cards, num_opponents)

        # Count the outcome of each simulation
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        else:
            ties += 1

    # Calculate the percentage of games we won
    win_probability = wins / simulations

    return win_probability, wins, losses, ties


# Test our simulation functions
if __name__ == "__main__":
    # Test creating a full deck
    deck = create_deck()
    print("Cards in deck:", len(deck))
    print("First card:", deck[0])
    print("Last card:", deck[-1])

    # Our bot's two cards
    my_cards = [
        {"rank": 14, "suit": "spades"},
        {"rank": 13, "suit": "hearts"}
    ]

    # Test removing known cards
    remaining_deck = remove_known_cards(deck, my_cards)

    print("Cards remaining:", len(remaining_deck))
    print("Ace of spades removed:", my_cards[0] not in remaining_deck)
    print("King of hearts removed:", my_cards[1] not in remaining_deck)

    # Community cards already dealt
    community_cards = [
        {"rank": 12, "suit": "clubs"},
        {"rank": 11, "suit": "diamonds"},
        {"rank": 2, "suit": "spades"}
    ]

    # Test dealing cards to three opponents
    opponent_hands, final_community = simulate_hand(
        my_cards, community_cards, num_opponents=3
    )

    print("Number of opponents:", len(opponent_hands))
    print("Cards per opponent:", [len(hand) for hand in opponent_hands])
    print("Community has five cards:", len(final_community) == 5)

    # Make sure there are no duplicate cards
    all_cards = my_cards + final_community

    for hand in opponent_hands:
        all_cards.extend(hand)

    unique_cards = {(card["rank"], card["suit"]) for card in all_cards}

    print("No duplicate cards:", len(unique_cards) == len(all_cards))

    # Test determining the winner against three opponents
    result = simulate_game(my_cards, community_cards, num_opponents=3)

    print("Multiplayer simulation result:", result)
    print("Valid multiplayer result:", result in ["win", "loss", "tie"])

   
    # Compare our winning chances against different numbers of opponents
    print("\nMonte Carlo Comparison")

    for num_opponents in [1, 2, 3, 5]:
        probability, wins, losses, ties = monte_carlo_simulation(
            my_cards,
            community_cards,
            simulations=1000,
            num_opponents=num_opponents
        )

        print("-----------------------")
        print("Opponents:", num_opponents)
        print("Wins:", wins)
        print("Losses:", losses)
        print("Ties:", ties)
        print("Win probability:", round(probability * 100, 2), "%")

    # Test calculating equity against three opponents
    equity_result = simulate_equity(
        my_cards, community_cards, num_opponents=3
    )

    print("Single game equity:", equity_result)
    print("Valid equity:", 0.0 <= equity_result <= 1.0)

    # Test average equity against three opponents
    average_equity = monte_carlo_equity(
        my_cards, community_cards,
        simulations=1000,
        num_opponents=3
    )

    print("Average equity:", round(average_equity * 100, 2), "%")
    print("Valid average equity:", 0.0 <= average_equity <= 1.0)


