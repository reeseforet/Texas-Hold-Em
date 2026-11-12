
from monte_carlo import simulate_hand

my_cards = [
    {"rank": 14, "suit": "spades"},
    {"rank": 13, "suit": "hearts"}
]

community_cards = []

# Simulate a game against three opponents
opponents, board = simulate_hand(my_cards, community_cards, 3)

print("Number of opponents:", len(opponents))
print("Cards per opponent:", [len(hand) for hand in opponents])
print("Community cards:", len(board))

# Make sure none of the dealt cards are repeated
all_cards = my_cards + board

for hand in opponents:
    all_cards.extend(hand)

unique_cards = {(card["rank"], card["suit"]) for card in all_cards}

print("No duplicate cards:", len(unique_cards) == len(all_cards))
