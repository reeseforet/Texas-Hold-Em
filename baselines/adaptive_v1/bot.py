import asyncio
import json
import websockets
import random
from itertools import combinations
from pathlib import Path

#Our bots name and ID 
MY_ID = "groupM-poker-bot"
MY_NAME = "Group M Poker Bot"

# Define the ranks and suits of the cards
RANKS = {str(n): n for n in range(2, 10)}
RANKS.update({"t": 10, "10": 10, "j": 11, "q": 12, "k": 13, "a": 14})
SUITS = {
    "c": "c", "d": "d", "h": "h", "s": "s",
    "clubs": "c", "diamonds": "d", "hearts": "h", "spades": "s",
}
FULL_DECK = [(rank, suit) for rank in range(2, 15) for suit in "cdhs"]

DEFAULT_STRATEGY = {"raise_at": 0.70, "call_margin": 0.0, "raise_fraction": 0.0}


def load_strategy():
    """Numbers written by train.py. Missing file keeps the built-in cutoffs."""
    path = Path(__file__).with_name("strategy.json")
    strategy = dict(DEFAULT_STRATEGY)
    if path.exists():
        strategy.update(json.loads(path.read_text()))
    return strategy


STRATEGY = load_strategy()

# Prior of 1 for each action. Real actions are added on top. A player is
# treated as unknown until we have seen MIN_SEEN actions from them.
OPPONENTS = {}
LAST_NOTED = None
MIN_SEEN = 15

# Parse the card into a tuple of rank and suit
def parse_card(card):
    """Turn a server card into (rank, suit)."""

    if isinstance(card, dict) and "rank" in card and "suit" in card:
        rank_text = str(card["rank"]).lower()
        suit_text = str(card["suit"]).lower()
    else:
        label = card.get("label") if isinstance(card, dict) else str(card)
        label = label.strip().lower().replace(" ", "")
        suit_text = label[-1]
        rank_text = label[:-1]

    rank = int(rank_text) if rank_text.isdigit() else RANKS[rank_text]

    return rank, SUITS[suit_text]

# Get the board cards from the state
def board_cards(state):
    for key in ("board", "community_cards", "community", "table"):
        cards = state.get(key)
        if cards:
            return [parse_card(card) for card in cards]
    return []

# Get the active opponents from the state
def active_opponents(state):
    count = 0
    for player in state["players"]:
        if player["id"] == MY_ID:
            continue
        if player.get("folded") or player.get("in_hand") is False:
            continue
        if player.get("status") in ("folded", "out", "sitting_out"):
            continue
        count += 1
    return max(count, 1)

# Rank the five cards into a tuple of rank and suit
def rank_five(cards):
    """Rank one 5-card hand. Higher tuples win."""
    ranks = sorted((card[0] for card in cards), reverse=True)
    flush = len({card[1] for card in cards}) == 1
    unique = sorted(set(ranks), reverse=True)

    straight_high = None
    if len(unique) == 5 and unique[0] - unique[4] == 4:
        straight_high = unique[0]
    if unique == [14, 5, 4, 3, 2]:
        straight_high = 5

    counts = {}
    for rank in ranks:
        counts[rank] = counts.get(rank, 0) + 1
    groups = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)

    if straight_high and flush:
        return (8, straight_high)
    if groups[0][1] == 4:
        return (7, groups[0][0], groups[1][0])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return (6, groups[0][0], groups[1][0])
    if flush:
        return (5, *ranks)
    if straight_high:
        return (4, straight_high)
    if groups[0][1] == 3:
        kickers = [rank for rank, _ in groups[1:]]
        return (3, groups[0][0], *kickers)
    if groups[0][1] == 2 and groups[1][1] == 2:
        high_pair, low_pair = sorted((groups[0][0], groups[1][0]), reverse=True)
        return (2, high_pair, low_pair, groups[2][0])
    if groups[0][1] == 2:
        kickers = [rank for rank, _ in groups[1:]]
        return (1, groups[0][0], *kickers)
    return (0, *ranks)


def best_hand(cards):
    return max(rank_five(five) for five in combinations(cards, 5))


def live_opponents(state):
    """Player ids still contesting the hand."""
    opponents = []
    for player in state["players"]:
        if player["id"] == MY_ID:
            continue
        if player.get("folded") or player.get("in_hand") is False:
            continue
        if player.get("status") in ("folded", "out", "sitting_out"):
            continue
        opponents.append(player["id"])
    return opponents


def note_action(state):
    """Count a fold, call, or raise once, and update the Bayesian range."""
    global LAST_NOTED
    from opponent_model import MODEL
    MODEL.observe_state(state, MY_ID)
    last = state.get("last_action") or {}
    player = last.get("player")
    kind = last.get("type")
    marker = (state.get("hand_number"), player, kind, last.get("amount"), last.get("target"))
    if marker == LAST_NOTED or not player or player == MY_ID:
        return
    LAST_NOTED = marker
    if kind == "check":
        kind = "call"
    if kind not in ("fold", "call", "raise"):
        return
    counts = OPPONENTS.setdefault(player, {"fold": 1, "call": 1, "raise": 1})
    counts[kind] += 1


def seen_actions(player_id):
    counts = OPPONENTS.get(player_id)
    if not counts:
        return 0, None
    total = counts["fold"] + counts["call"] + counts["raise"]
    return total - 3, counts


def tightness(player_id):
    """Positive when this opponent raises rarely. None until we know them."""
    seen, counts = seen_actions(player_id)
    if counts is None or seen < MIN_SEEN:
        return None
    raise_rate = counts["raise"] / (counts["fold"] + counts["call"] + counts["raise"])
    return max(-0.8, min(1.2, (0.30 - raise_rate) / 0.20))


def hand_prior(hand):
    """Higher for pairs and big cards. Used only as a range weight."""
    low, high = sorted(card[0] for card in hand)
    suited = hand[0][1] == hand[1][1]
    if high == low:
        return 0.45 + high / 30
    score = (high + low) / 40
    if suited:
        score += 0.08
    if high - low == 1:
        score += 0.04
    return score


def take_hand(deck, player_tightness, rng=None):
    """Deal two cards. Unknown opponents are uniform. Known ones use their range."""
    rng = rng or random
    if player_tightness is None or len(deck) < 2:
        return [deck.pop(), deck.pop()]
    pairs = list(combinations(range(len(deck)), 2))
    weights = []
    for i, j in pairs:
        prior = hand_prior((deck[i], deck[j]))
        weights.append(max(0.02, prior ** (1 + player_tightness)))
    i, j = rng.choices(pairs, weights=weights, k=1)[0]
    first, second = max(i, j), min(i, j)
    return [deck.pop(first), deck.pop(second)]


def win_chance(state, trials=200, rng=None):
    """Share of the pot over simulated runouts. Known opponents use weighted ranges."""
    rng = rng or random
    me = next(player for player in state["players"] if player["id"] == MY_ID)
    hole = [parse_card(card) for card in me["hole_cards"]]
    board = board_cards(state)
    known = set(hole + board)
    deck = [card for card in FULL_DECK if card not in known]

    missing_board = 5 - len(board)
    opponents = live_opponents(state) or ["unknown"]
    while len(opponents) > 1 and len(opponents) * 2 + missing_board > len(deck):
        opponents.pop()

    equity_total = 0.0

    for _ in range(trials):
        pool = deck[:]
        rng.shuffle(pool)
        opponent_hands = [take_hand(pool, tightness(player_id), rng) for player_id in opponents]
        full_board = board + pool[:missing_board]

        my_rank = best_hand(hole + full_board)
        opponent_ranks = [best_hand(hand + full_board) for hand in opponent_hands]
        best_rank = max([my_rank] + opponent_ranks)

        if my_rank == best_rank:
            num_winners = 1 + sum(rank == my_rank for rank in opponent_ranks)
            equity_total += 1 / num_winners

    return equity_total / trials


def table_rates(player_ids):
    """Average fold and raise rates for the opponents still in this hand."""
    folds = []
    raises = []
    for player_id in player_ids:
        seen, counts = seen_actions(player_id)
        if counts is None or seen < MIN_SEEN:
            continue
        total = counts["fold"] + counts["call"] + counts["raise"]
        folds.append(counts["fold"] / total)
        raises.append(counts["raise"] / total)
    if not folds:
        return None, None
    return sum(folds) / len(folds), sum(raises) / len(raises)

def choose_threshold(state):
    """Aggressor-style fallback. Used when the adaptive decision fails."""
    me = next(p for p in state["players"] if p["id"] == MY_ID)

    highest_bet = max(p["bet"] for p in state["players"])
    to_call = highest_bet - me["bet"]
    equity = win_chance(state)
    pot = state.get("pot", 0)
    phase = state.get("phase", "")
    fold_rate, raise_rate = table_rates(live_opponents(state))
    raise_at = STRATEGY["raise_at"]
    call_margin = STRATEGY["call_margin"]

    # Frequent folders are bluffed more. Rare raisers get more credit for a strong hand.
    if fold_rate is not None and fold_rate >= 0.45:
        raise_at = max(0.18, raise_at - 0.12)
    elif fold_rate is not None and fold_rate <= 0.22:
        raise_at = min(0.75, raise_at + 0.10)
    if raise_rate is not None and to_call > 0:
        if raise_rate <= 0.18:
            call_margin += 0.08
        elif raise_rate >= 0.45:
            call_margin -= 0.06

    if phase == "PREFLOP" and to_call > 0 and equity < STRATEGY.get("preflop_floor", 0):
        return "fold", 0

    if equity >= raise_at and me["stack"] > to_call:
        extra = max(state["minimum_raise"], int(pot * STRATEGY["raise_fraction"]))
        total = state["current_bet"] + extra
        total = min(total, me["bet"] + me["stack"])
        return "raise", int(total)

    if to_call == 0:
        return "check", 0

    pot_odds = to_call / (pot + to_call)
    if equity >= pot_odds + call_margin:
        return "call", 0
    return "fold", 0


def choose_action(state):
    """Adaptive EV decision. Falls back to the frozen Aggressor thresholds."""
    try:
        from ev import choose
        from opponent_model import MODEL
        if len(next(p for p in state["players"] if p["id"] == MY_ID).get("hole_cards") or []) < 2:
            return "check", 0
        action, amount, info = choose(state, MODEL, MY_ID)
        print(
            "Equity:", info["equity"],
            "EV:", info["ev"],
            "trials:", info["trials"],
            "->", action, amount,
        )
        return action, amount
    except Exception as error:
        print("Adaptive decision failed, using Aggressor fallback:", error)
        return choose_threshold(state)


# Connect to the poker server using WebSockets
async def session():
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        print("Connected to the poker server!")

        # Create a message with our bot's information and status 
        join_message = {
            "type": "join",
            "name": MY_NAME,
            "player_id": MY_ID
        }

        await websocket.send(json.dumps(join_message))
        print("Join request sent!")

        # Remember our last action in case the server rejects it
        last_sent = None

        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data["type"] == "state":
                state = data["payload"]
                note_action(state)
                print("Game phase:", state["phase"])
                print("Pot:", state["pot"])
                print("Current turn:", state["current_turn"])

                me = next(p for p in state["players"] if p["id"] == MY_ID)

                print("My chips:", me["stack"])
                print("My cards:", [card["label"] for card in me["hole_cards"]])

                # Check whether it is our bot's turn to act
                if state["current_turn"] == MY_ID:
                    print("It's my turn!")

                    action, amount = choose_action(state)
                    action_message = {
                        "type": "action",
                        "action": action,
                        "amount": amount
                    }
                    last_sent = action
                    await websocket.send(json.dumps(action_message))
                    print("Action sent:", action)


                else:
                    print("Waiting for my turn...")
     
            # Check whether the server accepted our action
            elif data["type"] == "action_result":
                result = data["payload"]

                # If our action was rejected, print the error
                if not result["ok"]:
                    print("Action rejected:", result["error"])

                    if last_sent != "call":
                        last_sent = "call"

                        await websocket.send(json.dumps({
                            "type": "action",
                            "action": "call"
                        }))




async def main():
    while True:
        try:
            await session()
        except KeyboardInterrupt:
            raise
        except Exception as error:
            print("Connection lost, retrying:", error)
            await asyncio.sleep(2)


# Run the main function when this file is executed
if __name__ == "__main__":
    asyncio.run(main())
