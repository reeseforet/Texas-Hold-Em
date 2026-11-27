# Import our Monte Carlo equity calculation
from monte_carlo import monte_carlo_equity


# Calculate the minimum equity needed to call a bet
def calculate_pot_odds(pot, to_call):
    # If checking is free, no equity is needed to continue
    if to_call <= 0:
        return 0

    # The server's pot already includes previous bets
    final_pot = pot + to_call

    # Return the fraction of the final pot we must contribute
    return to_call / final_pot


# Calculate the expected value of calling a bet
def calculate_call_ev(pot, to_call, equity):
    # Estimate our expected share of the pot after calling
    final_pot = pot + to_call

    # Subtract the chips we must contribute
    return (equity * final_pot) - to_call




# Calculate a possible raise amount based on the current game state
def calculate_raise_amount(my_bet, my_stack, highest_bet,
                           big_blind, minimum_raise=None, can_raise=True):

    # Don't attempt a raise if the server's betting rules prohibit it
    if not can_raise:
        return None

    # Maximum total bet we can make
    max_raise = my_bet + my_stack

    # Use the big blind as the minimum increase when not provided
    if minimum_raise is None:
        minimum_raise = big_blind

    min_raise_target = highest_bet + minimum_raise

    # Cannot raise beyond the current bet
    if max_raise <= highest_bet:
        return None

    # Allow a short all-in raise
    if max_raise < min_raise_target:
        return max_raise

    # Aim for twice the minimum increase
    desired_raise = highest_bet + (2 * minimum_raise)

    return min(desired_raise, max_raise)



# Choose an action based on our estimated equity and the cost to call
def decide_action(pot, to_call, equity):
    # Checking is free, so stay in the hand
    if to_call <= 0:
        return "check"

    # Calculate the expected value of calling
    call_ev = calculate_call_ev(pot, to_call, equity)

    # Call when the expected value is positive
    if call_ev > 0:
        return "call"

    # Fold when calling would have a negative or zero expected value
    return "fold"



# Decide whether our hand is strong enough to consider raising
def should_raise(equity, threshold=0.75):
    # Raise only when our estimated equity meets the threshold
    return equity >= threshold



# Decide whether to fold, check, or call based on our hand equity
def choose_strategy(my_cards, community_cards, pot, to_call,
                    num_opponents=1):

    # Estimate our hand equity using Monte Carlo simulation
    equity = monte_carlo_equity(
        my_cards,
        community_cards,
        simulations=1000,
        num_opponents=num_opponents
    )



  
    # Display our estimated equity
    print("Estimated equity:", round(equity * 100, 2), "%")

    # If no additional chips are required, check
    if to_call == 0:
        print("No bet to call, checking for free")
        return "check"

    # Calculate the expected value of calling
    call_ev = calculate_call_ev(pot, to_call, equity)

    # Display the calculations used to make our decision
    print("Pot odds:", round(calculate_pot_odds(pot, to_call) * 100, 2), "%")
    print("Call expected value:", round(call_ev, 2))

    # Use our decision function to choose the action
    return decide_action(pot, to_call, equity)



# Test our pot odds calculations
if __name__ == "__main__":
    print("Free check:", calculate_pot_odds(100, 0))
    print("Call 20 into pot of 100:", calculate_pot_odds(100, 20))
    print("Call 50 into pot of 150:", calculate_pot_odds(150, 50))
    # Test the expected value of calling
    print("Positive EV:", calculate_call_ev(100, 20, 0.60))
    print("Negative EV:", calculate_call_ev(100, 50, 0.20))

    # Test our Monte Carlo-based decision making
    my_cards = [
        {"rank": 14, "suit": "spades"},
        {"rank": 13, "suit": "hearts"}
    ]

    community_cards = [
        {"rank": 12, "suit": "clubs"},
        {"rank": 11, "suit": "diamonds"},
        {"rank": 2, "suit": "spades"}
    ]

    # Test a situation where we need to call 20 chips
    decision = choose_strategy(
        my_cards,
        community_cards,
        pot=100,
        to_call=20,
        num_opponents=3
    )

    print("AI decision:", decision)
    print("Valid decision:", decision in ["fold", "check", "call"])


    # Test calculating a normal raise
    print("Normal raise:", calculate_raise_amount(
        my_bet=0,
        my_stack=1000,
        highest_bet=20,
        big_blind=20
    ))

    # Test raising when we have limited chips
    print("Limited stack raise:", calculate_raise_amount(
        my_bet=20,
        my_stack=45,
        highest_bet=40,
        big_blind=20
    ))

    # Test when we cannot raise
    print("Cannot raise:", calculate_raise_amount(
        my_bet=20,
        my_stack=10,
        highest_bet=40,
        big_blind=20
    ))

