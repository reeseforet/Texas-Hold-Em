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

    # Call if the expected value is positive
    if call_ev > 0:
        return "call"
    else:
        return "fold"



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
