
import time
from monte_carlo import monte_carlo_simulation


# Cards used in our previous Monte Carlo experiments
my_cards = [
    {"rank": 14, "suit": "spades"},
    {"rank": 13, "suit": "hearts"}
]

community_cards = [
    {"rank": 12, "suit": "clubs"},
    {"rank": 11, "suit": "diamonds"},
    {"rank": 2, "suit": "spades"}
]


# Compare simulation counts using the same cards and opponents
simulation_counts = [100, 500, 1000, 5000]

print("Monte Carlo Simulation Count Comparison")

for count in simulation_counts:
    # Record when the simulation starts
    start_time = time.perf_counter()

    # Run the Monte Carlo simulation against three opponents
    probability, wins, losses, ties = monte_carlo_simulation(
        my_cards,
        community_cards,
        simulations=count,
        num_opponents=3
    )

    # Calculate how long the simulation took
    elapsed_time = time.perf_counter() - start_time

    print("-----------------------")
    print("Simulations:", count)
    print("Win probability:", round(probability * 100, 2), "%")
    print("Wins:", wins)
    print("Losses:", losses)
    print("Ties:", ties)
    print("Time:", round(elapsed_time, 3), "seconds")
