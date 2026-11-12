
import matplotlib.pyplot as plt

# Results from our first Monte Carlo comparison
opponents = [1, 2, 3, 5]
win_probabilities = [61.5, 41.0, 36.0, 23.7]

# Create a graph comparing win probability and opponent count
plt.plot(opponents, win_probabilities, marker="o")

plt.title("Monte Carlo Win Probability vs. Number of Opponents")
plt.xlabel("Number of Opponents")
plt.ylabel("Estimated Win Probability (%)")

plt.xticks(opponents)
plt.ylim(0, 100)
plt.grid(True)

# Save the graph for our final report
plt.savefig("monte_carlo_results.png", dpi=300, bbox_inches="tight")

# Display the graph
plt.show()


# Compare how long different simulation counts take
simulation_counts = [100, 500, 1000, 5000]
runtimes = [0.045, 0.197, 0.428, 2.133]

# Create a new graph for simulation runtime
plt.figure()

plt.plot(simulation_counts, runtimes, marker="o")

plt.title("Monte Carlo Simulation Count vs. Runtime")
plt.xlabel("Number of Simulations")
plt.ylabel("Runtime (seconds)")

plt.grid(True)

# Save the graph for our report
plt.savefig("simulation_runtime.png", dpi=300, bbox_inches="tight")

# Display the graph
plt.show()


