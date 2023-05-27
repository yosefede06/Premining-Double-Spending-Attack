import copy
import numpy as np
import pandas as pd
import plotly.express as px

ATTACKER_RATE = 0.25
HONEST_RATE = 1 - ATTACKER_RATE
MAXIMUM_BLOCK_ADVANTAGE = 30
ITERATIONS = 1000
SHOW_PROBABILITIES = min(MAXIMUM_BLOCK_ADVANTAGE, 10)
INITIAL_DISTRIBUTION = np.ones((MAXIMUM_BLOCK_ADVANTAGE, 1))/MAXIMUM_BLOCK_ADVANTAGE


def get_stationary_distribution(n):
    return ((1 - 2 * ATTACKER_RATE) / (1 - ATTACKER_RATE)) * (ATTACKER_RATE / (1 - ATTACKER_RATE))**n


def calculate_basis_matrix():
    t = np.zeros((MAXIMUM_BLOCK_ADVANTAGE, MAXIMUM_BLOCK_ADVANTAGE))
    t[0][0] = HONEST_RATE
    t[MAXIMUM_BLOCK_ADVANTAGE - 1][MAXIMUM_BLOCK_ADVANTAGE - 1] = ATTACKER_RATE
    for i in range(MAXIMUM_BLOCK_ADVANTAGE):
        for j in range(MAXIMUM_BLOCK_ADVANTAGE):
            if i - j == 1:
                t[i][j] = ATTACKER_RATE
            elif i - j == -1:
                t[i][j] = HONEST_RATE
    return t


def calculate_k_matrix(t, prev_t):
    # (t^k)i->j = sum((tj-1->j)(t^k-1)i->j-1, (tj->j)t^ki->tj, (tj+1->j)t^ki->tj+1)
    tk = np.zeros((MAXIMUM_BLOCK_ADVANTAGE, MAXIMUM_BLOCK_ADVANTAGE))
    for i in range(MAXIMUM_BLOCK_ADVANTAGE):
        for j in range(MAXIMUM_BLOCK_ADVANTAGE):
            if j == 0:
                tk[j][i] = t[j][j] * prev_t[j][i] + t[j][j+1] * prev_t[j+1][i]
            if j == MAXIMUM_BLOCK_ADVANTAGE - 1:
                tk[j][i] = t[j][j - 1] * prev_t[j - 1][i] + t[j][j] * prev_t[j][i]
            else:
                tk[j][i] = t[j][j-1] * prev_t[j-1][i] + t[j][j] * prev_t[j][i] + t[j][j+1] * prev_t[j+1][i]
    return tk


def compute_stationary_distribution(p0=INITIAL_DISTRIBUTION, iterations=ITERATIONS, max_states=MAXIMUM_BLOCK_ADVANTAGE):
    t = calculate_basis_matrix()
    prev_t = copy.deepcopy(t)
    plot_distribution = np.zeros(shape=(max_states, iterations))
    for k in range(iterations):
        for i in range(max_states):
            plot_distribution[i][k] = (prev_t @ p0)[i][0]
        prev_t = calculate_k_matrix(t, prev_t)
    return prev_t @ p0, plot_distribution


def question2():
    p_star = compute_stationary_distribution()[0]
    for i in range(SHOW_PROBABILITIES):
        print("Probability of finishing on state " + str(i) + ": " + str('{:.20f}'.format(p_star[i][0])))
    fig = px.scatter(title=f"Stationary Distribution",
                     x=np.arange(SHOW_PROBABILITIES),
                     y=p_star[:SHOW_PROBABILITIES].T[0])
    fig.update_layout(xaxis_title="State", yaxis_title="Probability")
    fig.show()


def question4():
    p0 = np.zeros((MAXIMUM_BLOCK_ADVANTAGE, 1))
    p0[0] = 1
    iterations = 100
    plot_distribution = compute_stationary_distribution(p0=p0, iterations=100)[1]
    fig = px.scatter(title=f"Convergence To Stationary Distribution For An Attacker With 25% Of Compute Power")
    for i in range(MAXIMUM_BLOCK_ADVANTAGE):
        fig.add_scatter(x=np.arange(iterations),
                        y=np.abs(plot_distribution[i] - [get_stationary_distribution(i)] * iterations),
                        name=str(i) + "th state")
    fig.update_layout(xaxis_title="Number Of Iterations",
                      yaxis_title="Probability Loss to Stationary Distribution")
    fig.show()


if __name__ == '__main__':
    question2()
    question4()

