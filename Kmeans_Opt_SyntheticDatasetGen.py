import numpy as np
import pandas as pd
import pulp
from sklearn.cluster import KMeans
import os

def determine_alpha(num_users):
    if num_users < 100:
        return 0.1
    elif num_users < 200:
        return 0.2
    elif num_users < 400:
        return 0.3
    else:
        return 0.4

def determine_num_clusters(num_users):
    if num_users < 100:
        return int(0.1 * num_users)
    elif num_users < 200:
        return int(0.2 * num_users)
    elif num_users < 400:
        return int(0.3 * num_users)
    else:
        return int(0.4 * num_users)

def compute_aij(d_ij, fc, C_speed, L_threshold, h_0):
    term1 = (4 * np.pi * fc / C_speed) ** 2
    return L_threshold - term1 * (d_ij ** 2 - h_0 ** 2)

def compute_aik(d_ik, fc, C_speed, L_threshold, h_0):
    term1 = (4 * np.pi * fc / C_speed) ** 2
    return L_threshold - term1 * (d_ik ** 2 - h_0 ** 2)

def uav_deployment_optimization(num_users, area_size):
    num_clusters = determine_num_clusters(num_users)
    alt_min = 50
    alt_max = 250
    gamma = 1

    # Generate random user locations within the defined area size
    user_locations = np.random.rand(num_users, 2) * area_size

    # Perform clustering to determine UAV positions
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=20, max_iter=500)
    kmeans.fit(user_locations)
    centroids = kmeans.cluster_centers_

    # Define the optimization problem
    prob = pulp.LpProblem("UAV_Deployment", pulp.LpMinimize)

    # Define variables for UAV deployment
    p = pulp.LpVariable.dicts("p", range(num_clusters), cat='Binary')
    h = pulp.LpVariable.dicts("h", range(num_clusters), lowBound=alt_min, upBound=alt_max, cat='Continuous')
    u = pulp.LpVariable.dicts("u", [(i, j) for i in range(num_clusters) for j in range(num_users)], cat='Binary')
    L = pulp.LpVariable.dicts("L", [(i, k) for i in range(num_clusters) for k in range(num_clusters)], lowBound=0, cat='Continuous')
    C = pulp.LpVariable.dicts("C", [(i, k) for i in range(num_clusters) for k in range(num_clusters)], cat='Binary')
    S = pulp.LpVariable.dicts("S", [(i, k) for i in range(num_clusters) for k in range(num_clusters)], cat='Binary')
    y = pulp.LpVariable.dicts("y", [(i, k) for i in range(num_clusters) for k in range(num_clusters)], lowBound=0, cat='Continuous')
    x = pulp.LpVariable.dicts("x", [(i, j) for i in range(num_clusters) for j in range(num_users)], lowBound=0, cat='Continuous')
    k = pulp.LpVariable.dicts("k", [(i, j) for i in range(num_clusters) for j in range(num_users)], lowBound=0, cat='Continuous')
    t = pulp.LpVariable.dicts("t", [(i, j) for i in range(num_clusters) for j in range(num_users)], lowBound=0, cat='Continuous')

    # Constants
    fc = 2 * 10 ** 9
    C_speed = 3 * 10 ** 8
    L_threshold = 350
    theta = np.radians(45)
    B = 300
    rd = 5
    M = 10 ** 12
    h_0 = (alt_max + alt_min) / 2

    # Objective function
    prob += pulp.lpSum(p[i] for i in range(num_clusters)) + pulp.lpSum(C[i, k] for i in range(num_clusters) for k in range(num_clusters) if i != k)

    # Constraint: Total number of selected UAVs should not exceed the optimal number of UAVs
    prob += pulp.lpSum(p[i] for i in range(num_clusters)) <= num_clusters

    # Altitude constraints for UAVs
    for i in range(num_clusters):
        prob += h[i] <= alt_max * p[i]
        prob += h[i] >= alt_min * p[i]

    # Each user must be assigned to exactly one UAV
    for j in range(num_users):
        prob += pulp.lpSum(u[i, j] for i in range(num_clusters)) == 1

    # Ensure each user is served by a feasible UAV position
    for i in range(num_clusters):
        for j in range(num_users):
            prob += u[i, j] <= p[i]

    # Total number of users served by all UAVs must exceed a specified percentage of the total users
    prob += pulp.lpSum(u[i, j] for i in range(num_clusters) for j in range(num_users)) >= gamma * num_users

    # Data rate constraints for each UAV
    for i in range(num_clusters):
        prob += pulp.lpSum(rd * u[i, j] for j in range(num_users)) <= B * p[i]

    # Coverage constraints for each UAV
    cot_theta = 1 / np.tan(theta)
    for i in range(num_clusters):
        for j in range(num_users):
            d_ij = np.linalg.norm(user_locations[j] - centroids[i])
            prob += cot_theta * u[i, j] * d_ij <= h[i]

    # Linearize and reformulate non-linear constraints
    for i in range(num_clusters):
        for j in range(num_users):
            d_ij = np.linalg.norm(user_locations[j] - centroids[i])
            a_ij = compute_aij(d_ij, fc, C_speed, L_threshold, h_0)
            prob += x[i, j] <= (M - h[i]) / (M - a_ij + 0.5)
            prob += x[i, j] <= u[i, j]

    # Connectivity constraints and other non-linear constraint reformulations
    for i in range(num_clusters):
        for j in range(num_users):
            prob += k[i, j] <= M * x[i, j]

    for i in range(num_clusters):
        for j in range(num_users):
            d_ij = np.linalg.norm(user_locations[j] - centroids[i])
            term1 = (4 * np.pi * fc / C_speed) ** 2 * (d_ij ** 2 - h_0 ** 2)
            term2 = (4 * np.pi * fc / C_speed) ** 2 * 2 * h_0 * t[i, j]
            prob += k[i, j] >= term1 * x[i, j] + term2

    for i in range(num_clusters):
        for j in range(num_users):
            prob += t[i, j] <= h[i]
            prob += t[i, j] <= alt_max * x[i, j]
            prob += t[i, j] >= h[i] - (1 - x[i, j]) * alt_max

    # Non-linear constraint linearization for connectivity
    for i in range(num_clusters):
        for k in range(num_clusters):
            if i != k:
                d_ik = np.linalg.norm(centroids[i] - centroids[k])
                a_ik = compute_aik(d_ik * 1.2, fc, C_speed, L_threshold, h_0)
                prob += C[i, k] <= (M - (h[i] - h[k])) / (M - a_ik + 0.5)
                prob += C[i, k] <= p[i]
                prob += C[i, k] <= p[k]

    # Additional constraints for connectivity
    for i in range(num_clusters):
        for k in range(num_clusters):
            if i != k:
                prob += L[i, k] <= M * C[i, k]

    for i in range(num_clusters):
        for k in range(num_clusters):
            if i != k:
                d_ik = np.linalg.norm(centroids[i] - centroids[k])
                term1 = (4 * np.pi * fc / C_speed) ** 2 * (d_ik ** 2 - h_0 ** 2) * 1.2
                term2 = (4 * np.pi * fc / C_speed) ** 2 * 2 * h_0 * y[i, k]
                prob += L[i, k] >= term1 * C[i, k] + term2

    for i in range(num_clusters):
        for k in range(num_clusters):
            if i != k:
                prob += y[i, k] <= h[i]
                prob += y[i, k] <= alt_max * C[i, k]
                prob += y[i, k] >= h[i] - (1 - C[i, k]) * alt_max

    # Ensure at least one connectivity per UAV
    for i in range(num_clusters):
        prob += pulp.lpSum(C[i, k] for k in range(num_clusters) if i != k) >= 1 * p[i]

    prob += pulp.lpSum(S[i, k] for i in range(num_clusters) for k in range(num_clusters) if i != k) >= pulp.lpSum(p[i] for i in range(num_clusters)) - 1
    for F in range(1, num_clusters):
        prob += pulp.lpSum(S[i, k] for i in range(F) for k in range(num_clusters) if i != k) <= F - 1

    # Solve the problem
    prob.solve()
    print("Status:", pulp.LpStatus[prob.status])

    optimal_positions = []
    optimal_altitudes = []

    for i in range(num_clusters):
        if p[i].varValue == 1:
            optimal_positions.append(centroids[i])
            optimal_altitudes.append(h[i].varValue)

    print("Optimal UAV positions:", optimal_positions)
    print("Optimal UAV altitudes:", optimal_altitudes)

    return optimal_positions, optimal_altitudes, user_locations

def generate_synthetic_data(num_scenarios, area_size, user_range):
    """
    Generate synthetic data for UAV deployment optimization scenarios and save it in CSV format.
    Returns input and output data for further use.
    """
    input_data = []
    output_data = []

    for scenario in range(num_scenarios):
        num_users = np.random.randint(user_range[0], user_range[1])
        print(f"Generating scenario {scenario + 1} with {num_users} users.")
        
        # Run optimization for the given number of users
        optimal_positions, optimal_altitudes, user_locations = uav_deployment_optimization(num_users, area_size)
        
        # Prepare input features: number of users, Xs, and Ys
        for loc in user_locations:
            input_data.append([num_users, loc[0], loc[1]])
        
        # Prepare output labels: number of users, Xd, Yd, and Ad
        for pos, alt in zip(optimal_positions, optimal_altitudes):
            output_data.append([num_users, pos[0], pos[1], alt])
    
    # Convert lists to DataFrames with specified columns
    input_df = pd.DataFrame(input_data, columns=['Num of users', 'Xs', 'Ys'])
    output_df = pd.DataFrame(output_data, columns=['Num of users', 'Xd', 'Yd', 'Ad'])
    
    # Create the directory if it does not exist
    if not os.path.exists('synthetic_dataset3'):
        os.makedirs('synthetic_dataset3')
    
    # Save DataFrames as CSV files
    input_df.to_csv('synthetic_dataset3/SensorsList_p22.csv', index=False)
    output_df.to_csv('synthetic_dataset3/DronesList_p22.csv', index=False)
    print("Synthetic dataset saved in CSV format.")

# Main script
area_size = 500
num_scenarios = 100  # Number of different scenarios to generate
user_range = (100,101)  # Range of users to simulate

# Generate the synthetic dataset
print("Generating synthetic dataset...")
generate_synthetic_data(num_scenarios, area_size, user_range)