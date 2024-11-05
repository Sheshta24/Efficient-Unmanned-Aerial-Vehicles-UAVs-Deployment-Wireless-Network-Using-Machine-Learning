import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pulp

def determine_alpha(num_users):
    if num_users <= 100:
        return 0.1
    elif num_users < 200:
        return 0.2
    elif num_users < 400:
        return 0.3
    else:
        return 0.4

def determine_num_clusters(num_users):
    if num_users <100:
        return int(0.1 * num_users)
    elif num_users < 200:
        return int(0.2 * num_users)
    elif num_users < 400:
        return int(0.3 * num_users)
    else:
        return int(0.4 * num_users)

def compute_aij(d_ij, fc, C_speed, L_threshold, h_0):
    term1 = (4 * np.pi * fc / C_speed)**2
    return L_threshold - term1 * (d_ij**2 - h_0**2)

def compute_aik(d_ik, fc, C_speed, L_threshold, h_0):
    term1 = (4 * np.pi * fc / C_speed)**2
    return L_threshold - term1 * (d_ik**2 - h_0**2)

def uav_deployment_optimization(num_users, area_size):
    num_clusters = determine_num_clusters(num_users)
    alt_min = 50
    alt_max = 250
    gamma = 1
    
    np.random.seed(42)

    user_locations = np.random.rand(num_users, 2) * area_size

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=20, max_iter=500)
    kmeans.fit(user_locations)
    centroids = kmeans.cluster_centers_

    prob = pulp.LpProblem("UAV_Deployment", pulp.LpMinimize)

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

    fc = 2 * 10**9
    C_speed = 3 * 10**8
    L_threshold = 350
    theta = np.radians(45)
    B = 300
    rd = 5
    M = 10**12
    h_0 = (alt_max + alt_min) / 2

    prob += pulp.lpSum(p[i] for i in range(num_clusters)) + pulp.lpSum(C[i, k] for i in range(num_clusters) for k in range(num_clusters) if i != k)

    # (1b) Total number of selected points should not exceed the optimal number of UAVs
    prob += pulp.lpSum(p[i] for i in range(num_clusters)) <= num_clusters

    # (1c) and (1d) Altitude constraints
    for i in range(num_clusters):
        prob += h[i] <= alt_max * p[i]
        prob += h[i] >= alt_min * p[i]

    # Each user must be assigned to exactly one UAV
    for j in range(num_users):
        prob += pulp.lpSum(u[i, j] for i in range(num_clusters)) == 1

    # (1f) Ensure each user is served by a feasible point if that point is among the optimal positions
    for i in range(num_clusters):
        for j in range(num_users):
            prob += u[i, j] <= p[i]

    # (1g) Total number of users served by all UAVs must exceed a specified percentage of the total users
    prob += pulp.lpSum(u[i, j] for i in range(num_clusters) for j in range(num_users)) >= gamma * num_users

    # (1h) Data rate constraints for each UAV
    for i in range(num_clusters):
        prob += pulp.lpSum(rd * u[i, j] for j in range(num_users)) <= B * p[i]

    # (1i) Coverage constraints for each UAV
    cot_theta = 1 / np.tan(theta)
    for i in range(num_clusters):
        for j in range(num_users):
            d_ij = np.linalg.norm(user_locations[j] - centroids[i])
            prob += cot_theta * u[i, j] * d_ij <= h[i]

    # Linearize and reformulate non-linear constraints
    # (1j) Linearized form with decision variable x_ij
    for i in range(num_clusters):
        for j in range(num_users):
            d_ij = np.linalg.norm(user_locations[j] - centroids[i])
            a_ij = compute_aij(d_ij, fc, C_speed, L_threshold, h_0)
            prob += x[i, j] <= (M - h[i]) / (M - a_ij + 0.5)
            prob += x[i, j] <= u[i, j]

    # (1k) Constraint implementation
    for i in range(num_clusters):
        for j in range(num_users):
            prob += k[i, j] <= M * x[i, j]

    # (1l) Implementation
    for i in range(num_clusters):
        for j in range(num_users):
            d_ij = np.linalg.norm(user_locations[j] - centroids[i])
            term1 = (4 * np.pi * fc / C_speed)**2 * (d_ij**2 - h_0**2)
            term2 = (4 * np.pi * fc / C_speed)**2 * 2 * h_0 * t[i, j]
            prob += k[i, j] >= term1 * x[i, j] + term2

    for i in range(num_clusters):
        for j in range(num_users):
            prob += t[i, j] <= h[i]
            prob += t[i, j] <= alt_max * x[i, j]
            prob += t[i, j] >= h[i] - (1 - x[i, j]) * alt_max

    # (1m/1n) Non-linear constraint linearization for connectivity
    for i in range(num_clusters):
        for k in range(num_clusters):
            if i != k:
                d_ik = np.linalg.norm(centroids[i] - centroids[k])
                a_ik = compute_aik(d_ik, fc, C_speed, L_threshold, h_0)
                prob += C[i, k] <= (M - (h[i] - h[k])) / (M - a_ik + 0.5)
                prob += C[i, k] <= p[i]
                prob += C[i, k] <= p[k]

    # (1o) Ensure L_ik <= M * C_ik
    for i in range(num_clusters):
        for k in range(num_clusters):
            if i != k:
                prob += L[i, k] <= M * C[i, k]

    # (1p) Linearized form with decision variable y_ik
    for i in range(num_clusters):
        for k in range(num_clusters):
            if i != k:
                d_ik = np.linalg.norm(centroids[i] - centroids[k])
                term1 = (4 * np.pi * fc / C_speed)**2 * (d_ik**2 - h_0**2)
                term2 = (4 * np.pi * fc / C_speed)**2 * 2 * h_0 * y[i, k]
                prob += L[i, k] >= term1 * C[i, k] + term2

    for i in range(num_clusters):
        for k in range(num_clusters):
            if i != k:
                prob += y[i, k] <= h[i]
                prob += y[i, k] <= alt_max * C[i, k]
                prob += y[i, k] >= h[i] - (1 - C[i, k]) * alt_max

    # Connectivity constraints (1q) to (1t)
    for i in range(num_clusters):
        prob += pulp.lpSum(C[i, k] for k in range(num_clusters) if i != k) >= 1 * p[i]

    prob += pulp.lpSum(S[i, k] for i in range(num_clusters) for k in range(num_clusters) if i != k) >= pulp.lpSum(p[i] for i in range(num_clusters)) - 1
    for F in range(1, num_clusters):
        prob += pulp.lpSum(S[i, k] for i in range(F) for k in range(num_clusters) if i != k) <= F - 1

    prob.solve()
    print("Status:", pulp.LpStatus[prob.status])

    optimal_positions = []
    optimal_altitudes = []
    user_assignments = np.zeros((num_clusters, num_users), dtype=int)
    original_uav_indices = []
    connectivity_links = 0

    for i in range(num_clusters):
        if p[i].varValue == 1:
            original_uav_indices.append(i)
            optimal_positions.append(centroids[i])
            optimal_altitudes.append(h[i].varValue)
            for j in range(num_users):
                if u[i, j].varValue == 1:
                    user_assignments[i, j] = 1

    # Corrected section for counting connectivity links (without double counting)
    for i in range(num_clusters):
        for k in range(i + 1, num_clusters):  # Ensure i < k to avoid double counting
            if C[i, k].varValue == 1:
                connectivity_links += 1

    print("Optimal UAV positions:", optimal_positions)
    print("Optimal UAV altitudes:", optimal_altitudes)

    return len(optimal_positions), user_locations, centroids, optimal_positions, optimal_altitudes, user_assignments, original_uav_indices, alt_min, alt_max, C, connectivity_links


def can_assign_user_to_uav(user_index, uav_index, uavs, user_locations, centroids, alt_min, alt_max, user_assignments, rd, B):
    user_location = user_locations[user_index]
    uav_position = centroids[uav_index]

    # Check if the user is within the coverage range of the UAV
    distance = np.linalg.norm(user_location - uav_position)
    coverage_check = distance <= np.tan(np.radians(45)) * (alt_max - alt_min)

    # New check to ensure bandwidth is not exceeded
    current_users_served = np.sum(user_assignments[uav_index])
    if current_users_served * rd < B and coverage_check:
        return True
    return False

def greedy_post_processing(uavs, user_assignments, original_uav_indices, user_locations, centroids, alt_min, alt_max, rd, B):
    
    rd = 5  # Define the data rate per user
    B = 300  # Define the bandwidth of the UAV
    while True:
        impact_scores = []
        for i, uav in enumerate(uavs):
            covered_users = [j for j in range(len(user_locations)) if user_assignments[i, j] == 1]
            impact_scores.append((len(covered_users), i))

        impact_scores.sort()

        if not impact_scores:
            break

        _, least_impact_uav_index = impact_scores[0]
        removed_users = [j for j in range(len(user_locations)) if user_assignments[least_impact_uav_index, j] == 1]

        reassigned = True
        for user in removed_users:
            reassigned_to_other_uav = False
            for i, uav in enumerate(uavs):
                if i == least_impact_uav_index:
                    continue
                if can_assign_user_to_uav(user, i, uavs, user_locations, centroids, alt_min, alt_max, user_assignments, rd, B):
                    user_assignments[i, user] = 1
                    reassigned_to_other_uav = True
                    break

            if not reassigned_to_other_uav:
                reassigned = False
                break

        if reassigned:
            uavs.pop(least_impact_uav_index)
            user_assignments = np.delete(user_assignments, least_impact_uav_index, axis=0)
            original_uav_indices.pop(least_impact_uav_index)
        else:
            break

    return uavs, user_assignments, original_uav_indices
# Main script
area_size = 500
user_counts = [100, 200, 300]  # List of user counts for each optimization run

results = {}
connectivity_links_list = []
max_altitudes_list = []

rd = 5  # Define the data rate per user
B = 300  # Define the bandwidth of the UAV

# Loop through each value in user_counts as an individual number
for num_users in user_counts:
    print(f"Running optimization for num_users={num_users}")

    # Ensure that num_users is passed as a single integer
    num_uavs, user_locations, centroids, optimal_positions, optimal_altitudes, user_assignments, original_uav_indices, alt_min, alt_max, C, connectivity_links = uav_deployment_optimization(num_users, area_size)

    uavs = list(range(num_uavs))
    # Corrected call to greedy_post_processing with rd and B
    uavs, user_assignments, original_uav_indices = greedy_post_processing(
        uavs, user_assignments, original_uav_indices, user_locations, centroids, alt_min, alt_max, rd, B
    )

    num_uavs = len(uavs)
    results[num_users] = num_uavs
    connectivity_links_list.append(connectivity_links)
    if optimal_altitudes:
        max_altitudes_list.append(max(optimal_altitudes))
    else:
        max_altitudes_list.append(None)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.xlabel('Number of Users')
plt.ylabel('Number of UAVs')
plt.title('Number of UAVs vs Number of Users')
plt.grid(True)
plt.show()

# Plotting connectivity links vs number of users
plt.figure(figsize=(10, 6))
plt.plot(user_counts, connectivity_links_list, marker='o', color='blue', label='Proposed')
plt.xlabel('Number of Users')
plt.ylabel('Connectivity Links among UAVs')
plt.title('Number of Connectivity Links among UAVs vs Number of Users')
plt.grid(True)
plt.legend()
plt.show()

# Plotting max altitude vs number of users
plt.figure(figsize=(10, 6))
plt.plot(user_counts, max_altitudes_list, marker='o', color='green', label='Max Altitude')
plt.xlabel('Number of Users')
plt.ylabel('Max Altitude (meters)')
plt.title('Max Altitude vs Number of Users')
plt.grid(True)
plt.legend()
plt.show()

# Calculate how many users each UAV is covering
uav_coverage_count = np.sum(user_assignments, axis=1)

# Print the number of users covered by each UAV
for i, coverage in enumerate(uav_coverage_count):
    print(f"UAV {i} is covering {int(coverage)} users")
# Additional plotting for only 100 users
num_users =100

# Calculate how many users each UAV is covering
uav_coverage_count = np.sum(user_assignments, axis=1)

# Print the number of users covered by each UAV
for i, coverage in enumerate(uav_coverage_count):
    print(f"UAV {i} is covering {int(coverage)} users")
    
num_uavs, user_locations, centroids, optimal_positions, optimal_altitudes, user_assignments, original_uav_indices, alt_min, alt_max, C, connectivity_links = uav_deployment_optimization(
    num_users, area_size)

if optimal_positions:
    uav_index_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(original_uav_indices)}

    plt.figure(figsize=(10, 10))
    plt.scatter(user_locations[:, 0], user_locations[:, 1], c='blue', label='Users')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    plt.title('User Distribution and Cluster Centroids (100 Users)')
    plt.xlabel('X coordinate (meters)')
    plt.ylabel('Y coordinate (meters)')
    plt.xlim(0, area_size)
    plt.ylim(0, area_size)
    plt.legend()
    plt.grid(True)
    plt.show()

    x_centroids = centroids[:, 0]
    y_centroids = centroids[:, 1]
    z_alt_min = np.full(len(centroids), alt_min)
    z_alt_max = np.full(len(centroids), alt_max)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_centroids, y_centroids, z_alt_min, c='blue', label='Centroids (min altitude)', marker='x')
    ax.scatter(x_centroids, y_centroids, z_alt_max, c='red', label='Centroids (max altitude)')

    for i in range(len(centroids)):
        ax.plot([x_centroids[i], x_centroids[i]], [y_centroids[i], y_centroids[i]], [alt_min, alt_max], c='red')

    ax.set_xlabel('Distance (meters)')
    ax.set_ylabel('Distance (meters)')
    ax.set_zlabel('Altitude (meters)')
    ax.set_title('Continuous Range of Altitude Values for the Given Centroids (100 Users)')
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_zlim(0, alt_max)
    ax.set_xticks(np.arange(0, area_size + 1, 100))
    ax.set_yticks(np.arange(0, area_size + 1, 100))
    ax.set_zticks(np.arange(0, alt_max + 1, 50))
    ax.invert_xaxis()
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for idx, (pos, alt) in enumerate(zip(optimal_positions, optimal_altitudes)):
        ax.scatter(pos[0], pos[1], alt, c='red')
        ax.text(pos[0], pos[1], alt, f'UAV {idx}', color='black')
        ax.scatter(user_locations[:, 0], user_locations[:, 1], c='blue')
        ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [0, alt], c='black', linestyle='--')

    for i in range(len(optimal_positions)):
        for j in range(num_users):
            if user_assignments[i, j] == 1:
                uav_pos = optimal_positions[i]
                user_pos = user_locations[j]
                ax.plot([uav_pos[0], user_pos[0]], [uav_pos[1], user_pos[1]], [optimal_altitudes[i], 0], c='black', linestyle='--')

    ax.set_xlabel('Distance (meters)')
    ax.set_ylabel('Distance (meters)')
    ax.set_zlabel('Altitude (meters)')
    ax.set_title('Optimal Altitudes for UAV Deployment (100 Users)')
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_zlim(0, alt_max)
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.scatter(user_locations[:, 0], user_locations[:, 1], c='blue', label='Users')
    plt.scatter([pos[0] for pos in optimal_positions], [pos[1] for pos in optimal_positions], c='red', marker='x', s=100, label='UAVs')
    plt.title('Connectivity Among the UAVs (100 Users)')
    plt.xlabel('X coordinate (meters)')
    plt.ylabel('Y coordinate (meters)')
    plt.xlim(0, area_size)
    plt.ylim(0, area_size)
    plt.legend()
    plt.grid(True)

    for i in range(len(optimal_positions)):
        for j in range(num_users):
            if user_assignments[i, j] == 1:
                uav_pos = optimal_positions[i]
                user_pos = user_locations[j]
                plt.plot([uav_pos[0], user_pos[0]], [uav_pos[1], user_pos[1]], c='black', linestyle='--')

    for i in range(len(optimal_positions)):
        for k in range(i + 1, len(optimal_positions)):
            if C[original_uav_indices[i], original_uav_indices[k]].varValue == 1:
                plt.plot([optimal_positions[i][0], optimal_positions[k][0]], [optimal_positions[i][1], optimal_positions[k][1]], c='green', linestyle='--')

    plt.show()
else:
    print("No optimal positions found. Problem is infeasible.")
