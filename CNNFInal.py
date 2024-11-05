#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:22:17 2024

@author: sheshta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import random

# Define constants
MAX_UAVS = 20
GRID_SIZE = 200
REFERENCE_ALTITUDE = 100
THETA = np.deg2rad(45)
TIME_STEPS = 20

# Function to calculate coverage radius based on altitude
def calculate_coverage_radius(altitude, theta=THETA):
    return altitude / np.tan(theta)

# Simulate random-waypoint mobility model for users
def simulate_random_waypoint_mobility(sensor_coords, grid_size, num_steps, max_speed):
    mobility_steps = [sensor_coords]
    for step in range(1, num_steps):
        new_positions = []
        for (x, y) in mobility_steps[-1]:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0, max_speed)
            new_x = np.clip(x + speed * np.cos(angle), 0, grid_size)
            new_y = np.clip(y + speed * np.sin(angle), 0, grid_size)
            new_positions.append([new_x, new_y])
        mobility_steps.append(np.array(new_positions))
    return mobility_steps

# Load and process sensor and drone data
def load_sensors_and_drones_data(sensors_file, drones_file):
    sensors_df = pd.read_csv(sensors_file)
    drones_df = pd.read_csv(drones_file)
    user_grids = []
    y_positions = []  # Move this up
    all_sensor_coords = []
    all_drone_positions = []

    num_scenarios = min(len(sensors_df) // 100, len(drones_df) // MAX_UAVS)

    for i in range(num_scenarios):
        initial_coords = sensors_df[['Xs', 'Ys']].iloc[i * 100: (i + 1) * 100].values
        mobility_steps = simulate_random_waypoint_mobility(initial_coords, GRID_SIZE, TIME_STEPS, max_speed=5)

        # Create the grid representation of user movement
        time_step_grids = []
        for coords in mobility_steps:
            grid = np.zeros((GRID_SIZE, GRID_SIZE))
            for x, y in coords:
                x = np.clip(int(x), 0, GRID_SIZE - 1)
                y = np.clip(int(y), 0, GRID_SIZE - 1)
                grid[x, y] += 1
            time_step_grids.append(grid)
        user_grids.append(np.stack(time_step_grids, axis=-1))
        all_sensor_coords.append(mobility_steps)

        # Process the drone data for this scenario
        drone_positions = drones_df[['Xd', 'Yd', 'Ad']].iloc[i * MAX_UAVS: (i + 1) * MAX_UAVS].values
        if len(drone_positions) == MAX_UAVS:
            # Normalize positions and altitudes
            drone_positions_normalized = np.copy(drone_positions)
            drone_positions_normalized[:, :2] /= GRID_SIZE
            drone_positions_normalized[:, 2] /= 100
            coverage_radii = np.array([calculate_coverage_radius(a) for a in drone_positions[:, 2]])
            coverage_radii_normalized = coverage_radii / calculate_coverage_radius(REFERENCE_ALTITUDE)
            combined_features = np.concatenate([drone_positions_normalized.flatten(), coverage_radii_normalized])
            y_positions.append(combined_features)  # Only append valid drone data
            all_drone_positions.append(drone_positions)

    return np.array(user_grids), np.array(y_positions), all_sensor_coords, all_drone_positions


# Define the paths to the sensor and drone data files
sensors_file = '/content/drive/MyDrive/2500 Scenarios/2500SensorsList_p22.csv'
drones_file = '/content/drive/MyDrive/2500 Scenarios/2500DronesList_p22.csv'
# Load datasets
user_grids, y_positions, all_sensor_coords, all_drone_positions = load_sensors_and_drones_data(sensors_file, drones_file)


# Now user_grids and y_positions should have consistent lengths
X_train, X_temp, y_train, y_temp = train_test_split(user_grids, y_positions, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define CNN model
def create_cnn_model(input_shape, output_shape_positions):
    input_grid = Input(shape=input_shape, name="input_grid")
    x = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(input_grid)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    positions_output = Dense(output_shape_positions, activation='linear', name='uav_positions')(x)
    lr_schedule = ExponentialDecay(0.001, decay_steps=500, decay_rate=0.96, staircase=True)
    model = Model(inputs=[input_grid], outputs=positions_output)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mse', metrics=['mae'])
    return model

# Define model
input_shape = (GRID_SIZE, GRID_SIZE, TIME_STEPS)
output_shape_positions = y_train.shape[1]
model = create_cnn_model(input_shape, output_shape_positions)

# Train model
#early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64)#, callbacks=[early_stopping])

# Plot training and validation loss and accuracy
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Iterations')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Mean Absolute Error (MAE)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error vs. Iterations')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.show()

plot_history(history)

# Extract and adjust predictions
def extract_and_adjust_predictions(predictions):
    predicted_positions = predictions[:, :MAX_UAVS * 2] * GRID_SIZE
    predicted_altitudes = predictions[:, MAX_UAVS * 2:MAX_UAVS * 3] * 100
    predicted_radii = np.array([calculate_coverage_radius(a) for a in predicted_altitudes.flatten()]).reshape(predicted_altitudes.shape)
    return predicted_positions, predicted_altitudes, predicted_radii

# Predict UAV positions and radii
predictions = model.predict(X_test)
predicted_positions, predicted_altitudes, predicted_radii = extract_and_adjust_predictions(predictions)

# Calculate number of covered users for each scenario
def calculate_covered_users(predicted_positions, predicted_radii, sensor_coords):
    covered_users_set = set()
    for (x, y, r) in zip(predicted_positions[::2], predicted_positions[1::2], predicted_radii.flatten()):
        for i, (ux, uy) in enumerate(sensor_coords):
            if (ux - x) ** 2 + (uy - y) ** 2 <= r ** 2:
                covered_users_set.add(i)
    return len(covered_users_set)

covered_users_list = []
for i in range(len(X_test)):
    predicted_positions_sample = predicted_positions[i]
    predicted_radii_sample = predicted_radii[i]
    sensor_coords = all_sensor_coords[i][0]
    covered_users = calculate_covered_users(predicted_positions_sample, predicted_radii_sample, sensor_coords)
    covered_users_list.append(covered_users)

# Sort and calculate CDF of covered users
covered_users_sorted = np.sort(covered_users_list)
cdf = np.arange(1, len(covered_users_sorted) + 1) / float(len(covered_users_sorted))

# Plot the CDF of covered users based on CNN model predictions
plt.figure(figsize=(10, 6))
plt.step(covered_users_sorted, cdf, where='post', linestyle='-', color='red', label='CNN Model Coverage')
plt.title('CDF of Covered Users (CNN Model Predictions)')
plt.xlabel('Number of Covered Users')
plt.ylabel('CDF')
plt.grid(True)
plt.xlim(0, 100)  # x-axis to reflect 100 users
plt.ylim(0, 1)    # y-axis to ensure CDF goes up to 1
plt.legend()
plt.show()

# Plot Average Number of Covered Users vs Steps
def plot_avg_covered_users_vs_steps(drone_positions, sensor_coords):
    avg_covered_users = []
    for x, y, a in drone_positions:
        r = calculate_coverage_radius(a)
        covered_users = np.sum((sensor_coords[:, 0] - x) ** 2 + (sensor_coords[:, 1] - y) ** 2 <= r ** 2)
        avg_covered_users.append(covered_users)
    steps = list(range(1, len(avg_covered_users) + 1))
    plt.plot(steps, avg_covered_users, marker='o', linestyle='-')
    plt.title('Average Number of Covered Users vs. Steps')
    plt.xlabel('Steps')
    plt.ylabel('Average Number of Covered Users')
    plt.grid(True)
    plt.show()

plot_avg_covered_users_vs_steps(all_drone_positions[0], all_sensor_coords[0][0])

# 2D coverage scatter plot
def plot_uav_coverage_2d(uav_x_pred, uav_y_pred, uav_r_pred, uav_x_actual, uav_y_actual, user_x, user_y):
    plt.figure(figsize=(12, 8))

    # Plot users
    plt.scatter(user_x, user_y, c='blue', marker='^', s=50, label='Users')

    # Plot predicted UAVs and their coverage areas, with label added only once
    first_pred = True
    for x, y, r in zip(uav_x_pred, uav_y_pred, uav_r_pred):
        circle_pred = plt.Circle((x, y), r, color='red', alpha=0.3)
        plt.gca().add_patch(circle_pred)
        if first_pred:
            plt.scatter(x, y, c='red', marker='x', s=100, label='Predicted UAVs')
            first_pred = False
        else:
            plt.scatter(x, y, c='red', marker='x', s=100, label='_nolegend_')

    plt.scatter(uav_x_actual, uav_y_actual, c='green', marker='o', s=100, label='Actual UAVs')

    plt.title('Predicted vs. Actual UAV Coverage')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

plot_uav_coverage_2d(predicted_positions[0][:MAX_UAVS], predicted_positions[0][MAX_UAVS:], predicted_radii[0],
                     y_test[0][:MAX_UAVS * 2:2] * GRID_SIZE, y_test[0][1:MAX_UAVS * 2:2] * GRID_SIZE,
                     all_sensor_coords[0][0][:, 0], all_sensor_coords[0][0][:, 1])

# 3D UAV position and coverage visualization
def plot_uav_positions_3d(uav_x_pred, uav_y_pred, uav_a_pred, uav_r_pred, uav_x_actual, uav_y_actual, uav_a_actual, user_x, user_y):
    fig = plt.figure(figsize=(14, 7))

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(uav_x_pred, uav_y_pred, uav_a_pred, c='r', marker='x', s=100, label='Predicted UAVs')
    ax.scatter(uav_x_actual, uav_y_actual, uav_a_actual, c='g', marker='o', s=100, label='Actual UAVs')
    ax.scatter(user_x, user_y, np.zeros_like(user_x), c='b', marker='^', s=50, label='Users')  # Add user locations
    ax.set_title('Predicted vs Actual UAV Positions and Altitudes')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Altitude')
    ax.legend()
    ax.grid(True)

    ax2 = fig.add_subplot(122, projection='3d')

    # Plot the predicted UAVs with their coverage areas
    for x, y, a, r in zip(uav_x_pred, uav_y_pred, uav_a_pred, uav_r_pred):
        ax2.scatter(x, y, a, c='r', marker='x', s=100)  # UAV positions
        u = np.linspace(0, 2 * np.pi, 100)
        x_disc = x + r * np.cos(u)
        y_disc = y + r * np.sin(u)
        z_disc = np.zeros_like(x_disc)
        ax2.plot_trisurf(x_disc, y_disc, z_disc, color='r', alpha=0.2)

    ax2.scatter(uav_x_actual, uav_y_actual, np.zeros_like(uav_x_actual), c='g', marker='o', s=100, label='Actual UAVs')
    ax2.scatter(user_x, user_y, np.zeros_like(user_x), c='b', marker='^', s=50, label='Users')
    ax2.set_title('Predicted Coverage Map (Top View)')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_zlabel('Altitude')
    ax2.view_init(elev=90, azim=-90)

    plt.tight_layout()
    plt.show()

plot_uav_positions_3d(predicted_positions[0][:MAX_UAVS], predicted_positions[0][MAX_UAVS:], predicted_altitudes[0],
                      predicted_radii[0], y_test[0][:MAX_UAVS * 2:2] * GRID_SIZE, y_test[0][1:MAX_UAVS * 2:2] * GRID_SIZE,
                      y_test[0][MAX_UAVS * 2:MAX_UAVS * 3] * 100, all_sensor_coords[0][0][:, 0], all_sensor_coords[0][0][:, 1])

# Calculate coverage efficiency
def calculate_coverage_efficiency(covered_users_list, total_users=100):
    coverage_efficiency_list = [(covered_users / total_users) * 100 for covered_users in covered_users_list]
    return coverage_efficiency_list

# Calculate coverage efficiency for each test scenario
coverage_efficiency_list = calculate_coverage_efficiency(covered_users_list)

# Plot the coverage efficiency
plt.figure(figsize=(10, 6))
plt.hist(coverage_efficiency_list, bins=10, color='blue', alpha=0.7, label='Coverage Efficiency')
plt.title('Distribution of Coverage Efficiency Across Test Scenarios')
plt.xlabel('Coverage Efficiency (%)')
plt.ylabel('Number of Scenarios')
plt.grid(True)
plt.legend()
plt.show()

# Display average coverage efficiency
avg_coverage_efficiency = np.mean(coverage_efficiency_list)
print(f"Average Coverage Efficiency: {avg_coverage_efficiency:.2f}%")