# Efficient-Unmanned-Aerial-Vehicles-UAVs-Deployment-Wireless-Network-Using-Machine-Learning

Project Overview

This project addresses the challenge of deploying UAVs to provide emergency wireless network coverage in resource-constrained scenarios, such as disaster zones or remote areas where traditional base stations are infeasible. By integrating Mixed Integer Programming (MIP) and Machine Learning (ML) techniques, we aim to optimize the positioning and altitude of UAVs to maximize user coverage and maintain efficient communication.

Problem Statement

In scenarios like disaster-stricken areas or large public gatherings, UAVs serve as mobile base stations, extending network coverage where terrestrial networks are unavailable or too costly. The goal is to determine the optimal number and positions of UAVs to deliver consistent, high-quality coverage under constraints such as limited resources, obstructions, and unpredictable user distributions.

Aims and Objectives

	•	Objective 1: Optimize 3D placement of UAVs by considering factors like height, path loss, and Line-of-Sight (LoS) and Non-Line-of-Sight (NLoS) components.
	•	Objective 2: Simulate deployments for varying numbers of users (100 to 800) in disaster scenarios to determine the minimum required UAVs.
	•	Objective 3: Expand existing research by applying models to large-scale networks.
	•	Objective 4: Generate synthetic training data using optimization models to train ML algorithms for real-time UAV positioning.
	•	Objective 5: Develop a Convolutional Neural Network (CNN) model to predict optimal UAV positions, altitudes, and coverage radii.

Methodology

The project combines several methodologies:

	1.	KMeans Clustering: Used to group users and define optimal deployment points for UAVs.
	2.	Greedy Post-Processing Algorithm: Identifies and removes the least impactful UAVs while ensuring user coverage, minimizing the deployment count.
	3.	Machine Learning Model: A CNN is trained with synthetic data to predict UAV positions and altitudes based on user distribution.

Data Generation

Synthetic datasets with 2500 scenarios were created using an optimization model. Each scenario includes user distributions and the corresponding optimal UAV locations and altitudes, serving as training data for the CNN model.

Key Algorithms

	•	KMeans Clustering: Segments users into clusters to aid in UAV positioning.
	•	CNN: Predicts UAV placement in real-time based on spatial data.
	•	Greedy Algorithm: Reduces UAV count by iteratively removing those with minimal user impact.

Results and Key Findings

	•	Optimized UAV Deployment: Significant reduction in the number of UAVs needed for effective coverage compared to previous methods.
	•	High Model Accuracy: The CNN model achieved strong performance with low Mean Absolute Error (MAE), indicating accurate predictions of UAV positions.
	•	Enhanced Connectivity: Additional constraints in the model provided balanced network coverage and connectivity.

Contributions

	1.	Introduced new connectivity constraints to improve UAV network reliability.
	2.	Created a synthetic dataset for ML-based UAV placement, addressing the lack of real-world data.
	3.	Developed a CNN model for real-time UAV positioning, which achieved high predictive accuracy.
	4.	Expanded the deployment model to simulate large-scale disaster scenarios effectively.

Limitations and Future Work

	•	Limitations:
	•	Computational constraints limited the dataset size and grid resolution.
	•	Simplistic mobility model and fixed altitude-angle assumptions.
	•	Future Directions:
	•	Explore advanced ML techniques, such as reinforcement learning, to enhance deployment efficiency.
	•	Improve the dataset with real-world data for better generalization.

How to Run the Code

	1.	Clone the repository.
	2.	Install the necessary dependencies (pip install -r requirements.txt).
	3.	Run KmeansV6.py for clustering-based positioning.
	4.	Execute CNNFinal.py to train or test the CNN model for UAV placement predictions.

References

	•	Vergara, M., Ramos, L., & Rivera-Campoverde, N. D. (2023). EngineFaultDB: A Novel Dataset for Automotive Engine Fault Classification and Baseline Results. IEEE Access, 11, 126155-126171.
	•	Indu, C., & Vipin, K. (2024). 3D Deployment of Multi-UAV Network in Disaster Zones. IEEE ICC ROBINS 2024.

This README provides a comprehensive overview of the project, methodology, results, and future scope. It’s designed to make the project approachable for new users while providing in-depth details for technical audiences. Let me know if you’d like any adjustments.
