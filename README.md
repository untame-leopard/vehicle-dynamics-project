Vehicle Dynamics Project: Quarter-Car Model

Project Overview

This project is an exploration into vehicle dynamics using a simple quarter-car model. The goal is to simulate how a vehicle's suspension system responds to different road inputs, such as bumps. This model helps to understand the fundamental physics of suspension design by focusing on just one corner of the vehicle.

The simulation uses Python to solve the second-order differential equations that govern the motion of the sprung and unsprung masses.

Model Description
The quarter-car model simplifies a vehicle into a system with two masses and two springs.

Sprung Mass (
boldsymbolm_s): Represents the mass of the car body, passenger, and cargo.

Unsprung Mass (
boldsymbolm_u): Represents the mass of the wheel, tire, and suspension components.

Suspension Spring (
boldsymbolk_s): The main spring that connects the sprung and unsprung masses.

Damper (
boldsymbolc_s): The shock absorber that dissipates energy and controls the suspension's motion.

Tire Stiffness (
boldsymbolk_t): The stiffness of the tire itself, modeled as a second spring.

Getting Started
To run this simulation, you'll first need to set up the Python environment and install the required libraries.

Clone the repository:

git clone https://github.com/untame-leopard/vehicle-dynamics-project.git

Then:

cd vehicle-dynamics-project

Activate the virtual environment:

source .venv/bin/activate

Install dependencies: Use the requirements.txt file to install the necessary libraries.

pip install -r requirements.txt

Run the simulation: Execute the main Python script.

python3 src/main.py

Dependencies
This project relies on the following Python libraries:

numpy: For efficient numerical computations.

scipy: For solving the differential equations of the model.

matplotlib: For plotting and visualizing the simulation results.

Expected Output
When you run the script, a plot will be generated showing the displacement of the sprung and unsprung masses over time after the vehicle hits a bump. This visualization helps in understanding the dynamic behavior of the suspension system.