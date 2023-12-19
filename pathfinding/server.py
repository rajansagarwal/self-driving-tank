import numpy as np
import tensorflow as tf
import requests
import time
import csv
import pandas as pd

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def calculate(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

num_samples = 100

# Generating random distances
forward_distances = np.random.uniform(1, 10, num_samples)
right_distances = np.random.uniform(1, 5, num_samples)
left_distances = np.random.uniform(1, 5, num_samples)

# Read data from a CSV file
data = pd.read_csv('edge_distances.csv') 
distances = data[['Distance Left', 'Distance Forwards', 'Distance Right']].values
move_choice = data['Move Choice'].values

# Convert move choice to one-hot encoding
num_classes = 3
move_choice_one_hot = tf.keras.utils.to_categorical(move_choice, num_classes)

# Define a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(5)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural network model
model.fit(distances, move_choice_one_hot[:, :3], epochs=20, batch_size=32)

# Function to send a POST request
def send_post_request(action, scaled, distance):
    url = "https://8f00-72-139-206-147.ngrok-free.app/"
    endpoints = ["/forward", "/right", "/left"]
    endpoint = endpoints[action] if action < len(endpoints) else "/default"
    full_url = url + endpoint
    params = {'speed': '0.95'} 
    requests.post(full_url, params=params)
    time.sleep(scaled)

# Some arbitrary distances for prediction
distances = np.array([[1.6, 1.2, 4.8], [7.1, 5.5, 4.2]])
predictions = model.predict(distances)
    
Kp = 0.1
Ki = 0.01
Kd = 0.01
pid_controller = PIDController(Kp, Ki, Kd)

for i, pred in enumerate(predictions):
    move_forward, rotate_left, rotate_right, angle, distance = pred
    action = np.argmax([move_forward, rotate_left, rotate_right])
    scaled_delay = int(distance)
    error = distance - scaled_delay

    pid_output = pid_controller.calculate(error, dt=1) 

    scaled_delay += pid_output

    # Adjust delays based on actions
    if action == 0:
        scaled_delay *= 2
    elif action == 1:
        scaled_delay /= 4
    else:
        scaled_delay /= 4

    send_post_request(action, scaled_delay, distance)
    time.sleep(scaled_delay)

time.sleep(3)
requests.post("https://4a13-208-98-222-1.ngrok-free.app/stop")