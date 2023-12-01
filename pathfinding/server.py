import numpy as np
import tensorflow as tf
import requests
import time

num_samples = 1000

forward_distances = np.random.uniform(1, 10, num_samples)
right_distances = np.random.uniform(1, 5, num_samples)
left_distances = np.random.uniform(1, 5, num_samples)

instructions = []
for i in range(num_samples):
    if forward_distances[i] >= right_distances[i] and forward_distances[i] >= left_distances[i]:
        instructions.append([1, 0, 0, 0, 0])  
    elif right_distances[i] >= left_distances[i]:
        instructions.append([0, 0, 1, 15, 30])
    else:
        instructions.append([0, 1, 0, 10, 20])

forward_distances = np.array(forward_distances).reshape(-1, 1)
right_distances = np.array(right_distances).reshape(-1, 1)
left_distances = np.array(left_distances).reshape(-1, 1)
instructions = np.array(instructions)

distances = np.concatenate((forward_distances, right_distances, left_distances), axis=1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(5)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(distances, instructions, epochs=20, batch_size=32)

def send_post_request(action, scaled, distance):
    url = "https://8f00-72-139-206-147.ngrok-free.app/"
    if action == 0:
        endpoint = "/forward"
    elif action == 1:
        endpoint = "/right"
    else:
        endpoint = "/left"

    full_url = url + endpoint
    params = {'speed': '0.95'} 
    requests.post(full_url, params=params)
    time.sleep(scaled)

distances = np.array([[1.6, 1.2, 4.8], [7.1, 5.5, 4.2]])
predictions = model.predict(distances)
    
for i, pred in enumerate(predictions):
    move_forward, rotate_left, rotate_right, angle, distance = pred
    action = np.argmax([move_forward, rotate_left, rotate_right])

    scaled_delay = int(distance)

    if action == 0:
        scaled_delay *= 2
        # print(f"Distances: {distances[i]}, Instruction: Move Forward for {distance} seconds")
    elif action == 1:
        scaled_delay /= 4
        # print(f"Distances: {distances[i]}, Instruction: Rotate Left by {angle} degrees, then Move Forward for {distance} seconds")
    else:
        scaled_delay /= 4
        # print(f"Distances: {distances[i]}, Instruction: Rotate Right by {angle} degrees, then Move Forward for {distance} seconds")

    send_post_request(action, scaled_delay, distance)

time.sleep(3)
requests.post("https://4a13-208-98-222-1.ngrok-free.app/stop")