# Autonomous Tank for Food Delivery

<img width="750" alt="image" src="https://github.com/rajanwastaken/self-driving-tank/assets/64426829/23c47253-1333-428f-af3e-79272d888cf4">

Software Engineering 1A Design Project @ University of Waterloo.

Video Demo: https://www.youtube.com/watch?v=lg0fSsBsl1k&ab_channel=alicezhao

We challenged ourselves to create an autonomous model tank that, upon receiving instructions to travel from Point A to Point B on the University of Waterloo’s campus, would use machine learning and computer vision to independently navigate to its destination and deliver a small, meal-sized parcel.

To identify walking paths, we first tried accessing existing walking path maps created by the university, but they were too generalized. As such, we could not accurately identify reference coordinates from these imprecise maps. Thus, we began by using OpenCV to apply filtering functions that converted a screenshot of Waterloo’s campus on Google Maps to a black and white image of purely walking paths. Each black pixel (or clickable point) on Google Maps is then converted into a node, creating multiple paths of thousands of nodes and edges.

*Figure 1. Walking Path Post-filtering Function, and Black Pixel to Node Conversion*

<img width="750" alt="image" src="https://github.com/rajanwastaken/self-driving-tank/assets/64426829/920c3407-55ad-4015-b66a-7c49fb39767f">

We use Dijkstra’s algorithm to take these nodes and map out the shortest path whenever the system is given a source and a target. A neural network, using the open-source NEAT algorithm, simulates countless cars traveling this path with Pygame, eventually training the neural network to understand what movement to use when it is a certain distance away from walls in three directions. 

*Figure 2. Shortest Distance Between Points, and Simulation on Shortest Path Generation*

<img width="750" alt="image" src="https://github.com/rajanwastaken/self-driving-tank/assets/64426829/801ee9e0-9de6-4ea3-adf9-fb039c66dd56">

On the hardware end, we constructed a model tank. We attached DC motors to two L298N motor controllers, which each connected to a 9V power supply. We then routed these motors to the Pi. Our processing, which would likely cause the Pi to crash, is instead run on a server. We used a Python Flask microservice bridging the aforementioned path algorithms and the Raspberry Pi, which sends endpoints to the Pi in the form of GPIO Zero commands in real-time. The endpoints, /forward, /backwards, /left, /right, and /stop, are processed by an ngrok server and direct voltage towards the respective DC motors.

*Figure 3. Completed Tank*

<img width="750" alt="image" src="https://github.com/rajanwastaken/self-driving-tank/assets/64426829/8f974034-c8db-4d78-b1a7-4adb6f332a46">

Our server processes the required information in two ways. First, a phone camera streams video to our laptop through OBS, which is processed by our algorithms at 30 frames per second. These frames undergo a sequence of image transformations, using varying hue, value, and contrast thresholds to isolate paths from its surroundings (e.g., grass). This processing enables us to generate Hough lines, in which we can project three lines: straight, 45° left, and 45° right. We then interpreted the depth, which was accomplished by creating reference images for how distances are perceived from our camera’s perspective 15 cm above the ground. 

*Figure 4. OpenCV Image Processing*

<img width="750" alt="image" src="https://github.com/rajanwastaken/self-driving-tank/assets/64426829/56dff0c9-26b6-4e7b-8906-b0410b5a2230">

*Figure 5. Screenshot from Real-time Distance Analysis and Hough Line Path Detection*

<img width="750" alt="image" src="https://github.com/rajanwastaken/self-driving-tank/assets/64426829/4ae15ae1-acdd-4f05-bfcf-702b79edf6c0">

The second source of information for our server is the ultrasonic sensor. An endpoint on the Pi, /distance, returns the distance to an object ahead of the Tank at that moment in time. This value is calculated by sending an ultrasonic signal and calculating the time for it to return, using existing libraries with GPIO Zero.

Instead of using a GPS for distance, we chose to analyze the Rotary PCB Encoders, which use magnets to calculate the number of rotations. When we send motor instructions in a specific direction, we can calculate the net distance traveled towards our final destination. We can then add weightings to certain directions, influencing the pathfinding to stay on track in a specific direction. During experimentation, we found that Apple’s public GPS distances have a significant margin of error (±100 metres), so this weighting estimation would not result in a significant loss. Although we were concerned that the tank would not consistently stop in the same place, we were willing to sacrifice slight inconsistencies in the endpoint to ensure that the tank stayed on path. We noticed drastic turns often led to small inconsistencies, but worked well for simpler paths like V1 greens.

Our model, which is trained on the simulation data, receives the four parameters (three distances and ultrasonic distance), allowing it to scale the tank’s speed and size relative to the real-life walking path. 
Realtime inputs enabled us to use PID controllers to evaluate error, and send adjustments; this was critical due to the inconsistencies with the brushed DC motor we were given. The model and feedback loops were an interesting challenge, only made possible with conversations and teachings from Mechatronics upper years, who worked at Tesla and AI companies. With the inputs that it receives from the path detection and ultrasonic sensor, the model can iteratively send post requests to our Pi’s ngrok server for a certain number of seconds, operating in real-time.
