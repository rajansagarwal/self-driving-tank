from flask import Flask, request
from gpiozero import DistanceSensor
import RPi.GPIO as GPIO

app = Flask(__name__)

GPIO.setmode(GPIO.BCM)
motor_pins = {
    'forward': [[13, 17], [23, 24]],
    'left': [[13, 0], [23, 24]],
    'right': [[0, 17], [24, 23]],
    'stop': [[0, 0], [0, 0]]
}

for pins in motor_pins.values():
    for pin in pins:
        if 0 not in pin:
            GPIO.setup(pin, GPIO.OUT)

def move_motors(pins):
    for idx, motor in enumerate(pins):
        if 0 not in motor:
            for pin in motor:
                GPIO.output(pin, GPIO.HIGH if idx == 0 else GPIO.LOW)

sensor = DistanceSensor(echo=23, trigger=4)

@app.route('/forward', methods=['POST'])
def forward():
    move_motors(motor_pins['forward'])
    return "Moving forward"

@app.route('/left', methods=['POST'])
def left():
    move_motors(motor_pins['left'])
    return "Moving left"

@app.route('/right', methods=['POST'])
def right():
    move_motors(motor_pins['right'])
    return "Moving right"

@app.route('/stop', methods=['POST'])
def stop():
    move_motors(motor_pins['stop'])
    return "Stopping"

@app.route('/distance', methods=['GET'])
def distance():
    return f"Distance: {sensor.distance} meters"

if __name__ == '__main__':
    app.run(debug=True)