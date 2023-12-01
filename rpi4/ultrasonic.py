from gpiozero import DistanceSensor

sensor = DistanceSensor(echo=23, trigger=4)

while True:
    print(sensor.distance)