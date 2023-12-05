import sys
import math
import pygame
import neat
import csv

pygame.init()
pygame.display.set_caption("Training BOAT")
WINDOW_SIZE = 1280, 720
SCREEN = pygame.display.set_mode(WINDOW_SIZE)

CAR_SIZE = 6, 10
CAR_CENTER = 80, 80
SPEED = 5
ROTATION = 10
WHITE = (255, 255, 255, 255)
MAP = pygame.image.load('tracks/track3.png').convert_alpha()
pygame.display.set_caption("Training BOAT")
FONT = pygame.font.SysFont("arial", 15)
CLOCK = pygame.time.Clock()
GENERATION = 0

def translate_kinda_sorta(point, angle, distance):
    radians = math.radians(angle)
    return int(point[0] + distance * math.cos(radians)),\
        int(point[1] + distance * math.sin(radians))

def start_here(track_surface):
    for x in range(track_surface.get_width()):
        for y in range(track_surface.get_height()):
            if track_surface.get_at((x, y)) == (255, 0, 0, 255):
                return x + 50, y + 50
    return None

class Car:
    def __init__(self):
        self.corners = []
        self.edge_points = []
        self.edge_distances = []
        self.travelled_distance = 0
        self.angle = 0
        self.car_center = start_here(MAP)
        self.car = pygame.image.load("white.png").convert_alpha()
        self.car = pygame.transform.scale(self.car, CAR_SIZE)
        self.crashed = False
        self.update_sensor_data()
        self.stopped = False

    def display_car(self):
        rotated_car = pygame.transform.rotate(self.car, self.angle)
        rect = rotated_car.get_rect(center=self.car_center)
        SCREEN.blit(rotated_car, rect.topleft)

    def crash_check(self):
        for corner in self.corners:
            if MAP.get_at(corner) == WHITE:
                return True
        return False

    def update_sensor_data(self):
        angles = [90 - self.angle, 45 - self.angle, 135 - self.angle]
        angles = [math.radians(i) for i in angles]
        edge_points = []
        edge_distances = []
        for angle in angles:
            distance = 0
            edge_x, edge_y = self.car_center
            while MAP.get_at((edge_x, edge_y)) != WHITE:
                edge_x = int(self.car_center[0] + distance * math.cos(angle))
                edge_y = int(self.car_center[1] + distance * math.sin(angle))
                distance += 1
            edge_points.append((edge_x, edge_y))
            edge_distances.append(distance)
        self.edge_points = edge_points
        self.edge_distances = edge_distances

    def display_edge_points(self):
        for point in self.edge_points:
            pygame.draw.line(SCREEN, (0, 255, 0), self.car_center, point)
            pygame.draw.circle(SCREEN, (0, 255, 0), point, 5)

    def update_position(self):
        self.car_center = translate_kinda_sorta(
            self.car_center, 90 - self.angle, SPEED)
        self.travelled_distance += SPEED
        dist = math.sqrt(CAR_SIZE[0]**2 + CAR_SIZE[1]**2)/2
        corners = []
        corners.append(translate_kinda_sorta(
            self.car_center, 60 - self.angle, dist))
        corners.append(translate_kinda_sorta(
            self.car_center, 120 - self.angle, dist))
        corners.append(translate_kinda_sorta(
            self.car_center, 240 - self.angle, dist))
        corners.append(translate_kinda_sorta(
            self.car_center, 300 - self.angle, dist))
        self.corners = corners

def run(genomes, config):
    global GENERATION
    stopped = False
    GENERATION += 1
    models = []
    cars = []
    top_cars_info = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        models.append(net)
        genome.fitness = 0
        cars.append(Car())

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        running_cars = 0
        move_instructions = []

        SCREEN.blit(MAP, (0, 0))

        top_car = None
        top_velocity = 0
        top_angle = 0

        def save_instructions_to_file(instructions):
            with open("move_instructions.txt", "w") as file:
                file.write("Move Instructions with Angles:\n")
                for i, move in enumerate(instructions):
                    file.write(f"Car: {i}, Angle: {move[1]}°, Edge: {move[2][0]}\n")

        for i, car in enumerate(cars):
            if not car.crashed and not stopped:
                running_cars += 1
                output = models[i].activate(car.edge_distances)
                choice = output.index(max(output))
                if choice == 0:
                    car.angle += ROTATION
                elif choice == 1:
                    car.angle -= ROTATION
                car.update_position()
                car.display_car()
                car.crashed = car.crash_check()
                car.update_sensor_data()
                genomes[i][1].fitness += car.travelled_distance
                car.display_edge_points()

                # if MAP.get_at(car.car_center) == (0, 255, 0):
                #     stopped = True
                #     print("Car reached RGB(0, 255, 0). Game Over.")
                #     save_instructions_to_file(move_instructions)
                    
                if top_car is None or car.travelled_distance > top_car.travelled_distance:
                    top_car = car
                    top_velocity = SPEED
                    top_angle = car.angle
                    
                if top_car is not None:
                    move_instructions.append((top_velocity, top_angle, car.edge_distances))

                if len(top_cars_info) < 10:
                    top_cars_info.append({
                        'velocity': top_velocity,
                        'angle': top_angle,
                        'distances': car.edge_distances,
                        'choice': choice
                    })

            save_instructions_to_file(move_instructions)

        with open("edge_distances.csv", "a", newline='') as file:
            writer = csv.writer(file)

                # Write header if the file is empty
            if file.tell() == 0:
                writer.writerow(['Car', 'Distance Left', 'Distance Forwards', 'Distance Right', 'Move Choice', 'Velocity(px)', 'Angle(deg)'])

            for i, car_info in enumerate(top_cars_info):
                if (i + 1) % 40 == 0:
                    top_cars_info = []
                    break
                writer.writerow([
                    f"Car {i + 1}",
                    car_info['distances'][0],
                    car_info['distances'][1],
                    car_info['distances'][2],
                    car_info['choice'],
                    car_info['velocity'],
                    car_info['angle']
                ])

        
        info_text = f"Top Car - Velocity: {top_velocity}px, Angle: {top_angle}°"
        info_render = FONT.render(info_text, True, (0, 0, 0))
        SCREEN.blit(info_render, (0, 30))

        if running_cars == 0:
            return
        
        msg = "Generation: {}, Running Cars: {}".format(GENERATION, running_cars)
        text = FONT.render(msg, True, (0, 0, 0))
        SCREEN.blit(text, (0, 0))
        pygame.display.update()
        CLOCK.tick(10)

neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 "config.txt")
population = neat.Population(neat_config)
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.run(run, 500)