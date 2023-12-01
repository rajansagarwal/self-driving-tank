import sys, math, pygame, neat

pygame.init()
pygame.display.set_caption("Simulating BOAT")
WINDOW_SIZE = 1280, 720
SCREEN = pygame.display.set_mode(WINDOW_SIZE)

BOAT_SIZE = 6, 10
BOAT_CENTER = 80, 80
VELOCITY = 7
ROTATION = 10
WHITE = (255, 255, 255, 255)
ROUTE = pygame.image.load('track3.png').convert_alpha()
ROUTE_COPY = ROUTE.copy()
FONT = pygame.font.SysFont("arial", 25)
CLOCK = pygame.time.Clock()
GENERATION = 0

def move_point(point, angle, distance):
    radians = math.radians(angle)
    return int(point[0] + distance * math.cos(radians)), int(point[1] + distance * math.sin(radians))

def find_start(surface):
    for x in range(surface.get_width()):
        for y in range(surface.get_height()):
            if surface.get_at((x, y)) == (255, 0, 0, 255):
                return x + 50, y + 50
    return None

start = find_start(ROUTE)

class Boat:
    def __init__(self):
        self.corners, self.edge_points, self.edge_distances = [], [], []
        self.travelled_distance, self.angle = 0, 0
        self.boat_center = start
        self.boat = pygame.transform.scale(pygame.image.load("white.png").convert_alpha(), BOAT_SIZE)
        self.crashed, self.stopped = False, False
        self.step_counter = 0
        self.update_sensor_data()

    def display_boat(self):
        rotated_boat = pygame.transform.rotate(self.boat, self.angle)
        rect = rotated_boat.get_rect(center=self.boat_center)
        SCREEN.blit(rotated_boat, rect.topleft)

    def crash_check(self):
        return any(ROUTE.get_at(corner) == WHITE for corner in self.corners)

    def update_sensor_data(self):
        angles = [90 - self.angle, 45 - self.angle, 135 - self.angle]
        angles = [math.radians(i) for i in angles]
        edge_points, edge_distances = [], []
        for angle in angles:
            distance = 0
            edge_x, edge_y = self.boat_center
            while ROUTE_COPY.get_at((edge_x, edge_y)) != WHITE:
                edge_x, edge_y = move_point((edge_x, edge_y), angle, distance)
                distance += 1
            edge_points.append((edge_x, edge_y))
            edge_distances.append(distance)
        self.edge_points, self.edge_distances = edge_points, edge_distances

    def display_edge_points(self):
        for point in self.edge_points:
            pygame.draw.line(SCREEN, (0, 255, 0), self.boat_center, point)
            pygame.draw.circle(SCREEN, (0, 255, 0), point, 5)

    def update_position(self):
        self.step_counter += 1
        self.boat_center = move_point(self.boat_center, 90 - self.angle, VELOCITY)
        self.travelled_distance += VELOCITY
        dist = math.sqrt(BOAT_SIZE[0]**2 + BOAT_SIZE[1]**2) / 2
        self.corners = [move_point(self.boat_center, angle - self.angle, dist) for angle in [60, 120, 240, 300]]
        if self.travelled_distance % 20 == 0:
            self.update_sensor_data()

def run(genomes, config):
    global GENERATION
    stopped = False
    GENERATION += 1
    models, boats = [], []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        models.append(net)
        genome.fitness = 0
        boats.append(Boat())

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        running_boats = 0
        SCREEN.blit(ROUTE, (0, 0))
        top_boat, top_velocity, top_angle = None, 0, 0

        for i, boat in enumerate(boats):
            if not boat.crashed and not stopped:
                running_boats += 1
                output = models[i].activate(boat.edge_distances)
                choice = output.index(max(output))
                if choice == 0: boat.angle += ROTATION
                elif choice == 1: boat.angle -= ROTATION
                boat.update_position()
                boat.display_boat()
                boat.crashed = boat.crash_check()
                if boat.travelled_distance % 10 == 0:
                    boat.update_sensor_data()
                genomes[i][1].fitness += boat.travelled_distance
                boat.display_edge_points()
                if ROUTE.get_at(boat.boat_center) == (0, 255, 0):
                    stopped = True
                    print("Boat reached RGB(0, 255, 0). Game Over.")

        if running_boats == 0:
            return
        SCREEN.blit(FONT.render(f"Generation: {GENERATION}, Running Boats: {running_boats}", True, (0, 0, 0)), (0, 0))
        pygame.display.update()
        CLOCK.tick(10)

neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation)

neat_config.fitness_criterion = 'max'
neat_config.fitness_threshold = 1
neat_config.pop_size = 200
neat_config.reset_on_extinction = True

neat_config.genome_config.activation_default = 'sigmoid'
neat_config.genome_config.activation_mutate_rate = 0.01
neat_config.genome_config.activation_options = 'tanh'

population = neat.Population(neat_config)
population.add_reporter(neat.StdOutReporter(True))
population.add_reporter(neat.StatisticsReporter())
population.run(run, 500)