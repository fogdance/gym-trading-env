import pygame
from track import Track
from car import Car
from controller import Controller
from utils import load_csv_data

class Game:
    def __init__(self, csv_path):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('Forex Racer')
        self.clock = pygame.time.Clock()

        data = load_csv_data(csv_path)
        self.track = Track(data)
        self.car = Car()
        self.controller = Controller()

    def run_human_mode(self):
        running = True
        while running:
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE]:
                        action = self.controller.get_human_action()

            if action is not None:
                self.car.update(action, self.track.current_segment())
                self.track.move_next()

            self.render()
            pygame.display.flip()
            self.clock.tick(60)

    def render(self):
        self.screen.fill((0, 0, 0))
        self.track.draw(self.screen, car_position=self.car.position, car_profit=self.car.profit)
        self.car.draw(self.screen)
        # 仪表盘绘制（盈亏、仓位）留空待实现
