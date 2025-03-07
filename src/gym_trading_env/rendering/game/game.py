import pygame
import numpy as np
import torch
from .track import Track
from .car import Car
from .controller import Controller
from torchvision.transforms import Grayscale


# 屏幕尺寸
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# 配置比例（左侧:右侧 = 4:6）
LEFT_WIDTH = SCREEN_WIDTH * 4 // 10  # 320像素
RIGHT_WIDTH = SCREEN_WIDTH * 6 // 10  # 480像素

TOP_HEIGHT = SCREEN_HEIGHT * 9 // 10
BOTTOM_HEIGHT = SCREEN_HEIGHT // 10

class Game:
    def __init__(self, size):
        self.df = None
        pygame.init()
        self.train_size = size
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Forex Racer')
        self.clock = pygame.time.Clock()

        self.track = Track((LEFT_WIDTH, 0, RIGHT_WIDTH, TOP_HEIGHT))
        self.car = Car((LEFT_WIDTH, 0, RIGHT_WIDTH, TOP_HEIGHT))
        self.controller = Controller()
        self.gs = Grayscale(num_output_channels=1)

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


    def step(self, df):
        self.df = df
        self.track.step(df)

    def render(self, position, profit, render_mode='human'):
        self.car.update(positon=position, profit=profit)

        if render_mode == 'human':
            self.screen.fill((0, 0, 0))
        
            # 左侧区域（显示K线图）
            pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, LEFT_WIDTH, SCREEN_HEIGHT))  # 左侧区域背景
            self.render_kline(self.screen)  # 在左侧区域绘制K线

            # 右侧区域（分为上下）
            pygame.draw.rect(self.screen, (50, 50, 50), (LEFT_WIDTH, 0, RIGHT_WIDTH, TOP_HEIGHT))  # 右上区域背景
            pygame.draw.rect(self.screen, (80, 80, 80), (LEFT_WIDTH, TOP_HEIGHT, RIGHT_WIDTH, BOTTOM_HEIGHT))  # 右下区域背景


            self.track.draw(self.screen, car_position=position, car_profit=profit)
            self.car.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(60)
        else:
            scaled_screen = pygame.transform.smoothscale(self.screen, self.train_size)
            x = np.transpose(
                np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
            )

            # (C, H, W) format
            x_tensor = torch.from_numpy(x).permute(2, 0, 1)  # (3, 96, 96)

            # grayscale
            grayscale_img = self.gs(x_tensor)  # (1, 96, 96)

            return grayscale_img.numpy().transpose(1, 2, 0)
        
    def render_kline(self, surface):
        """在左侧区域绘制K线图"""
        x_offset = 10
        y_offset = 500  # 初始Y位置
        line_color = (0, 255, 0)

        for index, row in self.df.iterrows():
            close_price = row['Close']
            pygame.draw.line(surface, line_color, (x_offset, y_offset), (x_offset + 5, y_offset - close_price * 100), 2)
            x_offset += 10