import pygame

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

CAR_WIDTH = 40
CAR_HEIGHT = 60
LANE_OFFSET = 150  # 左右道路偏移距离（仓位大小视觉化）

class Car:
    def __init__(self, draw_rect):
        self.position = 0      # 持仓方向: 0=空仓，负=多仓，正=空仓
        self.profit = 0.0     # 当前浮盈浮亏
        self.color = (180,180,180)  # 初始为灰色
        self.draw_rect = draw_rect  # 绘制区域（x, y, width, height）

    def update(self, positon, profit):
        self.position = positon
        self.profit = profit


    def draw(self, surface):
        """绘制赛车矩形到屏幕"""

        left, top, width, height = self.draw_rect
        center_x = left + width // 2

        car_x = center_x + (self.position * LANE_OFFSET)
        car_y = top

        # 创建赛车矩形
        car_rect = pygame.Rect(car_x - CAR_WIDTH // 2, car_y, CAR_WIDTH, CAR_HEIGHT)

        # 绘制赛车
        pygame.draw.rect(surface, self.color, car_rect)
