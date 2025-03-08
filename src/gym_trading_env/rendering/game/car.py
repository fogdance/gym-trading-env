import pygame


CAR_WIDTH = 15
LANE_OFFSET = 1000  # 左右道路偏移距离（仓位大小视觉化）

class Car:
    def __init__(self, draw_rect, df_size, trade_lot):
        self.trade_lot = trade_lot
        self.position = 0      # 持仓方向: 0=无仓，负=空仓，正=多仓
        self.profit = 0.0     # 当前浮盈浮亏
        self.color = (255, 0, 0)  # 初始为红色
        self.draw_rect = draw_rect  # 绘制区域（x, y, width, height）
        left, top, width, height = self.draw_rect

        self.car_height = (height-top) / df_size

    def update(self, positon, profit):
        self.position = positon
        self.profit = profit


    def draw(self, surface):
        """绘制赛车矩形到屏幕"""

        left, top, width, height = self.draw_rect
        center_x = left + width // 2

        car_x = center_x - (self.position/self.trade_lot) * CAR_WIDTH
        car_y = top

        # 创建赛车矩形
        car_rect = pygame.Rect(car_x - CAR_WIDTH // 2, car_y, CAR_WIDTH, self.car_height)

        # 绘制赛车
        pygame.draw.rect(surface, self.color, car_rect)
