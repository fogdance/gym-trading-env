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

    def update(self, action, segment):
        # 根据action更新赛车位置
        if action == 'left':
            self.position = max(self.position - 0.1, -1)
        elif action == 'right':
            self.position = min(1, self.position + 0.1)
        elif action == 'close':
            self.position = 0

        # 根据新的位置计算盈亏
        self.calculate_profit(segment=segment)

    def calculate_profit(self, segment):
        # 盈亏逻辑明确
        if self.position == 0:
            self.profit = 0
        else:
            direction = -1 if self.position < 0 else 1
            if (segment.angle * direction) > 0:
                self.profit = abs(segment.angle) * abs(self.position)  # 盈利状态
            else:
                self.profit = -abs(segment.angle * self.position)

        # 根据盈亏情况更新颜色
        if self.position == 0:
            self.color = (180, 180, 180)  # 空仓为灰色
        elif self.profit > 0:
            self.color = (0, 200, 0)  # 浮盈绿色
        else:
            self.color = (200, 0, 0)  # 浮亏红色

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
