import pygame
import numpy as np
import torch
from typing import Tuple, Optional
from torchvision.transforms import Grayscale


class BottomPanel:
    """右下区域的可视化面板，负责绘制仓位、盈亏、日亏损和回撤"""
    
    def __init__(self, rect: Tuple[int, int, int, int], day_lost: float, drawback: float,
                    trade_lot: float, max_long_position: float, max_short_position: float):
        """
        初始化底部面板
        
        Args:
            rect: (x, y, width, height) 表示面板区域
            day_lost: 最大日亏损限制
            drawback: 最大回撤限制
        """
        self.rect = pygame.Rect(rect)
        self.day_lost = day_lost
        self.drawback = drawback
        self.center_x = self.rect.x + self.rect.width // 2
        self.profit_width = self.rect.width * 0.2
        self.max_width = self.rect.width * 0.2
        self.bar_height = self.rect.height * 0.2
        self.spacing = self.rect.height * 0.05
        self.avg_position = (max_long_position + max_short_position) / 2

    def draw(self, screen: pygame.Surface, position: float, profit: float, 
             current_day_lost: float, current_drawback: float) -> None:
        """绘制所有指标"""
        # 背景
        pygame.draw.rect(screen, (0, 0, 0), self.rect)

        # 1. 仓位 (position)
        pos_width = position / self.avg_position * self.profit_width
        if position > 0:  # 做多 - 向左
            pos_rect = pygame.Rect(self.center_x - pos_width, self.rect.y + self.spacing,
                                 pos_width, self.bar_height)
            pygame.draw.rect(screen, (0, 255, 0), pos_rect)
        elif position < 0:  # 做空 - 向右
            pos_rect = pygame.Rect(self.center_x, self.rect.y + self.spacing,
                                 -pos_width, self.bar_height)
            pygame.draw.rect(screen, (255, 0, 0), pos_rect)

        # 2. 盈亏 (profit)
        profit_width = profit * self.profit_width
        if profit > 0:  # 盈利 - 向左
            profit_rect = pygame.Rect(self.center_x - profit_width,
                                    self.rect.y + self.bar_height + 2 * self.spacing,
                                    profit_width, self.bar_height)
            pygame.draw.rect(screen, (0, 255, 0), profit_rect)
        elif profit < 0:  # 亏损 - 向右
            profit_rect = pygame.Rect(self.center_x,
                                    self.rect.y + self.bar_height + 2 * self.spacing,
                                    -profit_width, self.bar_height)
            pygame.draw.rect(screen, (255, 0, 0), profit_rect)

        # 3. 日亏损 (current_day_lost) - 消失形式
        if current_day_lost < self.day_lost:
            lost_ratio = 1 - (current_day_lost / self.day_lost)  # 0时满，1时消失
            lost_width = lost_ratio * self.max_width
            lost_rect = pygame.Rect(self.center_x,
                                  self.rect.y + 2 * self.bar_height + 3 * self.spacing,
                                  lost_width, self.bar_height)
            pygame.draw.rect(screen, (255, 165, 0), lost_rect)

        # 4. 回撤 (current_drawback) - 消失形式
        if current_drawback < self.drawback:
            drawback_ratio = 1 - (current_drawback / self.drawback)  # 0时满，1时消失
            drawback_width = drawback_ratio * self.max_width
            drawback_rect = pygame.Rect(self.center_x,
                                      self.rect.y + 3 * self.bar_height + 4 * self.spacing,
                                      drawback_width, self.bar_height)
            pygame.draw.rect(screen, (255, 255, 0), drawback_rect)

