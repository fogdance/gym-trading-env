import pygame
import numpy as np
import torch
from .track import Track
from .car import Car
from .controller import Controller
from torchvision.transforms import Grayscale
from .bottom_panel import BottomPanel
from typing import Optional, Tuple

# 屏幕尺寸
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# 配置比例（左侧:右侧 = 4:6）
LEFT_WIDTH = SCREEN_WIDTH * 0 // 10  # 320像素
RIGHT_WIDTH = SCREEN_WIDTH * 10 // 10  # 480像素

TOP_HEIGHT = SCREEN_HEIGHT * 9 // 10
BOTTOM_HEIGHT = SCREEN_HEIGHT // 10

class Game:
    """外汇赛车游戏主类"""
    
    def __init__(self, train_size: Tuple[int, int], df_size: int, 
                 day_lost: float, drawback: float, 
                 trade_lot: float, max_long_position: float, max_short_position: float):
        """
        初始化游戏
        
        Args:
            train_size: 训练模式下屏幕缩放大小
            df_size: 数据帧大小
            day_lost: 最大日亏损限制
            drawback: 最大回撤限制
        """
        self.df = None
        pygame.init()
        self.train_size = train_size
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Forex Racer')
        self.clock = pygame.time.Clock()

        self.track = Track((LEFT_WIDTH, 0, RIGHT_WIDTH, TOP_HEIGHT))
        self.car = Car((LEFT_WIDTH, 0, RIGHT_WIDTH, TOP_HEIGHT), df_size)
        self.controller = Controller()
        self.gs = Grayscale(num_output_channels=1)
        self.bottom_panel = BottomPanel(
            (LEFT_WIDTH, TOP_HEIGHT, RIGHT_WIDTH, BOTTOM_HEIGHT),
            day_lost, drawback,
            trade_lot, max_long_position, max_short_position
        )

    def step(self, df) -> None:
        """更新游戏状态"""
        self.df = df
        self.track.step(df)

    def render(self, position: float, profit: float, current_day_lost: float, 
              current_drawback: float, render_mode: str = 'human') -> Optional[np.ndarray]:
        """
        渲染游戏画面
        
        Args:
            position: 当前仓位
            profit: 当前盈亏
            current_day_lost: 当前日亏损
            current_drawback: 当前回撤
            render_mode: 'human' 或其他（用于训练）
        
        Returns:
            如果render_mode不是'human'，返回灰度图像数组
        """
        self.car.update(position, profit)

        if render_mode == 'human':
            self.screen.fill((0, 0, 0))
            
            # 右侧上区域（赛道）
            pygame.draw.rect(self.screen, (50, 50, 50), 
                           (LEFT_WIDTH, 0, RIGHT_WIDTH, TOP_HEIGHT))


            # 绘制底部面板
            self.bottom_panel.draw(self.screen, position, profit, 
                                 current_day_lost, current_drawback)
            
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

