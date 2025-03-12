import pygame
import numpy as np
import torch
from .track import Track
from .car import Car
from .controller import Controller
from torchvision.transforms import Grayscale
from .bottom_panel import BottomPanel
from typing import Optional, Tuple
from PIL import Image  # 添加 PIL 库用于保存图像

# 屏幕尺寸
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# 配置比例（左侧:右侧 = 4:6）
LEFT_WIDTH = SCREEN_WIDTH * 0 // 10  # 0像素
RIGHT_WIDTH = SCREEN_WIDTH * 10 // 10  # 800像素

TOP_HEIGHT = SCREEN_HEIGHT * 9 // 10  # 540像素
BOTTOM_HEIGHT = SCREEN_HEIGHT // 10   # 60像素

class Game:
    """外汇赛车游戏主类"""
    
    def __init__(self, train_size: Tuple[int, int], df_size: int, 
                 day_lost_limit: float, drawback_limit: float, risk_reward_ratio: float,
                 trade_lot: float, max_long_position: float, max_short_position: float,
                 render_mode: str):
        """
        初始化游戏
        
        Args:
            train_size: 训练模式下屏幕缩放大小
            df_size: 数据帧大小
            day_lost_limit: 最大日亏损限制
            drawback_limit: 最大回撤限制
            render_mode: 'human' 或其他（用于训练）
        """
        self.render_mode = render_mode
        self.df = None
        self.train_size = train_size
        
        # 初始化 Pygame
        pygame.init()
        
        # 创建离屏表面用于统一绘制
        self.offscreen_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # human 模式下初始化显示窗口
        if render_mode == 'human':
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Forex Racer')
            self.clock = pygame.time.Clock()
        else:
            self.screen = None  # 非 human 模式下无需显示窗口

        self.track = Track((LEFT_WIDTH, 0, RIGHT_WIDTH, TOP_HEIGHT))
        self.car = Car((LEFT_WIDTH, 0, RIGHT_WIDTH, TOP_HEIGHT), df_size, day_lost_limit, 
                       drawback_limit, trade_lot, risk_reward_ratio, max_long_position, max_short_position)
        self.controller = Controller()
        self.gs = Grayscale(num_output_channels=1)
        self.bottom_panel = BottomPanel(
            (LEFT_WIDTH, TOP_HEIGHT, RIGHT_WIDTH, BOTTOM_HEIGHT),
            day_lost_limit, drawback_limit,
            trade_lot, max_long_position, max_short_position
        )

    def step(self, df) -> None:
        """更新游戏状态"""
        self.df = df
        self.track.step(df)

    def _render_common(self, surface, long_position: float, short_position: float, profit: float, 
                      current_day_lost: float, current_drawback: float, current_rrr: float) -> None:
        """通用渲染逻辑，绘制到指定表面"""
        surface.fill((0, 0, 0))
        
        # 右侧上区域（赛道）
        pygame.draw.rect(surface,  (220,220,220), 
                        (LEFT_WIDTH, 0, RIGHT_WIDTH, TOP_HEIGHT))

        # 绘制底部面板
        self.bottom_panel.draw(surface, long_position, short_position, profit, 
                              current_day_lost, current_drawback)
        
        self.track.draw(surface)
        self.car.draw(surface, self.track, current_day_lost, current_drawback, current_rrr)

    def render(self, long_position: float, short_position: float, profit: float, current_day_lost: float, 
              current_drawback: float, current_rrr: float, render_mode: str = None) -> np.ndarray:
        """
        渲染游戏画面
        
        Args:
            position: 当前仓位
            profit: 当前盈亏
            current_day_lost: 当前日亏损
            current_drawback: 当前回撤
            render_mode: 'human' 或其他（用于训练），优先使用实例变量
        
        Returns:
            灰度图像数组 (H, W, 1)
        """
        # 使用传入的 render_mode 或默认使用实例变量
        render_mode = render_mode if render_mode is not None else self.render_mode
        self.car.update(long_position - short_position, profit)

        # 统一在离屏表面上绘制
        self._render_common(self.offscreen_surface, long_position, short_position, profit, 
                           current_day_lost, current_drawback, current_rrr)

        # human 模式：将离屏表面渲染到屏幕
        if render_mode == 'human' and self.screen is not None:
            self.screen.blit(self.offscreen_surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(60)

        # 生成灰度图像并返回
        scaled_surface = pygame.transform.smoothscale(self.offscreen_surface, self.train_size)
        x = np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_surface)), axes=(1, 0, 2)
        )
        x_tensor = torch.from_numpy(x).permute(2, 0, 1)  # (3, H, W)
        grayscale_img = self.gs(x_tensor)  # (1, H, W)
        
        if False:
            grayscale_np = grayscale_img.numpy().transpose(1, 2, 0)  # (H, W, 1)
            grayscale_pil = Image.fromarray((grayscale_np.squeeze() * 255).astype(np.uint8), mode='L')
            grayscale_pil.save('data/gray_frame.png')

        return grayscale_img.numpy().transpose(1, 2, 0)  # (H, W, 1)

    def __del__(self):
        """清理资源"""
        pygame.quit()