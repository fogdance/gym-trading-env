import pygame
import math
import talib
import numpy as np

# 常量定义
TRACK_SEGMENT_HEIGHT = 60  # 每段道路高度
BASE_ROAD_WIDTH = 120      # 基础道路宽度
CAR_WIDTH = 15             # 车辆宽度
LANE_OFFSET = 1000         # 仓位偏移敏感度
BASE_FENCE_DISTANCE = 20   # 护栏距离
LOSS_SCALE = 200           # 浮亏系数
MIN_FENCE_DISTANCE = 2.5   # 最小护栏距离

class Car:
    def __init__(self, draw_rect, df_size, day_lost_limit: float, drawback_limit: float, trade_lot):
        self.day_lost_limit = day_lost_limit
        self.drawback_limit = drawback_limit
        self.trade_lot = trade_lot
        self.position = 0
        self.profit = 0.0
        self.color = (255, 0, 0)
        self.draw_rect = draw_rect
        left, top, width, height = draw_rect
        self.car_height = (height - top) / df_size * 2

    def update(self, position, profit):
        self.position = position
        self.profit = profit

    def draw(self, surface, track, current_day_lost: float, current_drawback: float):
        """
        绘制车辆及指标：
        - 车辆高度为 self.car_height（初始化时已乘2），但显示时使用其一半作为车身高度。
        - 指标方块的高度为车辆显示高度的一半，宽度等于车宽（CAR_WIDTH）。
        - 指标方块总高度与车身一致。
        - 指标一直显示在车辆右侧，默认距离为 4×CAR_WIDTH，
            当当前指标达到上限时，距离降为0，即紧贴车辆。
        """
        # 没有道路数据则不绘制
        if not track.segments:
            return

        # 获取最新道路片段的中轴坐标
        latest_segment = track.segments[-1]
        center_x = latest_segment.middle

        # 根据仓位计算车辆水平偏移：多头时向左，空头时向右
        ratio = 0.0
        if self.trade_lot != 0:
            ratio = self.position / self.trade_lot
        offset_x = -ratio * CAR_WIDTH  # 以车宽为偏移尺度
        car_x = center_x + offset_x

        # 取 track.draw_rect 的上边界作为起始 y 坐标，
        # 车辆绘制在最新道路片段的中间位置
        top = track.draw_rect[1]
        car_y = top + track.track_segment_height / 2

        # 使用 self.car_height 的一半作为显示车身高度
        display_car_height = self.car_height / 2
        car_rect = pygame.Rect(car_x - CAR_WIDTH / 2,
                            car_y - display_car_height / 2,
                            CAR_WIDTH, display_car_height)
        pygame.draw.rect(surface, self.color, car_rect)

        # --- 绘制指标小方块 ---
        # 每个指标方块高度为 display_car_height/2，宽度等于 CAR_WIDTH
        indicator_height = display_car_height / 2
        indicator_width = CAR_WIDTH

        # 默认指标与车辆之间的水平距离为 4 倍车宽
        default_offset = 4 * CAR_WIDTH

        # 根据当前指标与限制值，计算距离：
        # 当当前指标为0时，距离为默认值；达到限制时，距离为0。
        if self.day_lost_limit > 0:
            d_day = default_offset * (1 - min(1.0, current_day_lost / self.day_lost_limit))
        else:
            d_day = default_offset

        if self.drawback_limit > 0:
            d_account = default_offset * (1 - min(1.0, current_drawback / self.drawback_limit))
        else:
            d_account = default_offset

        # 指标一直显示在车辆右侧，分别计算两个指标的X坐标
        indicator1_x = car_rect.right + d_day
        indicator2_x = car_rect.right + d_account

        # 定义两个指标矩形：第一个为单日最大回撤，第二个为账户最大回撤
        indicator1_rect = pygame.Rect(indicator1_x, car_rect.top,
                                    indicator_width, indicator_height)
        indicator2_rect = pygame.Rect(indicator2_x, car_rect.top + indicator_height,
                                    indicator_width, indicator_height)

        # 绘制指标方块的边框
        outline_color = (255, 255, 255)
        pygame.draw.rect(surface, outline_color, indicator1_rect, 1)
        pygame.draw.rect(surface, outline_color, indicator2_rect, 1)

        # 指标颜色
        day_color = (255, 165, 0)      # 橙色表示单日最大回撤
        account_color = (255, 255, 0)  # 黄色表示账户最大回撤

        # 填充整个指标方块
        pygame.draw.rect(surface, day_color, indicator1_rect)
        pygame.draw.rect(surface, account_color, indicator2_rect)
