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
    def __init__(self, draw_rect, df_size, day_lost_limit: float, drawback_limit: float, trade_lot: float,
                 risk_reward_ratio: float, max_long_position: float, max_short_position: float):
        self.risk_reward_ratio = risk_reward_ratio
        self.max_long_position = max_long_position
        self.max_short_position = max_short_position
        self.day_lost_limit = day_lost_limit
        self.drawback_limit = drawback_limit
        self.trade_lot = trade_lot
        self.position = 0
        self.profit = 0.0
        self.color = (255,150,0)
        self.draw_rect = draw_rect
        left, top, width, height = draw_rect
        self.car_height = (height - top) / df_size * 4

    def update(self, position, profit):
        self.position = position
        self.profit = profit

    def draw(self, surface, track, current_day_lost: float, current_drawback: float, current_rrr: float):
        """
        绘制车辆，并根据单日最大亏损和盈亏比调整车辆在赛道上的横向位置：
        - 无持仓时车辆在赛道中轴。
        - 做多时：基础偏移基于当前仓位与最大多头仓位（max_long_position）的比例，理想状态下最大为左移 BASE_ROAD_WIDTH/4；
            同时，根据 current_day_lost（单日亏损）与 day_lost_limit 的比例，再额外向左偏移，最大为 BASE_ROAD_WIDTH/4；
            如果 current_rrr < target_rrr，则撞到左路边。
        - 做空时逻辑相反，车辆向右偏移，如果 current_rrr < target_rrr，则撞到右路边。
        """
        if not track.segments:
            return

        # 获取最新道路片段的中轴坐标和宽度
        latest_segment = track.segments[-1]
        center_x = latest_segment.middle
        road_width = latest_segment.road_width

        # 检查盈亏比是否达标
        rrr_not_met = False
        if current_rrr is not None:
            rrr_not_met = current_rrr < self.risk_reward_ratio

        # 根据仓位和亏损指标调整车辆横向位置
        if self.position > 0:  # 做多：车辆向左偏移
            if rrr_not_met:  # 盈亏比未达标，直接撞左路边
                car_x = center_x - road_width / 2  # 左路边位置
            else:
                # 基础偏移：当持仓达到最大时，基础偏移最大为 road_width/4
                base_offset = (road_width / 4) * min(1.0, self.position / self.max_long_position)
                # 额外偏移：当亏损达到单日亏损限额时，额外偏移最大也为 road_width/4
                additional_offset = (road_width / 4) * min(1.0, current_day_lost / self.day_lost_limit)
                final_offset = base_offset + additional_offset
                car_x = center_x - final_offset
        elif self.position < 0:  # 做空：车辆向右偏移
            if rrr_not_met:  # 盈亏比未达标，直接撞右路边
                car_x = center_x + road_width / 2  # 右路边位置
            else:
                base_offset = (road_width / 4) * min(1.0, abs(self.position) / self.max_short_position)
                additional_offset = (road_width / 4) * min(1.0, current_day_lost / self.day_lost_limit)
                final_offset = base_offset + additional_offset
                car_x = center_x + final_offset
        else:
            car_x = center_x

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