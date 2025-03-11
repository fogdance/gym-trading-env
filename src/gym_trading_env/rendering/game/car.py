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
        self.car_height = (height - top) / df_size * 2

    def update(self, position, profit):
        self.position = position
        self.profit = profit

    def draw(self, surface, track, current_day_lost: float, current_drawback: float, current_rrr: float):
        """
        绘制车辆，并根据单日最大亏损和盈亏比调整车辆在赛道上的横向位置：
        - 无持仓时车辆在赛道中轴。
        - 做多时：基础偏移基于当前仓位，最大为 road_width/4；
            根据 current_day_lost 增加偏移，最大为 road_width/4；
            根据 current_rrr 与 target_rrr 的差距增加额外偏移：
            - current_rrr >= target_rrr 时无额外偏移
            - current_rrr < target_rrr 时偏移逐渐增大，越接近 2.0 偏移越大，2.0以下撞左路边。
        - 做空时逻辑相反。
        """
        if not track.segments:
            return

        # 获取最新道路片段的中轴坐标和宽度
        latest_segment = track.segments[-1]
        center_x = latest_segment.middle
        road_width = latest_segment.road_width

        # 计算基于盈亏比的额外偏移
        rrr_offset = 0
        target_rrr = self.risk_reward_ratio * 1.25
        if current_rrr is not None and current_rrr < target_rrr:
            # 在 2.0 到 target_rrr (2.5) 之间线性插值
            # current_rrr = 2.5 时偏移为 0，current_rrr = 2.0 时偏移为 road_width/4
            if current_rrr <= self.risk_reward_ratio:
                rrr_offset = road_width / 4 + 10 # 最大偏移，撞边
            else:
                # 线性计算：(target_rrr - current_rrr) / (target_rrr - 2.0) * (road_width / 4)
                rrr_offset = (target_rrr - current_rrr) / (target_rrr - self.risk_reward_ratio) * (road_width / 4)

        # 根据仓位和亏损指标调整车辆横向位置
        if self.position > 0:  # 做多：车辆向左偏移
            # 基础偏移：当持仓达到最大时，基础偏移最大为 road_width/4
            base_offset = (road_width / 4) * min(1.0, self.position / self.max_long_position)
            # 总偏移 = 基础 + 亏损 + 盈亏比偏移
            final_offset = base_offset + rrr_offset
            # 限制最大偏移不超过路边
            final_offset = min(final_offset, road_width / 2)
            car_x = center_x - final_offset
        elif self.position < 0:  # 做空：车辆向右偏移
            base_offset = (road_width / 4) * min(1.0, abs(self.position) / self.max_short_position)
            final_offset = base_offset + rrr_offset
            final_offset = min(final_offset, road_width / 2)
            car_x = center_x + final_offset
        else:  # 空仓：加上盈亏比偏移
            base_offset = 0
            # 总偏移 = 左车道基准 + 盈亏比偏移
            final_offset = base_offset + rrr_offset
            final_offset = min(final_offset, road_width / 2)  # 限制不超过左路边
            car_x = center_x - final_offset

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