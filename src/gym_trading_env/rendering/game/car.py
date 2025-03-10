import pygame


import math
import talib
import numpy as np

# 常量定义
TRACK_SEGMENT_HEIGHT = 60  # 每段道路高度
BASE_ROAD_WIDTH = 120     # 基础道路宽度
CAR_WIDTH = 15            # 车辆宽度
LANE_OFFSET = 1000        # 仓位偏移敏感度
BASE_FENCE_DISTANCE = 20  # 护栏距离
LOSS_SCALE = 200          # 浮亏系数
MIN_FENCE_DISTANCE = 2.5  # 最小护栏距离


class Car:
    def __init__(self, draw_rect, df_size, trade_lot):
        self.trade_lot = trade_lot
        self.position = 0
        self.profit = 0.0
        self.color = (255, 0, 0)
        self.draw_rect = draw_rect
        left, top, width, height = draw_rect
        self.car_height = (height - top) / df_size

    def update(self, position, profit):
        self.position = position
        self.profit = profit

    def draw(self, surface, track):
        # 如果没有道路数据，直接返回
        if not track.segments:
            return

        # 取最新的道路片段（假设最新片段是追加在末尾的）
        latest_segment = track.segments[-1]
        # 最新道路片段的中轴坐标
        center_x = latest_segment.middle

        # 计算水平偏移量
        # trade_lot 为最大仓位，当仓位为 0 时 ratio = 0；为正（多头）时 ratio > 0；为负（空头）时 ratio < 0
        ratio = 0.0
        if self.trade_lot != 0:
            ratio = self.position / self.trade_lot
            # 限制在 -1 到 1 之间
            # ratio = max(-1, min(1, ratio))
        # 设定偏移方向：多头时向左（负偏移），空头时向右（正偏移）
        offset_x = -ratio * CAR_WIDTH

        # 最终赛车的 x 坐标
        car_x = center_x + offset_x

        # 赛车显示在最新道路片段的中间位置
        # 注意：track.draw 中，track_segment_height 是每段的高度，
        # 假设最新片段绘制在 track.draw_rect 的顶部
        top = track.draw_rect[1]
        car_y = top + track.track_segment_height / 2

        # 绘制赛车（这里用一个矩形表示，中心对齐）
        rect = pygame.Rect(car_x - CAR_WIDTH / 2, car_y - self.car_height / 2, CAR_WIDTH, self.car_height)
        pygame.draw.rect(surface, self.color, rect)
