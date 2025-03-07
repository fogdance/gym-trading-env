import pygame
import pandas as pd
import math

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TRACK_SEGMENT_HEIGHT = 60  # 每段道路的高度
VISIBLE_TRACK_SEGMENTS = 10  # 屏幕显示道路段数
ANGLE_SCALE = 1000  # 倾斜敏感度放大
BASE_ROAD_WIDTH = 300
MIN_ROAD_WIDTH = 100

# 防护栏参数
BASE_FENCE_DISTANCE = 20  # 护栏距离道路边界基础距离
LOSS_SCALE = 200           # 浮亏系数，越大越敏感
MIN_FENCE_DISTANCE = 5

class TrackSegment:
    """表示单个道路片段"""
    def __init__(self, open_p, high_p, low_p, close_p, segment_type='historical'):
        self.open = open_p
        self.high = high_p
        self.low = low_p
        self.close = close_p
        self.type = segment_type
        
        self.angle = self.calc_angle()
        self.left_ratio, self.right_ratio = self.calc_lane_ratios()

    def calc_angle(self):
        """计算当前K线的倾斜角度"""
        price_change_pct = (self.close - self.open) / self.open
        return math.atan(price_change_pct * ANGLE_SCALE)

    def calc_lane_ratios(self):
        """根据OHLC数据动态计算车道宽度的比例"""
        if self.close >= self.open:  # 上涨
            left = self.high - self.open  # 上涨部分宽度
            right = self.open - self.low  # 下跌部分宽度
        else:  # 下跌
            left = self.open - self.low   # 下跌部分宽度
            right = self.high - self.open  # 上涨部分宽度

        # 计算两侧车道宽度比例
        total = left + right
        left_ratio = left / total if total != 0 else 0.5
        right_ratio = right / total if total != 0 else 0.5

        return left_ratio, right_ratio


class Track:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.current_index = VISIBLE_TRACK_SEGMENTS
        self.segments = self.generate_segments()

    def generate_segments(self):
        segments = []
        for _, row in self.df.iterrows():
            segment = TrackSegment(
                row['Open'], row['High'], row['Low'], row['Close'], 'historical')
            segments.append(segment)
        return segments

    def current_segment(self):
        return self.segments[self.current_index]

    def move_next(self):
        if self.current_index < len(self.segments) - 1:
            self.current_index += 1

    def draw(self, surface, car_position, car_profit):
        surface.fill((0, 0, 0))
        center_x, bottom_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT
        current_y = bottom_y
        current_center_x = center_x

        for offset in range(VISIBLE_TRACK_SEGMENTS-1, -1, -1):
            idx = self.current_index - offset
            if idx < 0:
                continue
            segment = self.segments[idx]
            current_y, current_center_x = self._draw_segment(
                surface, segment, current_center_x, current_y, car_position=car_position, car_profit=car_profit)

    def _draw_segment(self, surface, segment, center_x, bottom_y, car_position, car_profit):
        """绘制单个道路片段（改进版）"""
        angle = segment.angle
        height = TRACK_SEGMENT_HEIGHT

        # 根据角度计算中心线的水平偏移（关键改进）
        center_offset_x = math.tan(angle) * (height / 2)
        adjusted_center_x = center_x + center_offset_x

        # 左右车道动态计算
        left_ratio, right_ratio = segment.calc_lane_ratios()
        left_width = BASE_ROAD_WIDTH * left_ratio
        right_width = BASE_ROAD_WIDTH - left_width

        points = [
            (center_x - left_width, bottom_y),
            (center_x + right_width, bottom_y),
            (adjusted_center_x + right_width, bottom_y - height),
            (adjusted_center_x - left_width, bottom_y - height)
        ]

        # 绘制道路主体
        road_color = (80, 80, 80)
        pygame.draw.polygon(surface, road_color, points)

        # 绘制中心线
        pygame.draw.line(surface, (255, 255, 255), (center_x, bottom_y), (adjusted_center_x, bottom_y - height), 2)

        # 绘制左右边界线
        pygame.draw.line(surface, (200, 200, 200), points[0], points[3], 2)
        pygame.draw.line(surface, (200, 200, 200), points[1], points[2], 2)

        # 动态计算防护栏位置（关键实现）
        self.draw_fence(surface, points, car_position, car_profit)

        return bottom_y - height, adjusted_center_x


    def draw_fence(self, surface, points, car_position, car_profit):
        """防护栏根据赛车位置与盈亏动态绘制"""
        left_start, left_end = points[0], points[3]
        right_start, right_end = points[1], points[2]

        base_fence_offset = 20
        loss_scale = LOSS_SCALE = 100  # 生产中调整此参数

        if car_position < 0:  # 多仓，左侧
            offset = base_fence_offset - abs(car_profit) * loss_scale if car_profit < 0 else base_fence_offset * 2
            offset = max(5, offset)
            fence_start = (points[0][0] - offset, points[0][1])
            fence_end = (points[3][0] - offset, points[3][1])
        elif car_position > 0:  # 空仓，右侧
            offset = base_fence_offset - abs(car_profit) * loss_scale
            offset = max(5, offset)
            fence_offset = offset
            fence_start = (points[1][0] + fence_offset, points[1][1])
            fence_end = (points[2][0] + fence_offset, points[2][1])
        else:  # 空仓
            return

        # 绘制防护栏
        fence_color = (200,0,0)
        if car_position < 0:
            pygame.draw.line(surface, fence_color, fence_start, fence_end, 3)
        elif car_position > 0:
            pygame.draw.line(surface, fence_color, fence_start, fence_end, 3)

