import pygame
import pandas as pd
import math


TRACK_SEGMENT_HEIGHT = 60  # 每段道路的高度
ANGLE_SCALE = 1000  # 倾斜敏感度放大
BASE_ROAD_WIDTH = 100

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
        # 计算左右车道的宽度比例
        if self.close >= self.open:  # 上涨
            left = self.high - self.open  # 上涨部分宽度
            right = self.open - self.low  # 下跌部分宽度
        else:  # 下跌
            left = self.open - self.low   # 下跌部分宽度
            right = self.high - self.open  # 上涨部分宽度

        total = left + right
        
        # 确保总宽度不为零，避免出现负值
        if total == 0:
            left_ratio = 0.5
            right_ratio = 0.5
        else:
            left_ratio = left / total
            right_ratio = right / total

        # 设置最小车道宽度比例（避免某一侧车道为零）
        min_width = 0.1
        if left_ratio < min_width:
            left_ratio = min_width
            right_ratio = 1 - left_ratio
        if right_ratio < min_width:
            right_ratio = min_width
            left_ratio = 1 - right_ratio

        return left_ratio, right_ratio



class Track:
    def __init__(self, draw_rect):
        self.segments = None
        self.draw_rect = draw_rect  # 绘制区域（x, y, width, height）
        self.df = None
        self.track_segment_height = TRACK_SEGMENT_HEIGHT

    def step(self, df):
        self.df = df

    def generate_segments(self, df):
        segments = []
        for _, row in df.iterrows():
            segment = TrackSegment(
                row['Open'], row['High'], row['Low'], row['Close'], 'historical')
            segments.append(segment)
        return segments

    def draw(self, surface, car_position, car_profit):
        # 更新：使用传递的绘制区域参数
        left, top, width, height = self.draw_rect

        self.segments = self.generate_segments(self.df)
        self.track_segment_height = (height - top) / len(self.df)
        
        # 创建临时表面
        temp_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        temp_surface.fill((0, 0, 0, 0))  # 透明背景
        
        center_x = width // 2
        current_y = height
        current_center_x = center_x

        for segment in self.segments:
            
            # 在临时表面上绘制，使用相对坐标
            current_y, current_center_x = self._draw_segment_safe(
                temp_surface, segment, current_center_x, current_y, 
                car_position=car_position, car_profit=car_profit, 
                surface_width=width)
        
        # 将临时表面绘制到主表面
        surface.blit(temp_surface, (left, top))

    def _draw_segment_safe(self, surface, segment, center_x, bottom_y, car_position, car_profit, surface_width):
        """安全绘制单个道路片段，防止越界"""
        angle = segment.angle
        height = self.track_segment_height

        # 限制最大倾斜角度
        # MAX_ANGLE = math.radians(30)
        # angle = max(-MAX_ANGLE, min(angle, MAX_ANGLE))

        offset_x = math.tan(angle) * height / 2
        adjusted_center_x = center_x + offset_x

        # 左右车道动态计算
        left_ratio, right_ratio = segment.calc_lane_ratios()
        
        # 基础道路宽度，考虑边界
        base_width = min(BASE_ROAD_WIDTH, surface_width * 0.8)
        
        left_width = base_width * left_ratio
        right_width = base_width * right_ratio
        
        # 确保道路不会超出边界
        max_left = center_x
        max_right = surface_width - center_x
        
        if left_width > max_left or right_width > max_right:
            scale = min(max_left / left_width, max_right / right_width)
            left_width *= scale
            right_width *= scale
        
        # 计算道路点
        points = [
            (center_x - left_width, bottom_y),
            (center_x + right_width, bottom_y),
            (adjusted_center_x + right_width, bottom_y - height),
            (adjusted_center_x - left_width, bottom_y - height)
        ]

        # 绘制道路主体
        road_color = (80, 80, 80)
        pygame.draw.polygon(surface, road_color, points)

        # 绘制中心线和边界线
        pygame.draw.line(surface, (255, 255, 255), (center_x, bottom_y), (adjusted_center_x, bottom_y - height), 2)
        pygame.draw.line(surface, (200, 200, 200), points[0], points[3], 2)
        pygame.draw.line(surface, (200, 200, 200), points[1], points[2], 2)

        # 动态计算防护栏位置（关键实现）
        self.draw_fence(surface, points, car_position, car_profit)

        return bottom_y - height, adjusted_center_x

    def draw_fence(self, surface, points, car_position, car_profit):
        # 基础参数
        base_fence_offset = BASE_FENCE_DISTANCE
        loss_scale = LOSS_SCALE
        min_distance = MIN_FENCE_DISTANCE
        fence_color = (200, 0, 0)
        
        # 如果没有持仓，不绘制防护栏
        if car_position == 0:
            return
        
        # 计算防护栏偏移量
        if car_profit > 0:  # 亏损状态
            # 亏损越大，防护栏越靠近道路
            offset = base_fence_offset - abs(car_profit) * loss_scale
            offset = max(min_distance, offset)
        else:  # 盈利状态
            # 盈利时，防护栏远离道路
            offset = base_fence_offset * 2
        
        # 根据仓位方向确定防护栏位置
        if car_position > 0:  # 多仓，左侧
            fence_start = (points[0][0] - offset, points[0][1])
            fence_end = (points[3][0] - offset, points[3][1])
        else:  # 空仓，右侧
            fence_start = (points[1][0] + offset, points[1][1])
            fence_end = (points[2][0] + offset, points[2][1])
        
        # 绘制防护栏
        pygame.draw.line(surface, fence_color, fence_start, fence_end, 3)

