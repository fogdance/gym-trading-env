import pygame
import pandas as pd
import math


TRACK_SEGMENT_HEIGHT = 60  # 每段道路的高度
ANGLE_SCALE = 1000  # 倾斜敏感度放大
BASE_ROAD_WIDTH = 120
BASE_ROAD_RANGE = 5

# 防护栏参数
BASE_FENCE_DISTANCE = 20  # 护栏距离道路边界基础距离
LOSS_SCALE = 200           # 浮亏系数，越大越敏感
MIN_FENCE_DISTANCE = 2.5

class TrackSegment:
    """表示单个道路片段"""
    def __init__(self, date, open_p, high_p, low_p, close_p, segment_type='historical'):
        self.date = date
        self.open = open_p
        self.high = high_p
        self.low = low_p
        self.close = close_p
        self.type = segment_type
        
        self.angle = self.calc_angle()

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
            right = self.open - self.low   # 下跌部分宽度
            left = self.high - self.open  # 上涨部分宽度

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


        return left_ratio, right_ratio, abs(self.high - self.low)



class Track:
    def __init__(self, draw_rect):
        self.segments = None
        self.draw_rect = draw_rect  # 绘制区域（x, y, width, height）
        self.df = None
        self.track_segment_height = TRACK_SEGMENT_HEIGHT
        self.avg_road_width = 0

    def step(self, df):
        self.df = df

    def generate_segments(self, df):
        segments = []
        avg_road_width = 0
        for idx, row in df.iterrows():
            segment = TrackSegment(
                idx, row['Open'], row['High'], row['Low'], row['Close'], 'historical')
            segments.append(segment)
            avg_road_width = avg_road_width + abs(row['High'] - row['Low'])
        self.avg_road_width = avg_road_width / len(df)
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
            last = segment is self.segments[-1]
            # 在临时表面上绘制，使用相对坐标
            current_y, current_center_x = self._draw_segment_safe(
                temp_surface, segment, current_center_x, current_y, 
                car_position=car_position, car_profit=car_profit, 
                surface_width=width, last=last)
        
        # 将临时表面绘制到主表面
        surface.blit(temp_surface, (left, top))

    def _draw_segment_safe(self, surface, segment, center_x, bottom_y, car_position, car_profit, surface_width, last=False):
        """安全绘制单个道路片段，防止越界"""
        angle = segment.angle
        height = self.track_segment_height

        # 限制最大倾斜角度
        # MAX_ANGLE = math.radians(30)
        # angle = max(-MAX_ANGLE, min(angle, MAX_ANGLE))

        offset_x = math.tan(angle) * height / 2
        adjusted_center_x = center_x + offset_x

        # 左右车道动态计算
        left_ratio, right_ratio, road_width = segment.calc_lane_ratios()
        
        # 基础道路宽度，考虑边界
        base_width = min(BASE_ROAD_WIDTH*4, BASE_ROAD_WIDTH * (road_width / self.avg_road_width))
        
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

    
        # 如果是最后一个segment，绘制日期时间
        if last:
            self.draw_date(surface, segment.date, adjusted_center_x, bottom_y - height, surface_width)

        return bottom_y - height, adjusted_center_x

    def draw_date(self, surface, date, x, y, surface_width):
        """在道路的右侧绘制日期"""
        # 设置字体
        font = pygame.font.SysFont('Arial', 24)
        
        # 渲染日期字符串
        date_str = date.strftime('%Y-%m-%d %H:%M:%S')  # 格式化日期为字符串
        text = font.render(date_str, True, (255, 255, 255))  # 白色字体
        
        # 计算文本位置
        text_width, text_height = text.get_size()
        text_x = surface_width - text_width - 10  # 右侧距离
        text_y = y + text_height // 2  # 垂直居中

        # 绘制文本
        surface.blit(text, (text_x, text_y))    


