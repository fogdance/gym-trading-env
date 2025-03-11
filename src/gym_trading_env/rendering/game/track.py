import pygame
import pandas as pd
import talib
import numpy as np

# 常量定义
TRACK_SEGMENT_HEIGHT = 60
BASE_ROAD_WIDTH = 160  # 屏幕宽度的 1/5（800 * 0.2 = 160）
MIDDLE_OFFSET_RANGE = 200  # 中心线最大偏移范围（像素）

class TrackSegment:
    def __init__(self, date, upper, middle, lower, road_width, close):
        self.date = date
        self.close = close
        self.upper = upper  # 映射后的上轨
        self.middle = middle  # 映射后的中轨
        self.lower = lower  # 映射后的下轨
        self.road_width = road_width  # 固定宽度

class Track:
    def __init__(self, draw_rect):
        self.draw_rect = draw_rect  # (x, y, width, height)
        self.df = None
        self.segments = []
        self.track_segment_height = TRACK_SEGMENT_HEIGHT
        self.avg_band_width = None  # 固定平均宽度
        self.value_min = 0
        self.value_max = 0
        self.band_widths = []
        self.initialized = False  # 标记是否初始化
        self.max_segments = 0

    def step(self, df):
        self.df = df
        if not self.initialized:
            # 第一次调用，初始化所有段
            self.segments = self.generate_segments(df)
            self.initialized = True
        else:
            # 增量更新，只追加新段
            self.append_new_segments(df)

    def generate_segments(self, df):
        close_prices = df['Close'].values
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
        
        # 计算全局最小值和最大值
        all_values = []
        self.band_widths = []
        for i, (u, m, l) in enumerate(zip(upper, middle, lower)):
            if np.isnan(u) or np.isnan(m) or np.isnan(l):
                u, m, l = close_prices[i], close_prices[i], close_prices[i]
            all_values.extend([u, m, l])
            self.band_widths.append(u - l)
        
        self.value_min = min(all_values)
        self.value_max = max(all_values)
        if self.value_max == self.value_min:
            self.value_max = self.value_min + 1
        
        # 固定平均宽度
        self.avg_band_width = sum(self.band_widths) / len(self.band_widths)
        
        segments = []
        for i, (date, u, m, l) in enumerate(zip(df.index, upper, middle, lower)):
            if np.isnan(u) or np.isnan(m) or np.isnan(l):
                u, m, l = close_prices[i], close_prices[i], close_prices[i]
            
            u_mapped = self.map_to_screen(u, 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])
            m_mapped = self.map_to_screen(m, 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])
            l_mapped = self.map_to_screen(l, 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])
            close_mapped = self.map_to_screen(close_prices[i], 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])

            road_width = BASE_ROAD_WIDTH * (self.band_widths[i] / self.avg_band_width)
            road_width = min(BASE_ROAD_WIDTH * 4, max(BASE_ROAD_WIDTH * 0.5, road_width))
            
            segment = TrackSegment(date, u_mapped, m_mapped, l_mapped, road_width, close_mapped)
            segments.append(segment)

        self.max_segments = len(self.segments)
        
        return segments

    def append_new_segments(self, df):
        close_prices = df['Close'].values
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
        
        # 只处理新数据（假设 df 追加了一行）
        new_date = df.index[-1]
        u, m, l = upper[-1], middle[-1], lower[-1]
        
        if np.isnan(u) or np.isnan(m) or np.isnan(l):
            u, m, l = close_prices[-1], close_prices[-1], close_prices[-1]
        
        # 更新全局最小值和最大值
        all_values = [u, m, l]
        self.value_min = min(self.value_min, min(all_values))
        self.value_max = max(self.value_max, max(all_values))
        if self.value_max == self.value_min:
            self.value_max = self.value_min + 1
        
        # 计算新段宽度，但不更新 avg_band_width
        band_width = u - l
        self.band_widths.append(band_width)
        
        u_mapped = self.map_to_screen(u, 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])
        m_mapped = self.map_to_screen(m, 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])
        l_mapped = self.map_to_screen(l, 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])
        close_mapped = self.map_to_screen(close_prices[-1], 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])

        road_width = BASE_ROAD_WIDTH * (band_width / self.avg_band_width)
        road_width = min(BASE_ROAD_WIDTH * 4, max(BASE_ROAD_WIDTH * 0.5, road_width))
        
        segment = TrackSegment(new_date, u_mapped, m_mapped, l_mapped, road_width, close_mapped)

        if len(self.segments) > self.max_segments:
            self.segments.pop(0)

        self.segments.append(segment)

    def map_to_screen(self, value, screen_min, screen_max):
        normalized = (value - self.value_min) / (self.value_max - self.value_min)
        return screen_min + normalized * (screen_max - screen_min)

    def draw(self, surface):
        left, top, width, height = self.draw_rect
        num_segments = len(self.df)
        self.track_segment_height = (height - top) / num_segments

        temp_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        temp_surface.fill((0, 0, 0, 0))

        centerline_points = []  # 用来存储每个分段的中心点坐标
        current_y = top

        # 逆序绘制每个分段
        for i in range(len(self.segments) - 1, -1, -1):
            segment = self.segments[i]
            # 计算道路两侧宽度
            center_x = segment.middle
            road_width = segment.road_width
            left_width = road_width / 2
            right_width = road_width / 2
            max_left = center_x
            max_right = width - center_x
            if left_width > max_left or right_width > max_right:
                scale = min(max_left / left_width, max_right / right_width)
                left_width *= scale
                right_width *= scale

            road_points = [
                (center_x - left_width, current_y),
                (center_x + right_width, current_y),
                (center_x + right_width, current_y + self.track_segment_height),
                (center_x - left_width, current_y + self.track_segment_height)
            ]
            pygame.draw.polygon(temp_surface, (100,100,100), road_points)

            # 保存当前分段的中心点（取该分段高度的中间）
            centerline_points.append((segment.close, current_y + self.track_segment_height / 2))

            # 调试，则绘制最新分段的日期文本
            if False and i == len(self.segments) - 1:
                self.draw_date_on_segment(temp_surface, segment, current_y, self.track_segment_height, width)


            current_y += self.track_segment_height

        # 绘制连续的中轴曲线
        pygame.draw.lines(temp_surface, (255, 255, 255), False, centerline_points, 2)

        surface.blit(temp_surface, (left, top))

    def draw_date_on_segment(self, surface, segment, current_y, segment_height, surface_width):
        """
        在给定 surface 上绘制指定分段的日期文本，文本绘制在分段右侧。
        
        参数：
            surface: 要绘制的 pygame.Surface 对象
            segment: 当前分段（包含 date 属性）
            current_y: 当前分段在 surface 上的纵坐标
            segment_height: 分段高度
            surface_width: surface 的宽度，用于计算右侧边距
        """
        font = pygame.font.SysFont(None, 20)
        date_str = str(segment.date)  # 根据需要调整日期格式
        text_surface = font.render(date_str, True, (255, 0, 0))
        x_pos = surface_width - text_surface.get_width() - 10
        y_pos = current_y + segment_height / 2 - text_surface.get_height() / 2
        surface.blit(text_surface, (x_pos, y_pos))

    def _draw_segment_safe(self, surface, segment, bottom_y, surface_width, last):
        height = self.track_segment_height
        center_x = segment.middle
        road_width = segment.road_width

        left_width = road_width / 2
        right_width = road_width / 2

        max_left = center_x
        max_right = surface_width - center_x
        if left_width > max_left or right_width > max_right:
            scale = min(max_left / left_width, max_right / right_width)
            left_width *= scale
            right_width *= scale

        road_points = [
            (center_x - left_width, bottom_y),
            (center_x + right_width, bottom_y),
            (center_x + right_width, bottom_y + height),
            (center_x - left_width, bottom_y + height)
        ]

        pygame.draw.polygon(surface, (80, 80, 80), road_points)
        pygame.draw.line(surface, (255, 255, 255), (center_x, bottom_y), 
                        (center_x, bottom_y + height), 2)
        pygame.draw.line(surface, (200, 200, 200), road_points[0], road_points[3], 2)
        pygame.draw.line(surface, (200, 200, 200), road_points[1], road_points[2], 2)



        return bottom_y + height

