import pygame
import pandas as pd
import math

# Track配置参数(生产中可调整到配置文件)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TRACK_SEGMENT_HEIGHT = 60  # 每段道路的高度
VISIBLE_TRACK_SEGMENTS = 10  # 屏幕显示道路段数
ANGLE_SCALE = 1000  # 倾斜敏感度放大
WIDTH_SCALE = 500  # 宽度敏感度放大
BASE_ROAD_WIDTH = 200  # 道路基础宽度
MIN_ROAD_WIDTH = 60  # 最小宽度

class TrackSegment:
    """表示单个道路片段"""
    def __init__(self, angle: float, width: float, segment_type: str):
        self.angle = angle
        self.width = width
        self.type = segment_type  # 'historical' 或 'realtime'

class Track:
    """道路类，负责数据到道路的映射与渲染"""
    def __init__(self, df_5m: pd.DataFrame):
        self.df = df_5m.reset_index(drop=True)
        self.segments = self._generate_segments()
        self.current_index = VISIBLE_TRACK_SEGMENTS  # 从这里开始显示
        self.realtime_segment = None

    def _generate_segments(self):
        """生成所有历史道路片段"""
        segments = []
        for _, row in self.df.iterrows():
            angle = self.calc_angle(row['Open'], row['Close'])
            width = self.calc_width(row['High'], row['Low'], row['Open'])
            segments.append(TrackSegment(angle, width, 'historical'))
        return segments

    def current_segment(self):
        return self.segments[self.current_index]

    @staticmethod
    def calc_angle(open_price, close_price):
        """价格变动转为角度"""
        price_change_pct = (close_price - open_price) / open_price
        return math.atan(price_change_pct * ANGLE_SCALE)

    @staticmethod
    def calc_width(high_price, low_price, open_price):
        """根据价格波动计算道路宽度"""
        volatility = (high_price - low_price) / open_price
        width = BASE_ROAD_WIDTH - volatility * WIDTH_SCALE
        return max(width, MIN_ROAD_WIDTH)

    def update_realtime_segment(self, realtime_bar):
        """实时道路更新(未完成5分钟数据)"""
        angle = self.calc_angle(realtime_bar['Open'], realtime_bar['Close'])
        width = self.calc_width(realtime_bar['High'], realtime_bar['Low'], realtime_bar['Open'])
        self.realtime_segment = TrackSegment(angle, width, 'realtime')

    def move_next(self):
        """每5分钟调用一次，推动道路向前"""
        if self.current_index < len(self.segments) - 1:
            self.current_index += 1
            self.realtime_segment = None

    def draw(self, surface):
        """完整绘制方法（中心线动态跟踪）"""
        surface.fill((0, 0, 0))  # 清屏
        initial_center_x, bottom_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT

        current_y = bottom_y
        current_center_x = initial_center_x

        # 绘制历史道路
        for offset in range(VISIBLE_TRACK_SEGMENTS - 1, -1, -1):
            idx = self.current_index - offset
            if idx < 0:
                continue
            segment = self.segments[idx]
            current_y, current_center_x = self._draw_segment(
                surface, segment, current_center_x, current_y)

        # 实时道路（如果有）
        if self.realtime_segment:
            self._draw_segment(surface, self.realtime_segment, current_center_x, current_y)


    def _draw_segment(self, surface, segment, center_x, bottom_y):
        """绘制单个道路片段（改进版）"""
        angle = segment.angle
        width = segment.width
        height = TRACK_SEGMENT_HEIGHT

        # 根据角度计算中心线的水平偏移（关键改进）
        center_offset_x = math.tan(angle) * (height / 2)
        adjusted_center_x = center_x + center_offset_x

        # 动态左右车道宽度
        if angle > 0:  # 上涨（左倾斜），左车道稍宽
            left_width = width * 0.6
            right_width = width * 0.4
        elif angle < 0:  # 下跌（右倾斜），右车道稍宽
            left_width = width * 0.4
            right_width = width * 0.6
        else:  # 平稳行情
            left_width = right_width = width / 2

        # 四个顶点计算（明确）
        top_x = adjusted_center_x + math.tan(angle) * height / 2
        bottom_x = adjusted_center_x - math.tan(angle) * height / 2

        points = [
            (bottom_x - left_width, bottom_y),            # 左下
            (bottom_x + right_width, bottom_y),           # 右下
            (top_x + right_width, bottom_y - height),     # 右上
            (top_x - left_width, bottom_y - height)       # 左上
        ]

        # 道路颜色
        road_color = (80, 80, 80) if segment.type == 'historical' else (120, 120, 120)

        # 绘制道路区域
        pygame.draw.polygon(surface, road_color, points)

        # 绘制道路中心线（动态对齐）
        pygame.draw.line(surface, (255, 255, 255), 
                        (bottom_x, bottom_y), 
                        (top_x, bottom_y - height), 2)

        # 边界线
        pygame.draw.lines(surface, (150, 150, 150), False, points[:2], 2)  # 下边界
        pygame.draw.lines(surface, (150, 150, 150), False, points[2:], 2)  # 上边界

        # 护栏线（止损线）清晰绘制
        pygame.draw.line(surface, (200, 0, 0),
                        (bottom_x - left_width - 5, bottom_y),
                        (top_x - left_width - 5, bottom_y - height), 3)

        pygame.draw.line(surface, (200, 0, 0),
                        (bottom_x + right_width + 5, bottom_y),
                        (top_x + right_width + 5, bottom_y - height), 3)

        return bottom_y - height, top_x  # 返回新的顶部中心X坐标

