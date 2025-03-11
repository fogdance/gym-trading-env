import pygame
import pandas as pd
import talib
import numpy as np

# 常量定义
TRACK_SEGMENT_HEIGHT = 60
BASE_ROAD_WIDTH = 160   # 屏幕宽度的 1/5（800 * 0.2 = 160）
MIDDLE_OFFSET_RANGE = 200  # 中心线最大偏移范围（像素）

class TrackSegment:
    def __init__(self, date, upper, middle, lower, road_width, close):
        self.date = date
        self.close = close
        self.upper = upper   # 映射后的上轨
        self.middle = middle # 映射后的中轨
        self.lower = lower   # 映射后的下轨
        self.road_width = road_width

class Track:
    def __init__(self, draw_rect):
        """
        初始化赛道。draw_rect: (x, y, width, height)
        """
        self.draw_rect = draw_rect
        self.df = None
        self.segments = []

        self.track_segment_height = TRACK_SEGMENT_HEIGHT
        self.avg_band_width = None
        self.band_widths = []

        # 滑动窗口大小
        self.window_size = 60

        # 当前可用的映射区间
        self.value_min = 0.0
        self.value_max = 1.0

        # 上一次映射区间，用于“平滑过渡”和“阈值比较”
        self.old_min = None
        self.old_max = None

        self.initialized = False
        self.max_segments = 0

        # 以下两个参数可根据需求调整
        # (1) 扩大映射区间的阈值, 超过此阈值立即更新
        self.expand_threshold_factor = 0.05
        # (2) 收缩时的阈值, 与扩大可以相同或不同(非对称)
        self.shrink_threshold_factor = 0.05

        # 缓冲比例
        self.margin_factor = 0.1
        # 平滑系数 alpha：接近0则快，接近1则慢
        # 在此采用“快速扩大、慢速收缩”的做法：expanding时alpha小，shrinking时alpha大
        self.alpha_expand = 0.3
        self.alpha_shrink = 0.7

    def step(self, df):
        """
        接受新的 DataFrame 数据并更新赛道。
        如果是首次，进行初始化；否则只追加最新段。
        """
        self.df = df
        self.window_size = len(df)
        if not self.initialized:
            self.segments = self.generate_segments(df)
            self.initialized = True
        else:
            self.append_new_segments(df)

    def generate_segments(self, df):
        """
        首次批量初始化赛道段。
        """
        close_prices = df['Close'].values
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)

        # 计算布林带宽度平均值，仅用于道路宽度缩放
        self.band_widths = []
        for i, (u, m, l) in enumerate(zip(upper, middle, lower)):
            if np.isnan(u) or np.isnan(m) or np.isnan(l):
                u, m, l = close_prices[i], close_prices[i], close_prices[i]
            self.band_widths.append(u - l)
        self.avg_band_width = np.mean(self.band_widths) if len(self.band_widths) else 1.0

        # 更新当前映射区间
        self._update_value_range(close_prices)

        segments = []
        for i, (date, u, m, l) in enumerate(zip(df.index, upper, middle, lower)):
            if np.isnan(u) or np.isnan(m) or np.isnan(l):
                u, m, l = close_prices[i], close_prices[i], close_prices[i]

            u_mapped = self.map_to_screen(u, 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])
            m_mapped = self.map_to_screen(m, 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])
            l_mapped = self.map_to_screen(l, 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])
            c_mapped = self.map_to_screen(close_prices[i], 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])

            bw = u - l
            if bw < 0:
                bw = 0
            road_w = BASE_ROAD_WIDTH * (bw / self.avg_band_width) if self.avg_band_width != 0 else BASE_ROAD_WIDTH
            road_w = min(BASE_ROAD_WIDTH * 4, max(BASE_ROAD_WIDTH * 0.5, road_w))

            seg = TrackSegment(date, u_mapped, m_mapped, l_mapped, road_w, c_mapped)
            segments.append(seg)

        self.max_segments = len(segments)
        return segments

    def append_new_segments(self, df):
        """
        追加最新一行数据到 segments。
        """
        close_prices = df['Close'].values
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)

        new_date = df.index[-1]
        u, m, l = upper[-1], middle[-1], lower[-1]
        if np.isnan(u) or np.isnan(m) or np.isnan(l):
            u, m, l = close_prices[-1], close_prices[-1], close_prices[-1]

        # 更新映射区间
        self._update_value_range(close_prices)

        bw = (u - l) if (not np.isnan(u) and not np.isnan(l)) else 0.0
        if self.avg_band_width is None or self.avg_band_width < 1e-9:
            self.avg_band_width = 1.0
        road_w = BASE_ROAD_WIDTH * (bw / self.avg_band_width)
        road_w = min(BASE_ROAD_WIDTH * 4, max(BASE_ROAD_WIDTH * 0.5, road_w))

        u_mapped = self.map_to_screen(u, 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])
        m_mapped = self.map_to_screen(m, 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])
        l_mapped = self.map_to_screen(l, 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])
        c_mapped = self.map_to_screen(close_prices[-1], 0.25 * self.draw_rect[2], 0.75 * self.draw_rect[2])

        seg = TrackSegment(new_date, u_mapped, m_mapped, l_mapped, road_w, c_mapped)

        if len(self.segments) >= self.max_segments:
            self.segments.pop(0)
        self.segments.append(seg)

    def map_to_screen(self, value, screen_min, screen_max):
        """
        将数值 value 线性映射到 screen_min ~ screen_max。
        """
        denom = self.value_max - self.value_min
        if denom < 1e-9:
            denom = 1e-9
        normalized = (value - self.value_min) / denom
        return screen_min + normalized * (screen_max - screen_min)

    def draw(self, surface):
        """
        在指定 surface 上绘制赛道与中轴线。
        """
        left, top, width, height = self.draw_rect
        num_segments = len(self.segments)
        if num_segments == 0:
            return

        self.track_segment_height = (height - top) / num_segments

        temp_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        temp_surface.fill((0, 0, 0, 0))

        centerline_points = []
        current_y = top

        # 逆序绘制赛道
        for i in range(num_segments - 1, -1, -1):
            seg = self.segments[i]
            center_x = seg.middle
            road_width = seg.road_width
            left_w = road_width / 2
            right_w = road_width / 2

            max_left = center_x
            max_right = width - center_x
            if left_w > max_left or right_w > max_right:
                scale = min(max_left / left_w, max_right / right_w)
                left_w *= scale
                right_w *= scale

            road_pts = [
                (center_x - left_w, current_y),
                (center_x + right_w, current_y),
                (center_x + right_w, current_y + self.track_segment_height),
                (center_x - left_w, current_y + self.track_segment_height)
            ]
            pygame.draw.polygon(temp_surface, (100, 100, 100), road_pts)

            # 中线使用 close 值
            centerline_points.append((seg.close, current_y + self.track_segment_height / 2))

            current_y += self.track_segment_height

        # 绘制中线
        pygame.draw.lines(temp_surface, (255, 255, 255), False, centerline_points, 2)
        surface.blit(temp_surface, (left, top))

        if False:
            # 调试：显示最新分段日期
            overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 0))
            if len(self.segments) > 0:
                latest_date = self.segments[-1].date
                self.draw_date_center(overlay, latest_date)
            surface.blit(overlay, (0, 0))

    def draw_date_center(self, surface, date):
        font = pygame.font.SysFont(None, 96)
        date_str = str(date)
        text_surface = font.render(date_str, True, (255, 0, 0))
        screen_rect = surface.get_rect()
        x = screen_rect.centerx - text_surface.get_width() // 2
        y = screen_rect.centery - text_surface.get_height() // 2
        surface.blit(text_surface, (x, y))

    # ----------------------------------------------------------
    #  以下是“动态更新映射区间”的内部方法，处理滑动窗口、缓冲、阈值、非对称策略、平滑过渡
    # ----------------------------------------------------------
    def _update_value_range(self, close_prices):
        """
        仅使用最近 window_size 条收盘价，计算局部最值并加缓冲。
        再与旧区间做比较，若超出阈值则更新映射区间。
        采用“快速扩张、慢速收缩”的非对称策略 + 指数平滑方式。
        """
        if len(close_prices) == 0:
            return

        # 1) 取滑动窗口范围
        data_slice = close_prices[-self.window_size:]
        local_min = float(np.min(data_slice))
        local_max = float(np.max(data_slice))
        base_range = local_max - local_min
        if base_range < 1e-9:
            base_range = 1e-9

        # 2) 缓冲
        new_min = local_min - self.margin_factor * base_range
        new_max = local_max + self.margin_factor * base_range

        # 首次初始化
        if self.old_min is None or self.old_max is None:
            self.old_min = new_min
            self.old_max = new_max
            self.value_min = new_min
            self.value_max = new_max
            return

        old_range = self.old_max - self.old_min
        if old_range < 1e-9:
            old_range = 1e-9

        # 3) 判断是否超过阈值
        #    (1) 扩大: 超出 expand_threshold_factor * old_range 时，立即更新
        #    (2) 收缩: 超出 shrink_threshold_factor * old_range 时，进行慢速收缩
        expand_thresh = self.expand_threshold_factor * old_range
        shrink_thresh = self.shrink_threshold_factor * old_range

        # 快速扩张情况
        # 如果 new_max 高于旧 max 太多，或 new_min 低于旧 min 太多 -> 立即拉大
        need_expand_top = (new_max > self.old_max + expand_thresh)
        need_expand_bot = (new_min < self.old_min - expand_thresh)

        # 慢速收缩情况
        # 如果 new_max < 旧区间较多 或 new_min > 旧区间较多 -> 可以考虑收紧
        need_shrink_top = (new_max < self.old_max - shrink_thresh)
        need_shrink_bot = (new_min > self.old_min + shrink_thresh)

        # 4) 不同情况采用不同 alpha
        #    (a) 扩张时 alpha 较小 -> 变化更快
        #    (b) 收缩时 alpha 较大 -> 变化更慢(迟滞)
        #    (c) 无需更新 -> 保持原状
        if need_expand_top:
            self.old_max = (1 - self.alpha_expand) * new_max + self.alpha_expand * self.old_max
        elif need_shrink_top:
            self.old_max = (1 - self.alpha_shrink) * new_max + self.alpha_shrink * self.old_max

        if need_expand_bot:
            self.old_min = (1 - self.alpha_expand) * new_min + self.alpha_expand * self.old_min
        elif need_shrink_bot:
            self.old_min = (1 - self.alpha_shrink) * new_min + self.alpha_shrink * self.old_min

        # 5) 赋值给当前映射区间
        self.value_min = self.old_min
        self.value_max = self.old_max
        if self.value_max - self.value_min < 1e-9:
            self.value_max = self.value_min + 1e-3
