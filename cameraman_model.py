from collections import deque
import numpy as np
from filterpy.kalman import KalmanFilter
import random

class KalmanFilter1D:
    def __init__(self, initial_x, variance_position=10.0, variance_measurement=40.0):
        """
        初始化一维卡尔曼滤波器。
        
        :param initial_x: 初始状态估计(x 坐标)
        :param dt: 时间步长
        :param variance_position: 位置的方差
        :param variance_measurement: 测量的方差
        """
        self.filter = KalmanFilter(dim_x=1, dim_z=1)
        
        # 设置状态转移矩阵
        self.filter.F = np.array([[1.0]])
        
        # 设置观测矩阵
        self.filter.H = np.array([[1.0]])
        
        # 设置状态协方差矩阵
        self.filter.P = np.array([[variance_position]])
        
        # 设置过程噪声协方差矩阵
        self.filter.Q = np.array([[0.01]])  # 可以根据实际情况调整
        
        # 设置观测噪声协方差矩阵
        self.filter.R = np.array([[variance_measurement]])
        
        # 设置初始状态
        self.filter.x = np.array([[initial_x]])

    def filter_measurement(self, measurement):
        """
        对单个观测值进行卡尔曼滤波。
        
        :param measurement: 单个观测值
        :return: 滤波后的状态估计
        """
        self.filter.predict()
        self.filter.update(measurement)
        return self.filter.x[0, 0]
    
def bound(value, minimum, maximum):
    return min(max(value, minimum), maximum)





class CameramanModel(object):
    def __init__(self, points_config, fps=30):
        self.points_config = points_config

        # 要记忆的秒数
        self.memory_length = 0.5
        self.fps = fps

        self.left_most = points_config["left_most_setting"][0]
        self.right_most = points_config["right_most_setting"][0]


        # 记录过去$memory_length$秒的球员位置
        self.player_pos_memory = deque(maxlen=int(self.fps*self.memory_length))
        self.player_max_pos_memory = deque(maxlen=int(self.fps*self.memory_length))
        self.player_min_pos_memory = deque(maxlen=int(self.fps*self.memory_length))
        self.last_positions = None


        # 记录过去15帧的球位置
        self.ball_pos_memory = deque(maxlen=15)

        self.focus_slider = 0.5 # 0代表关注最左侧球员，1代表关注最右侧球员

        # 最高速度(像素/秒)
        self.speed_max = 150 * self.memory_length

        # 缓冲的像素点数量(0-50)，数值越大，则镜头越稳定但攻防转换跟随越慢
        self.buffer_pixels = 30

        self.kalman_filters = {}
        self.slider_filter = KalmanFilter1D(0.5, 0.05, 0.5)

        self.inited = False

    def get_kalman_predict(self, name, observe):
        variance_position = 10.0
        variance_measurement = 40.0
        if name not in self.kalman_filters:
            # if "speed" in name:
            #     variance_position = 5
            #     variance_measurement = 20
            self.kalman_filters[name] = KalmanFilter1D(initial_x=observe, variance_measurement=variance_measurement, variance_position=variance_position)
            result = observe
        else:
            kalman_filter = self.kalman_filters[name]
            result = kalman_filter.filter_measurement(observe)
        return result
        
    def parse_noise_player(self, positions, threshold=200):
        if self.last_positions is None or len(positions) <= 3:
            self.last_positions = positions
            return positions
        
        positions = sorted(positions)

        last_min = min(self.last_positions)
        last_max = max(self.last_positions)


        if last_min - positions[0] > threshold and positions[1] - positions[0] > threshold:
            positions.remove(min(positions))
        if positions[-1] - last_max > threshold and positions[-1] - positions[-2] > threshold:
            positions.remove(max(positions))

        self.last_positions = positions
        return positions

    def get_accumulated_speed(self, pos_memory):
        last_pos = None
        accumulated_speed = 0
        for pos in pos_memory:
            if last_pos is None:
                last_pos = pos
                continue
            accumulated_speed += pos - last_pos
            last_pos = pos
        return accumulated_speed


    def ball_player_pos_merge(self, player_pos, ball_pos):
        return 0.6*player_pos + 0.4*ball_pos
    
    def init_historys(self, player_positions, ball_positions):
        if len(player_positions) == 0:
            self.player_pos_memory.append(1200)
            self.player_max_pos_memory.append(1200)
            self.player_min_pos_memory.append(1200)
        else:
            self.player_pos_memory.append(np.mean(player_positions))
            self.player_max_pos_memory.append(max(player_positions))
            self.player_min_pos_memory.append(min(player_positions))

        if len(ball_positions) == 0:
            self.ball_pos_memory.append(self.player_pos_memory[-1])
        else:
            self.ball_pos_memory.append(np.mean(ball_positions))
        self.inited = True


    def predict(self, player_positions, ball_positions, debug_mode=False):
        # 目前的方法只使用x坐标
        player_positions = [pos[0] for pos in player_positions]
        ball_positions = [pos[0] for pos in ball_positions]



        if not self.inited:
            self.init_historys(player_positions, ball_positions)

        if len(player_positions) == 0:
            player_positions.append(self.player_pos_memory[-1])

        # player_positions = self.parse_noise_player(player_positions)

        mean_player_pos = np.mean(player_positions)
        mean_player_pos = self.get_kalman_predict("mean_pos", mean_player_pos)
        max_player_pos = max(player_positions)
        min_player_pos = min(player_positions)

        

        self.player_pos_memory.append(mean_player_pos)
        self.player_max_pos_memory.append(max_player_pos)
        self.player_min_pos_memory.append(min_player_pos)

        player_mean_speed = self.get_accumulated_speed(self.player_pos_memory)
        player_mean_speed = self.get_kalman_predict("player_mean_speed", player_mean_speed)
        player_max_speed = self.get_accumulated_speed(self.player_max_pos_memory)
        player_max_speed = self.get_kalman_predict("player_max_speed", player_max_speed)
        player_min_speed = self.get_accumulated_speed(self.player_min_pos_memory)
        player_min_speed = self.get_kalman_predict("player_min_speed", player_min_speed)

        side_speed = player_max_speed if abs(player_max_speed) > abs(player_min_speed) else player_min_speed
        speed = bound(np.mean([side_speed, player_mean_speed]), -self.speed_max, self.speed_max)

        
        
        self.focus_slider = (bound(speed, -self.speed_max, self.speed_max) / self.speed_max) / 2 + 0.5
        self.focus_slider = self.slider_filter.filter_measurement(self.focus_slider)

        focus_side = min_player_pos if self.focus_slider < 0.5 else max_player_pos
        player_focus = mean_player_pos + ((abs(self.focus_slider - 0.5) * 3 + 0.5) / 2.) * (focus_side - mean_player_pos)
        
        ball_detected = len(ball_positions)

        ball_pos = np.mean([self.ball_pos_memory[-1], player_focus]) if len(ball_positions) == 0 else np.mean(ball_positions)
        ball_pos = self.get_kalman_predict("ball_pos", ball_pos)
        
        self.ball_pos_memory.append(ball_pos)

        

        target_x = self.ball_player_pos_merge(player_focus, ball_pos)
        target_x = bound(target_x, self.left_most - self.buffer_pixels, self.right_most + self.buffer_pixels)
        
        target_x_filtered = self.get_kalman_predict("focus", target_x)
        target_x_filtered = bound(target_x_filtered, self.left_most, self.right_most)

        if debug_mode:
            return target_x, target_x_filtered, mean_player_pos, speed, player_focus, ball_pos, max_player_pos, min_player_pos, ball_detected, self.focus_slider, player_positions
        else:
            return target_x_filtered
    

def court_area(points):
    points = sorted(points, key=lambda x: x[0])
    points_left = sorted(points[0:2], key=lambda x: x[1])
    points_right = sorted(points[-2:], key=lambda x: x[1])
    print(points_left, points_right)
    print(abs(points_left[0][0] - points_right[0][0]), abs(points_left[1][0] - points_right[1][0]), abs(points_left[0][1] - points_left[1][1]))
    area = (abs(points_left[0][0] - points_right[0][0]) + abs(points_left[1][0] - points_right[1][0])) * abs(points_left[0][1] - points_left[1][1]) / 2
    return area



def test_kalman():
    kalman_filter = KalmanFilter1D(1500)
    for i in range(50):
        j = random.randint(-50, 50)
        
        print(kalman_filter.filter_measurement(j+1500))
    for i in range(50):
        print(kalman_filter.filter_measurement(1500-i*10))


if __name__ == '__main__':
    from equirectangular_model import EquirectangularModel
    import cv2
    from utils import get_and_init_stitcher
    # from tqdm import tqdm
    # points = [(296, 439), (882, 196), (2658, 514), (2097, 255), (1500,194), (1572, 721)]

    # left_most_setting = (1080, 367, 31)
    # middle_point_setting = (1500, 470, 42)
    # right_most_setting = (1920, 389, 31)

    stitcher = get_and_init_stitcher()


    left = cv2.imread("left.png", cv2.IMREAD_COLOR)
    right = cv2.imread("right.png", cv2.IMREAD_COLOR)
    panorama = stitcher.stitch(left, right)

    cv2.imwrite(f'panorama_new.png', panorama)

    # e.set_funcs_with_init_settings(left_most_setting, middle_point_setting, right_most_setting)

    # for i in tqdm(range(11)):
    #     x = 1080 + i/10*(1920-1080)
    #     res = e.GetPerspectiveFromCoordStupid(img, x)
    #     cv2.imwrite(f'test_res_{i}.png', res)

    # test_kalman()

    # points = sorted(points, key=lambda x: x[0])

    # left_points = points[0:4].copy()
    # left_points[2:4] =  sorted(left_points[2:4], key=lambda x: x[1], reverse=True)
    # right_points = points[2:6].copy()
    # left_points[0:2] =  sorted(left_points[0:2], key=lambda x: x[1], reverse=True)

    # print(left_points, right_points)
    # print(court_area(points))

    # cameraman_model = CameramanModel()