import cv2
import os
import numpy as np
from tqdm import tqdm
import time

x_horizon = 210     # x轴方向视野的角度（两个镜头融合起来，即全景视角中的视野宽度）
y_horizon = 70     # 同上，y轴方向

MATRIX_SAMPLE_RATE = 1


def xyz2lonlat(xyz):
    # print(xyz)

    atan2 = np.arctan2
    asin = np.arcsin

    t0 = time.time()

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    t1 = time.time()
    xyz_norm = xyz / norm
    t2 = time.time()
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]
    t3 = time.time()

    lon = atan2(x, z)
    lat = asin(y)
    t4 = time.time()

    # print(lon)
    # print(lat.shape)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    t5 = time.time()
    # print(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (x_horizon / 360. * np.pi) + 0.5) * (shape[1] - 1)
    # print(X)
    Y = (lonlat[..., 1:] / (y_horizon / 360. * np.pi) + 0.5) * (shape[0] - 1)

    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out


class EquirectangularModel:
    def __init__(self, size=None):
        # self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        self._height, self._width = None, None
        self.xyz = None
        self.x_func = None
        self.y_func = None

        self.x = None
        self.y = None
        if size is not None:
            [self._height, self._width] = size
            # print(self._width, self._height)
            self.get_funcs()
        # cp = self._img.copy()
        # w = self._width
        # self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        # self._img[:, w/8:, :] = cp[:, :7*w/8, :]





    def get_funcs(self):
        if self.y_func is None:

            self.y_func = lambda x: int(((-5.625*(x/self._width-0.5)**2+0.47))*self._height)  # 输入x得到y，是一条二次函数曲线，在中场的时候最低点

            self.fov_func = lambda x: (48 - 19 * abs(x - self._width // 2) / (self._width / 6))  # 输入x得到fov，绝对值一次函数，在中场的时候视角最远

    def GetPerspective(self, image):
        self.get_size(image)
        # print(self.XY[0])
        
        self.XY = self.XY.astype(np.int32).astype(np.float32)
        # print(self.XY[0])
        persp = cv2.remap(image, self.XY[..., 0], self.XY[..., 1], cv2.INTER_CUBIC)

        return persp

    def GetPerspective_cuda(self, image):
        self.get_size(image)
        
        # 将输入图像上传到 GPU
        d_src = cv2.cuda_GpuMat(image)
        # d_src.upload(image)

        # 将映射矩阵上传到 GPU
        d_map1 = cv2.cuda_GpuMat(self.XY[..., 0])
        # d_map1.upload(self.XY[..., 0])

        d_map2 = cv2.cuda_GpuMat(self.XY[..., 1])
        # d_map2.upload(self.XY[..., 1])

        # # 创建目标图像
        # d_dst = cv2.cuda_GpuMat()
        # type_int = cv2.CV_8UC3 if image.dtype == np.uint8 else cv2.CV_32FC3

        # d_dst.create((self._width, self._height), type_int)

        # 使用 GPU 加速的 remap 函数
        dst = cv2.cuda.remap(d_src, d_map1, d_map2, interpolation=cv2.INTER_CUBIC)

        # 下载结果
        persp = dst.download()

        return persp

    def get_matrix(self, THETA, PHI, save=False):
        t0 = time.time()
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)

        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = self.xyz @ R.T

        # print("xyz:", time.time() - t0)

        t0 = time.time()
        # rotation_angle = -THETA / 45. * 20
        rotation_angle = -THETA / 45. * 10
        # print(rotation_angle)
        rotate = np.array([
            [np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle)), 0],
            [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle)), 0],
            [0, 0, 1]
        ])
        xyz = xyz @ rotate

        # print("rotate", time.time() - t0)

        t0 = time.time()
        lonlat = xyz2lonlat(xyz)
        self.XY = lonlat2XY(lonlat, shape=[self._height, self._width]).astype(np.float32)
        # print("lonlat2XY", time.time() - t0)

        if save:
            if not os.path.exists("matrices"):
                os.mkdir("matrices")
            np.save("matrices/mat_{}.npy".format(self.x), self.XY)

    def set_funcs_with_init_settings(self, left_most_setting, middle_point_setting, right_most_setting):
        # 使得视角和y值线性变化
        # print(left_most_setting)
        a_left = (left_most_setting[1] - middle_point_setting[1]) / ((middle_point_setting[0] - left_most_setting[0]) ** 2)
        a_right = (right_most_setting[1] - middle_point_setting[1]) / ((middle_point_setting[0] - right_most_setting[0]) ** 2)

        y_func_left = lambda x: (middle_point_setting[1] + a_left * (x - middle_point_setting[0]) ** 2)
        y_func_right = lambda x: (middle_point_setting[1] + a_right * (x - middle_point_setting[0]) ** 2)
        # y_func_left = lambda x: (left_most_setting[1] + (middle_point_setting[1] - left_most_setting[1]) * (x - left_most_setting[0]) / (middle_point_setting[0] - left_most_setting[0]))
        # y_func_right = lambda x: (middle_point_setting[1] + (right_most_setting[1] - middle_point_setting[1]) * (1 - (right_most_setting[0] - x) / (right_most_setting[0] - middle_point_setting[0])))

        fov_func_left = lambda x: (left_most_setting[2] + (middle_point_setting[2] - left_most_setting[2]) * (x - left_most_setting[0]) / (middle_point_setting[0] - left_most_setting[0]))
        fov_func_right = lambda x: (middle_point_setting[2] + (right_most_setting[2] - middle_point_setting[2]) * (1- (right_most_setting[0] - x) / (right_most_setting[0] - middle_point_setting[0])))


        # self.fov_func = fov_func_left
        # self.y_func = y_func_left
        self.fov_func = lambda x: (fov_func_left(x) if x <= middle_point_setting[0] else fov_func_right(x)) 
        self.y_func = lambda x: (y_func_left(x) if x <= middle_point_setting[0] else y_func_right(x)) 



    def get_xyz(self, FOV):
        height, width = 1080, 1920
        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ], np.float32)
        K_inv = np.linalg.inv(K)
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        self.xyz = xyz @ K_inv.T

    def get_size(self, image):
        if self._height is None:
            self._height, self._width = image.shape[0], image.shape[1]
            self.get_funcs()

    def get_max_fov(self, y, min_y, max_y, y_length):
        # print(y, min_y, max_y, y_length)
        # print((y - min_y) * 180 / y_length, (max_y - y) * 180 / y_length)
        max_fov = min(abs(y - min_y) * 180 / y_length, abs(max_y - y) * 180 / y_length)-5
        return max_fov

    def get_cood_from_x(self, x):
        print(x, self.y_func(x), self.fov_func(x))

    def GetPerspectiveFromCoordStupid(self, image, x, y=None, fov=None):
        self.get_size(image)
        if y is None:
            y = int(self.y_func(x))
            # print(x, y)
            # print(self.fov_func(x))
        THETA = ((x - self._width / 2.) / (self._width / 2.)) * 90
        # print(THETA)
        PHI =  -((y - self._height / 2.) / (self._height / 2.)) * 45
        # print(PHI)

        self.x = x
        self.y = y
        if fov is None:
            fov = self.fov_func(x)
        # print(x, y, fov)
        self.get_xyz(fov)
        self.get_matrix(THETA, PHI, save=False)
            # self.XY = np.load("mat.npy")

        return self.GetPerspective(image)
    
    def get_mat(self, x, y=None, fov=None):
        t0 = time.time()
        if y is None:
            y = int(self.y_func(x))
        THETA = ((x - self._width / 2.) / (self._width / 2.)) * 90
        PHI = -((y - self._height / 2.) / (self._height / 2.)) * 45
        if fov is None:
            fov = self.fov_func(x)
        # print("compute time", time.time() - t0)
        # print(x, y, fov)
        t0 = time.time()
        self.get_xyz(fov)
        # print("get_xyz time", time.time() - t0)
        t0 = time.time()
        self.get_matrix(THETA, PHI)
        # print("get_matrix time", time.time() - t0)
        return self.XY

    def GetPerspectiveFromCoord(self, image, x, y=None, fov=None, fast_mode=False):
        t0 = time.time()
        self.get_size(image)
        t1 = time.time()
        # print("get size time: {}".format(t1-t0))
        if y is None:
            y = int(self.y_func(x))
        

        if self.x is None or self.y is None or self.x != x or self.y != y:

            self.x = x
            self.y = y
            if fast_mode and os.path.exists("matrices/mat_{}.npy".format(x//MATRIX_SAMPLE_RATE*MATRIX_SAMPLE_RATE)):
                # print("shit")
                self.XY = np.load("matrices/mat_{}.npy".format(x//MATRIX_SAMPLE_RATE*MATRIX_SAMPLE_RATE))
                t2 = time.time()
                # print("load matrix time: {}".format(t2-t1))
            else:
                THETA = ((x - self._width / 2.) / (self._width / 2.)) * 90
                PHI = -((y - self._height / 2.) / (self._height / 2.)) * 45
                if fov is None:
                    fov = self.fov_func(x)
                self.get_xyz(fov)
                t3 = time.time()
                # print("get_xyz time: {}".format(t3-t1))
                self.get_matrix(THETA, PHI)
                t4 = time.time()
                # print("get_matrix time: {}".format(t4-t3))
            # self.XY = np.load("mat.npy")
        t5 = time.time()
        result = self.GetPerspective(image)
        t6 = time.time()
        # print("GetPerspective time: {}".format(t6-t5))
        return result


    def get_angles_from_points(self, x, y):
        THETA = ((x - self._width / 2.) / (self._width / 2.)) * 90
        # print(THETA)
        PHI = -((y - self._height / 2.) / (self._height / 2.)) * 45
        return THETA, PHI



if __name__ == '__main__':

    import json
    from utils import get_points_config_path, get_regist_imgs, get_and_init_stitcher
    
    config_path = "test_config.json"
    dp_live_config = json.load(open(config_path, "r"))
    match_id = dp_live_config["match_id"]
    device_id = dp_live_config["device_id"]
    
    left_channel = f"{device_id}_{match_id}_left"
    right_channel = f"{device_id}_{match_id}_right"

    points_config = json.load(open(get_points_config_path(device_id, match_id), "r"))
    e = EquirectangularModel()
    panorama = cv2.imread("panorama_new.png", cv2.IMREAD_COLOR)
    e.GetPerspectiveFromCoordStupid(panorama, 1500)
    e.set_funcs_with_init_settings(points_config["left_most_setting"], points_config["middle_point_setting"], points_config["right_most_setting"])

    # t0 = time.time()
    # e.get_mat(1500)
    # print("get_mat time:", time.time() - t0)
    # left_most_setting = (870, 360, 34)
    # middle_point_setting = (1350, 485, 48)
    # right_most_setting = (1750, 355, 36)

    # e.set_funcs_with_init_settings(left_most_setting, middle_point_setting, right_most_setting)

    res = e.GetPerspectiveFromCoordStupid(panorama, points_config["left_most_setting"][0])
    cv2.imwrite('test_left_most2.png', res)
    res = e.GetPerspectiveFromCoordStupid(panorama, points_config["middle_point_setting"][0])
    cv2.imwrite('test_middle2.png', res)
    res = e.GetPerspectiveFromCoordStupid(panorama, points_config["right_most_setting"][0])
    cv2.imwrite('test_right_most2.png', res)

    """调参数"""
    # persp = e.GetPerspectiveFromCoordStupid(img, x_value)
    # # cv2.imshow("Image", persp)

    # #
    # cv2.namedWindow("Image")
    # cv2.createTrackbar("x_rate", "Image",  x_horizon, 360, update1)
    # cv2.createTrackbar("y_rate", "Image", y_horizon, 360, update2)

    # cv2.waitKey(0)

    # """保存矩阵"""
    # e.save_all_matrices(range(700, 1700, MATRIX_SAMPLE_RATE))


