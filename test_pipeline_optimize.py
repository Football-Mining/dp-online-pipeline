from reader import ffmpegMultiTrackReader
from preprocess import undistort_image
from image_transformer import ImageTransformer
import cv2
import time

import subprocess


# 一些配置样例

# 球场角点以及cameraman最左侧、中间、最右侧的位置
points_config = {
    "polygon": [[1529, 705], [47, 1027], [2741, 1988], [5390, 1009], [3921, 717]],
    "left_crop_size": [0, 677],
    "right_crop_size": [2687, 759],
    "left_most_setting": [1911, 925, 40],
    "middle_point_setting": [2693, 1006, 47],
    "right_most_setting": [3485, 930, 41],
}
dp_live_config = {
    "match_id": "LRcEqMhaPo9",
    "device_id": "test_4K",
    "init_frame_seconds": 1.5,
    "skip_frame_seconds": 0,
    "debug": False,
    "pull_url": "MOV_0082.mp4",
}

img_size = (2160, 3840, 3)
video_path = "/home/mhliu/robot_demo_sat/MOV_0082.mp4"

reader = ffmpegMultiTrackReader(video_path, img_size)

img_transformer = ImageTransformer(
    points_config, dp_live_config, warper_type="spherical"
)

left, right = reader.next()

# 目标x坐标from cameraman
target_x = 2200

# img_transformer.precalculate()
while True:
    t0 = time.time()
    left = img_transformer.compute_img(left, target_x, "left")
    right = img_transformer.compute_img(right, target_x, "right")

    result = img_transformer.transform(left, right, target_x)
    print(f"Transform Time: {time.time() - t0}")


# cv2.imwrite(f"test_left_1201.png", left)
# cv2.imwrite(f"test_right_1201.png", right)
# cv2.imwrite(f"test_result_1201.png", result)

# 最后ffmpeg编码
