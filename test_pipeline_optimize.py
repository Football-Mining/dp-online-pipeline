from reader import ffmpegMultiTrackReader
from preprocess import undistort_image
from image_transformer import ImageTransformer
import cv2

import subprocess


# 一些配置样例

# 球场角点以及cameraman最左侧、中间、最右侧的位置
points_config = {"polygon": [[1222, 524], [347, 840], [2085, 1472], [3755, 750], [2874, 477]], "left_crop_size": [342, 492], "right_crop_size": [2367, 464], "left_most_setting": [1446, 615, 33], "middle_point_setting": [1983, 763, 42], "right_most_setting": [2646, 590, 33]}

dp_live_config = {
    "match_id": "3KbfMFiNKX2",
    "device_id": "test_undistort",
    "init_frame_seconds": 1.5,
    "skip_frame_seconds": 0,
    "debug": False,
    "pull_url": "MOV_0082.mp4"
}

img_size = (1520, 2688, 3)
video_path = "MOV_0082.mp4"

reader = ffmpegMultiTrackReader(video_path, img_size)


img_transformer = ImageTransformer(points_config, dp_live_config, warper_type="spherical")

left, right = reader.next()

# 目标x坐标from cameraman
target_x = 2200

left = img_transformer.compute_img(left, target_x, "left")
right = img_transformer.compute_img(right, target_x, "right")

result = img_transformer.transform(left, right, target_x)




cv2.imwrite(f'test_left_1201.png', left)
cv2.imwrite(f'test_right_1201.png', right)
cv2.imwrite(f'test_result_1201.png', result)

# 最后ffmpeg编码
