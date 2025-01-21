from dp_stitching.details_stitcher import DetailsStitcher
import cv2
import os
from pathlib import Path

MATRICES_ROOT_DIR = "./matrices" # "/ssd/matrices"

def read_ref_imgs(device_id="test_4K"):
    """
    We need the ref img dir to be like
    left | right
    Each directory contains the images for the left and right cameras, and pairs are matched by name
    This func reads ref img paths, return a list in a format of [left1, right1, left2, right2, ...]
    """
    ref_imgs = []
    ref_img_dir = Path(f"camera_configs/{device_id}")
    left_img_dir = ref_img_dir / "left"
    right_img_dir = ref_img_dir / "right"
    left_img_paths = sorted(left_img_dir.glob("*.png"))
    right_img_paths = sorted(right_img_dir.glob("*.png"))
    
    for left_img_path, right_img_path in zip(left_img_paths, right_img_paths):
        ref_imgs.append(cv2.imread(str(left_img_path)))
        ref_imgs.append(cv2.imread(str(right_img_path)))

    return ref_imgs


def get_regist_imgs(device_id="test_4K"):
    imgs = read_ref_imgs(device_id)
    return imgs[0], imgs[1]


def get_points_config_path(device_id, match_id):
    return f"camera_configs/{device_id}/{match_id}/points_config.json"


def get_audio_path(device_id, match_id):
    return f"{device_id}_{match_id}.aac"


def get_pull_url(channel_name):
    return f"http://jushoop-live-videos.oss-cn-shanghai.aliyuncs.com/{channel_name}/playlist.m3u8"


def get_and_init_stitcher(device_id=None, warper_type=None):
    # cv2.ocl.setUseOpenCL(False)  # 显式启用 OpenCL
    if warper_type is None:
        warper_type = (
            "spherical" # gpu" if cv2.cuda.getCudaEnabledDeviceCount() else "spherical"
        )
    if cv2.cuda.getCudaEnabledDeviceCount():
        cv2.cuda.setDevice(0)
    stitcher = DetailsStitcher(warper_type=warper_type)

    regist_imgs = get_regist_imgs(device_id)
    stitcher.regist_multiple_image(*regist_imgs)
    stitcher.initialize_camera_from_features()
    # stitcher.get_mat_from_file()
    stitcher.initialize_warp()
    return stitcher


def get_points_config(
    dp_live_config,
    origin_points,
    left_most_setting,
    middle_point_setting,
    right_most_setting,
):
    stitcher = get_and_init_stitcher(
        device_id=dp_live_config["device_id"], warper_type="spherical"
    )
    points = origin_points
    points = sorted(points, key=lambda x: x[0])
    # 顺序
    polygon = (
        sorted(points[:2], key=lambda x: x[1], reverse=False)
        + [sorted(points[2:4], key=lambda x: x[1], reverse=True)[0]]
        + sorted(points[4:], key=lambda x: x[1], reverse=True)
    )
    origin_court_points = [
        stitcher.map_point_from_parorama(point, 2688, 1520, "left")
        for point in polygon[:2]
    ] + [
        stitcher.map_point_from_parorama(point, 2688, 1520, "right")
        for point in polygon[3:]
    ]
    left_crop_size = list(
        map(int, [origin_court_points[1][0], origin_court_points[0][1] - 40])
    )
    right_crop_size = list(
        map(int, [origin_court_points[2][0], origin_court_points[3][1] - 40])
    )
    print(left_crop_size)
    print(right_crop_size)
    points_config = {
        "polygon": polygon,
        "left_crop_size": left_crop_size,
        "right_crop_size": right_crop_size,
        "left_most_setting": left_most_setting,
        "middle_point_setting": middle_point_setting,
        "right_most_setting": right_most_setting,
    }
    return points_config
