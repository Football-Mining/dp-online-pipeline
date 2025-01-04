import cv2
import numpy as np


def undistort_image(image):
    """
    去除图像的桶形畸变。

    参数:
    image: 输入的图像。
    k: 畸变系数，正值表示桶形畸变，负值表示枕形畸变。

    返回:
    去畸变后的图像。
    """
    k = -0.07
    h, w = image.shape[:2]
    K = np.eye(3)
    fx = max(h, w)  # 焦距
    fy = min(h, w)  # 焦距
    cx = w / 2.0  # 主点 x 坐标
    cy = h / 2.0  # 主点 y 坐标
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.array([[k], [k], [k], [k]])

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.15
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )

    undistorted_image = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return undistorted_image


def on_trackbar(val):
    global img, k
    k = val / -500.0  # 将滑块值转换为畸变系数
    undistorted_img = undistort_image(img, k)
    cv2.imshow("Undistorted Image", undistorted_img)


# 读取图像

img_name = "MOV_0079-0003"

img = cv2.imread(f"{img_name}.png")

undistort_image = undistort_image(img)

cv2.imwrite(f"{img_name}_undistorted.png", undistort_image)


# 创建窗口并添加滑块
# cv2.namedWindow('Undistorted Image')
# cv2.createTrackbar('Distortion', 'Undistorted Image', 0, 200, on_trackbar)

# # 显示初始图像
# cv2.imshow('Undistorted Image', img)

# 等待用户操作
# cv2.waitKey(0)
# cv2.destroyAllWindows()
