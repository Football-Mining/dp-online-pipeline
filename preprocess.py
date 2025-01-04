import cv2
import numpy as np


def undistort_image(image):
    if isinstance(image, list):
        results = []
        for img in image:
            results.append(undistort_image(img))
        return results
    else:
        return _undistort_image(image)


def _undistort_image(image):
    k = -0.07  # 用trackbar获得的参数
    h, w = image.shape[:2]
    K = np.eye(3)
    fx = max(h, w)
    fy = min(h, w)
    cx = w / 2.0
    cy = h / 2.0
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


# undistort_image = undistort_image(img)


# 创建窗口并添加滑块
# cv2.namedWindow('Undistorted Image')
# cv2.createTrackbar('Distortion', 'Undistorted Image', 0, 200, on_trackbar)

# # 显示初始图像
# cv2.imshow('Undistorted Image', img)

# 等待用户操作
# cv2.waitKey(0)
# cv2.destroyAllWindows()
