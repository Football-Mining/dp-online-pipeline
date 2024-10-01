import cv2
import os
from tqdm import tqdm
import time
import numpy as np
import sqlite3
import shutil
import subprocess

from equirectangular_model import EquirectangularModel
from utils import get_and_init_stitcher, get_regist_imgs
from std_sender import *


def combine_remaps(map1_xy, map2_xy, map3_xy=None):
    combined_map_xy = cv2.remap(map1_xy, map2_xy, None, cv2.INTER_NEAREST)
    if map3_xy is not None:
        combined_map_xy = cv2.remap(combined_map_xy, map3_xy, None, cv2.INTER_NEAREST)

    return combined_map_xy

def concat_imgs(img1, img2):
    assert img1.shape == img2.shape, "Images must have the same shape"

    final_image = np.zeros((img1.shape[0], img1.shape[1] * 2, img1.shape[2]), dtype=img1.dtype)

    final_image[:, :img1.shape[1], :] = img1
    final_image[:, img1.shape[1]:, :] = img2

    return final_image

def concat_maps(imgs):
    max_height = 0 # find the max width of all the images
    total_width = 0 # the total height of the images (vertical stacking)

    for image in imgs:
        # find the max width of all the images
        max_height = max(max_height,image.shape[0])
        # add the height of the current image to the total height
        total_width += image.shape[1]
    
    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((total_width,max_height,imgs[0].shape[-1]))

    current_x = 0 # keep track of where your current image was last placed in the x coordinate
    for image in imgs:
        # add an image to the final array and increment the x coordinate
        final_image[current_x:image.shape[1]+current_x,:image.shape[0],:] = image.transpose(1,0,2)
        current_x += image.shape[1]
    
    return final_image.transpose(1,0,2)


class ImageTransformer:
    """
    stitcher和EquirectangularModel的更高层封装，从原图直接转换为目标图，跳过全景图生成
    """
    def __init__(self, points_config, dp_live_config, size=(1920, 1080), warper_type=None):

        if warper_type is None:
            self.stitcher = get_and_init_stitcher(dp_live_config["device_id"])
        else:
            self.stitcher = get_and_init_stitcher(dp_live_config["device_id"], warper_type=warper_type)

        self.points_config = points_config
        self.e = EquirectangularModel()
        self.e.set_funcs_with_init_settings(points_config["left_most_setting"], points_config["middle_point_setting"], points_config["right_most_setting"])
        self.init_remap()


    def init_remap(self):
        left_img, right_img = get_regist_imgs()
        init_panorama = self.stitcher.stitch(left_img, right_img)
        self.e.GetPerspectiveFromCoordStupid(init_panorama, 1500)
        warped_imgs = list(self.stitcher.stitcher.warp_imgs([left_img, right_img], self.stitcher.cameras))
        warped_maps = list(self.stitcher.stitcher.build_warp_maps([left_img, right_img], self.stitcher.cameras))
        blend_maps = list(self.stitcher.stitcher.build_blend_maps(warped_imgs, (x for x in self.stitcher.seam_masks), self.stitcher.img_corners))

        remapped_mask = [cv2.remap(self.stitcher.seam_masks[i], blend_maps[i], None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).get() for i in range(2)]

        self.left_warped_map = warped_maps[0]
        self.right_warped_map = warped_maps[1]
        self.left_blend_map = blend_maps[0]
        self.right_blend_map = blend_maps[1]
        self.left_remapped_mask = remapped_mask[0]
        self.right_remapped_mask = remapped_mask[1]


    def compute_img(self, origin_img, x, direction):
        remap = self.get_local_mat(x, f"{direction}_remap")
        result = gpu_remap(origin_img, remap, None, cv2.INTER_CUBIC)
        return result


    def save_remap(self, x):
        if not os.path.exists("matrices/"):
            os.mkdir("matrices")
        x = int(x)
        e_mat = self.e.get_mat(x)
        t0 = time.time()
        left_remap = combine_remaps(self.left_warped_map, self.left_blend_map, e_mat)
        t1 = time.time()
        right_remap = combine_remaps(self.right_warped_map, self.right_blend_map, e_mat)
        t2 = time.time()
        left_mask = cv2.remap(self.left_remapped_mask, e_mat, None, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        t3 = time.time()
        right_mask = cv2.remap(self.right_remapped_mask, e_mat, None, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        t4 = time.time()
        # print(f"{x} left_remap: {t1-t0}, right_remap: {t2-t1}, left_mask: {t3-t2}, right_mask: {t4-t3}")
        self.save_local_mat(right_remap, x, "right_remap")
        self.save_local_mat(left_remap, x, "left_remap")
        self.save_local_mat(left_mask, x, "left_mask")
        self.save_local_mat(right_mask, x, "right_mask")
        t5 = time.time()
        # print(f"save_remap: {t5-t4}")

    def get_path(self, x, mat_name):
        subfolder = f"{x // 100}"
        dir_path = os.path.join("matrices", mat_name, subfolder)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return os.path.join(dir_path, f"{x}.npy")

    def save_local_mat(self, mat, x, mat_name):
        x = int(x)
        
        return np.save(self.get_path(x, mat_name), mat)

    def get_local_mat(self, x, mat_name):
        x = int(x)
        return np.load(self.get_path(x, mat_name))

    def transform(self, left_img, right_img, x, y=None, fov=None):
        
        cmd = [
            'paranoma',
            cv2.imencode('.jpg', left_img)[1].tobytes(),
            cv2.imencode('.jpg', right_img)[1].tobytes(),
            x,
            # y,
            # fov
        ]

        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        res = process.stdout.read()
        
        return res
    
    def precalculate(self):
        shutil.rmtree("matrices")
        os.makedirs("matrices")
        min_x = self.points_config["left_most_setting"][0]
        max_x = self.points_config["right_most_setting"][0]
        # max_x = min_x
        for x in tqdm(range(min_x, max_x+1)):
            self.save_remap(x)
        

# def save_to_db(mat, x, db_name="matrices.db"):
#     conn = sqlite3.connect(db_name)
#     cursor = conn.cursor()
#     cursor.execute('''CREATE TABLE IF NOT EXISTS Matrices (id INTEGER PRIMARY KEY, matrix BLOB)''')
#     cursor.execute('INSERT INTO Matrices (id, matrix) VALUES (?, ?)', (x, mat.tobytes()))
#     conn.commit()
#     conn.close()

# def load_from_db(x, db_name="matrices.db"):
#     conn = sqlite3.connect(db_name)
#     cursor = conn.cursor()
#     cursor.execute('SELECT matrix FROM Matrices WHERE id=?', (x,))
#     mat_blob = cursor.fetchone()[0]
#     conn.close()
#     return np.frombuffer(mat_blob, dtype=np.float32).reshape(expected_shape)  # Adjust for your matrix type
    

class ImageTransformerV1:
    """
    stitcher和EquirectangularModel的更高层封装，从原图直接转换为目标图，跳过全景图生成
    """
    def __init__(self, points_config, size=(1920, 1080)):
        self.stitcher = get_and_init_stitcher()
        self.points_config = points_config
        self.e = EquirectangularModel()
        self.e.set_funcs_with_init_settings(points_config["left_most_setting"], points_config["middle_point_setting"], points_config["right_most_setting"])
        self.size = size
        self.combined_remap = None
        self.precalculated_remaps = {}
        self.last_x = None
    
    def init_remap(self, left_img, right_img, x=1500):
        init_panorama = self.stitcher.stitch(left_img, right_img)
        self.e.GetPerspectiveFromCoordStupid(init_panorama, 1500)

        warped_imgs = list(self.stitcher.stitcher.warp_imgs([left_img, right_img], self.stitcher.cameras))
        warped_maps = list(self.stitcher.stitcher.build_warp_maps([left_img, right_img], self.stitcher.cameras))
        blend_maps = list(self.stitcher.stitcher.build_blend_maps(warped_imgs, (x for x in self.stitcher.seam_masks), self.stitcher.img_corners))

        concat_warped_maps = concat_maps(warped_maps).astype(np.float32)
        concat_warped_maps[:, warped_maps[0].shape[1]:, 0] += left_img.shape[1]

        # concat_warped_img = cv2.remap(concat_raw_imgs, concat_warped_maps, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        # cv2.imwrite("assets/concat_warped_img.png", concat_warped_img)

        remapped_mask = [cv2.remap(self.stitcher.seam_masks[i], blend_maps[i], None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).get() for i in range(2)]
        merged_blend_map = blend_maps[0]
        blend_maps[1][:, :, 0] += warped_maps[0].shape[1]

        # combined_remap = combine_remaps(concat_warped_maps, blend_maps[1])
        # right_blend_img = cv2.remap(concat_raw_imgs, combined_remap, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # cv2.imwrite("assets/right_blend_img.png", right_blend_img)

        merged_blend_map[remapped_mask[0] == 0] = 0
        merged_blend_map[remapped_mask[1] != 0] = blend_maps[1][remapped_mask[1] != 0]

        self.concat_warped_maps = concat_warped_maps
        self.merged_blend_map = merged_blend_map

        init_panorama = self.stitcher.stitch(left_img, right_img)
        persp = self.e.GetPerspectiveFromCoordStupid(init_panorama, x)
        self.x = x
        # cv2.imwrite("panaroma.png", init_panorama)


        combined_remap = combine_remaps(concat_warped_maps, merged_blend_map, self.e.XY)
        self.combined_remap = combined_remap


    def precalculate(self):
        min_x = self.points_config["left_most_setting"][0]
        max_x = self.points_config["right_most_setting"][0]
        # max_x = min_x
        for x in tqdm(range(min_x, max_x+1)):
            self.save_remap(x)



    def save_remap(self, x):
        remap = combine_remaps(self.concat_warped_maps, self.merged_blend_map, self.e.get_mat(x))
        np.save(self.get_remap_save_path(x), remap)

    def get_remap_save_path(self, x):
        return f'matrices/remap_{x}.npy'

    def transform(self, left_img, right_img, x, y=None, fov=None):
        if self.combined_remap is None:
            self.init_remap(left_img, right_img)
        x = int(x)
        if self.x != x:
            t0 = time.time()
            if os.path.exists(self.get_remap_save_path(x)):
                print("matrix found")
                self.combined_remap = np.load(self.get_remap_save_path(x))
                print(f"load time: {time.time() - t0}")
            else:
                self.combined_remap = combine_remaps(self.concat_warped_maps, self.merged_blend_map, self.e.get_mat(x, y, fov))
                print(f"compute time: {time.time() - t0}")
        self.x = x
        concated_img = concat_imgs(left_img, right_img)
        result = gpu_remap(concated_img, self.combined_remap, None)
        return result
    
    def get_max_fov(self, y):
        min_y = self.stitcher.map_point_to_parorama((self.size[0], 0))[1]
        max_y = self.stitcher.map_point_to_parorama((self.size[0], self.size[1]))[1]
        return self.e.get_max_fov(y, min_y, max_y, self.e._height)
    
def gpu_remap(image, remap_matrix_x, remap_matrix_y, interpolation=cv2.INTER_NEAREST):    
    d_src = cv2.cuda_GpuMat(image)

    if remap_matrix_y is None:
        d_map1 = cv2.cuda_GpuMat(remap_matrix_x[..., 0])
        d_map2 = cv2.cuda_GpuMat(remap_matrix_x[..., 1])
    else:
        d_map1 = cv2.cuda_GpuMat(remap_matrix_x)
        d_map2 = remap_matrix_y
    
    dst = cv2.cuda.remap(d_src, d_map1, d_map2, interpolation=interpolation)

    # 下载结果
    result = dst.download()

    return result


if __name__ == "__main__":
    import json
    dp_live_config = json.load(open("dp_live_config.json", "r"))
    match_id = dp_live_config["match_id"]
    device_id = dp_live_config["device_id"]
    points_config = json.load(open(f"{device_id}/{match_id}/points_config.json", "r"))

    img_transformer = ImageTransformer(points_config, dp_live_config)


    img_transformer.precalculate()
    # x = 910
    # left = transformer.compute_img(left_img, x, "left")
    # right = transformer.compute_img(right_img, x, "right")
    # test_result = transformer.transform(left, right, x)

    # cv2.imwrite("combined_image_from_concat_mbb.png", test_result)

    
