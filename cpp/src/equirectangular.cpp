#include "equirectangular.hpp"

using namespace std;
using namespace cv;

double radians(double degrees) {
    return degrees * CV_PI / 180.0;
}

void meshgrid(const std::vector<int>& x_range, const std::vector<int>& y_range, std::vector<std::vector<int>>& x, std::vector<std::vector<int>>& y) {
    int width = x_range.size();
    int height = y_range.size();
    
    x.resize(height, std::vector<int>(width));
    y.resize(height, std::vector<int>(width));
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            x[i][j] = x_range[j];
            y[i][j] = y_range[i];
        }
    }
}

cv::Mat xyz2lonlat(const cv::Mat& xyz) {
    // Calculate the norm
    cv::Mat squared;
    cv::reduce(xyz.mul(xyz), squared, 2, cv::REDUCE_SUM);
    cv::Mat norm;
    cv::sqrt(squared, norm);
    norm = norm.reshape(1, xyz.rows);

    // Normalize xyz
    cv::Mat xyz_norm = xyz / norm;

    // Calculate longitude and latitude
    cv::Mat lon, lat;
    cv::phase(xyz_norm.col(0), xyz_norm.col(2), lon, true);
    
    lat = xyz_norm.col(1).clone();
    for (int i = 0; i < lat.rows; ++i) {
        lat.at<float>(i) = std::asin(lat.at<float>(i));
    }

    // Concatenate lon and lat
    std::vector<cv::Mat> lst = {lon, lat};
    cv::Mat out;
    cv::hconcat(lst, out);
    out.convertTo(out, CV_32F);

    return out;
}
Mat lonlat2XY(const Mat& lonlat, const Size& shape) {
    Mat X = (lonlat.colRange(0, 1) / (X_HORIZON / 360.0 * CV_PI) + 0.5) * (shape.width - 1);
    Mat Y = (lonlat.colRange(1, 2) / (Y_HORIZON / 360.0 * CV_PI) + 0.5) * (shape.height - 1);

    vector<Mat> lst = { X, Y };
    Mat out;
    hconcat(lst, out);

    return out;
}

cv::Mat concatenateXYZ(const cv::Mat& x, const cv::Mat& y, const cv::Mat& z) {
    CV_Assert(x.size() == y.size() && y.size() == z.size());
    CV_Assert(x.type() == y.type() && y.type() == z.type());

    int rows = x.rows;
    int cols = x.cols;
    int channels = x.channels();

    cv::Mat xyz(rows, cols, CV_MAKE_TYPE(x.depth(), 3 * channels));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int c = 0; c < channels; ++c) {
                xyz.at<cv::Vec3f>(i, j)[0] = x.at<float>(i, j);
                xyz.at<cv::Vec3f>(i, j)[1] = y.at<float>(i, j);
                xyz.at<cv::Vec3f>(i, j)[2] = z.at<float>(i, j);
            }
        }
    }

    return xyz;
}

EquirectangularModel::EquirectangularModel(Size size) {
    _height = _width = -1;
    if (size != Size()) {
        _height = size.height;
        _width = size.width;
        get_funcs();
    }
}

void EquirectangularModel::get_funcs() {
    if (y_func == nullptr) {
        y_func = [this](int x) {
            return static_cast<int>(((-5.625 * pow(x / static_cast<double>(_width) - 0.5, 2) + 0.47) * _height));
        };
        fov_func = [this](int x) {
            return 48 - 19 * abs(x - _width / 2) / (_width / 6);
        };
    }
}

void EquirectangularModel::get_matrix(double theta, double phi, bool save) {
    auto t0 = chrono::high_resolution_clock::now();

    Mat y_axis = (Mat_<double>(3, 1) << 0.0, 1.0, 0.0);
    Mat x_axis = (Mat_<double>(3, 1) << 1.0, 0.0, 0.0);

    Mat R1, R2;
    Rodrigues(y_axis * radians(theta), R1);
    Rodrigues(R1 * x_axis * radians(phi), R2);
    Mat R = R2 * R1;
    Mat xyz_rotated = xyz.reshape(1, target_height*target_width) * R.t();

    double rotation_angle = -theta / 45.0 * 20;
    Mat rotate = (Mat_<double>(3, 3) << cos(radians(rotation_angle)), -sin(radians(rotation_angle)), 0,
        sin(radians(rotation_angle)), cos(radians(rotation_angle)), 0,
        0, 0, 1);
    xyz_rotated = xyz_rotated * rotate;

    Mat lonlat = xyz2lonlat(xyz_rotated.reshape(3, target_height));
    xy = lonlat2XY(lonlat, Size(_width, _height)).clone();

    if (save) {
        if (!filesystem::exists("matrices")) {
            filesystem::create_directory("matrices");
        }
        string filename = "matrices/mat_" + to_string(x) + ".npy";
        FileStorage fs(filename, FileStorage::WRITE);
        fs << "xy" << xy;
        fs.release();
    }
}

void EquirectangularModel::set_funcs_with_init_settings(const vector<int>& left_most_setting, const vector<int>& middle_point_setting, const vector<int>& right_most_setting) {
    y_func = [left_most_setting, middle_point_setting, right_most_setting](int x) {
        if (x <= middle_point_setting[0]) {
            return left_most_setting[1] + (middle_point_setting[1] - left_most_setting[1]) * (x - left_most_setting[0]) / (middle_point_setting[0] - left_most_setting[0]);
        }
        else {
            return middle_point_setting[1] + (right_most_setting[1] - middle_point_setting[1]) * (1 - (right_most_setting[0] - x) / (right_most_setting[0] - middle_point_setting[0]));
        }
    };

    fov_func = [left_most_setting, middle_point_setting, right_most_setting](int x) {
        if (x <= middle_point_setting[0]) {
            return left_most_setting[2] + (middle_point_setting[2] - left_most_setting[2]) * (x - left_most_setting[0]) / (middle_point_setting[0] - left_most_setting[0]);
        }
        else {
            return middle_point_setting[2] + (right_most_setting[2] - middle_point_setting[2]) * (1 - (right_most_setting[0] - x) / (right_most_setting[0] - middle_point_setting[0]));
        }
    };
}

void EquirectangularModel::get_xyz(double fov) {
    double f = 0.5 * target_width / tan(0.5 * fov * M_PI / 180.0);
    double cx = (target_width - 1) / 2.0;
    double cy = (target_height - 1) / 2.0;

    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        f, 0, cx,
        0, f, cy,
        0, 0, 1);

    cv::Mat K_inv = K.inv(cv::DECOMP_CHOLESKY);

    int out[3] = {target_height, target_width, 3};
    cv::Mat tmp(3, out, CV_64F, cv::Scalar(0));
    for (int i = 0; i < target_height; ++i) {
        for (int j = 0; j < target_width; ++j) {
            tmp.at<double>(i, j, 0) = j;
            tmp.at<double>(i, j, 1) = i;
            tmp.at<double>(i, j, 2) = 1;
        }
    }

    xyz = tmp.reshape(1, target_height*target_width) * K_inv.t();
    xyz = xyz.reshape(3, target_height);
}

void EquirectangularModel::get_size(const Mat& image) {
    if (_height == -1) {
        _height = image.rows;
        _width = image.cols;
        get_funcs();
    }
}

double EquirectangularModel::get_max_fov(int y, int min_y, int max_y, int y_length) {
    return min(abs(y - min_y) * 180.0 / y_length, abs(max_y - y) * 180.0 / y_length) - 5;
}

void EquirectangularModel::get_cood_from_x(int x) {
    cout << x << " " << y_func(x) << " " << fov_func(x) << endl;
}

Mat EquirectangularModel::get_mat(int x, int y, double fov) {
    if (y == -1) {
        y = y_func(x);
    }
    double theta = ((x - _width / 2.0) / (_width / 2.0)) * 90;
    double phi = -((y - _height / 2.0) / (_height / 2.0)) * 45;
    if (fov == -1) {
        fov = fov_func(x);
    }
    get_xyz(fov);
    get_matrix(theta, phi);
    return xy;
}

pair<double, double> EquirectangularModel::get_angles_from_points(int x, int y) {
    double theta = ((x - _width / 2.0) / (_width / 2.0)) * 90;
    double phi = -((y - _height / 2.0) / (_height / 2.0)) * 45;
    return make_pair(theta, phi);
}

void EquirectangularModel::save_all_matrices(const vector<int>& x_range) {
    int tmp_x = x, tmp_y = y;
    for (int x : x_range) {
        int y = y_func(x);
        get_xyz(fov_func(x));
        auto [theta, phi] = get_angles_from_points(x, y);
        this->x = x;
        this->y = y;
        get_matrix(theta, phi, true);
    }
    this->x = tmp_x;
    this->y = tmp_y;
}

Mat EquirectangularModel::GetPerspectiveFromCoordStupid(const Mat& image, int x, int y, double fov) {
    get_size(image);
    if (y == -1) {
        y = y_func(x);
    }
    double theta = ((x - _width / 2.0) / (_width / 2.0)) * 90;
    double phi = -((y - _height / 2.0) / (_height / 2.0)) * 45;

    this->x = x;
    this->y = y;
    if (fov == -1) {
        fov = fov_func(x);
    }
    get_xyz(fov);
    get_matrix(theta, phi, false);

    return GetPerspective(image);
}

Mat EquirectangularModel::GetPerspective(const Mat& image) {
    get_size(image);
    Mat persp;
    remap(image, persp, xy.col(0), xy.col(1), INTER_CUBIC);
    return persp;
}

Mat EquirectangularModel::GetPerspective_cuda(const Mat& image) {
    get_size(image);

    cuda::GpuMat d_src(image);
    cuda::GpuMat d_map1(xy.col(0));
    cuda::GpuMat d_map2(xy.col(1));

    cuda::GpuMat dst;
    cuda::remap(d_src, dst, d_map1, d_map2, INTER_CUBIC);

    Mat persp;
    dst.download(persp);
    return persp;
}

Mat EquirectangularModel::GetPerspectiveFromCoord(const Mat& image, int x, int y, double fov, bool fast_mode) {
    auto t0 = chrono::high_resolution_clock::now();
    get_size(image);
    auto t1 = chrono::high_resolution_clock::now();

    if (y == -1) {
        y = y_func(x);
    }

    if (this->x == -1 || this->y == -1 || this->x != x || this->y != y) {
        this->x = x;
        this->y = y;
        if (fast_mode && filesystem::exists("matrices/mat_" + to_string(x / MATRIX_SAMPLE_RATE * MATRIX_SAMPLE_RATE) + ".npy")) {
            string filename = "matrices/mat_" + to_string(x / MATRIX_SAMPLE_RATE * MATRIX_SAMPLE_RATE) + ".npy";
            FileStorage fs(filename, FileStorage::READ);
            fs["xy"] >> xy;
            fs.release();
            auto t2 = chrono::high_resolution_clock::now();
        }
        else {
            double theta = ((x - _width / 2.0) / (_width / 2.0)) * 90;
            double phi = -((y - _height / 2.0) / (_height / 2.0)) * 45;
            if (fov == -1) {
                fov = fov_func(x);
            }
            get_xyz(fov);
            auto t3 = chrono::high_resolution_clock::now();
            get_matrix(theta, phi);
            auto t4 = chrono::high_resolution_clock::now();
        }
    }
    auto t5 = chrono::high_resolution_clock::now();
    Mat result = GetPerspective(image);
    auto t6 = chrono::high_resolution_clock::now();
    return result;
}

int main() {
// 打开图片
// string panorama_path = "path_to_your_image.jpg";
// Mat img = imread(panorama_path, IMREAD_COLOR);
// if (img.empty()) {
//     cerr << "Error: Could not open or find the image!" << endl;
//     return -1;
// }

// 创建实例
// EquirectangularModel e(Size(img.cols, img.rows));
EquirectangularModel e(Size(1274, 2930));

// 设置初始位置
int x_value = 1250;
int y_value = 500;
double fov = 34;

// 测量 get_mat 函数的推理时间
auto start = std::chrono::high_resolution_clock::now();
Mat result = e.get_mat(x_value, y_value, fov);
auto end = std::chrono::high_resolution_clock::now();

cout << result << endl;

// 计算并打印推理时间
auto duration = std::chrono::duration_cast<chrono::milliseconds>(end - start).count();
cout << "Inference time: " << duration << " ms" << endl;

// 显示初始图片
// Mat persp = e.GetPerspectiveFromCoord(img, x_value, y_value, fov);
// string window_name = "Perspective View";
// namedWindow(window_name, WINDOW_AUTOSIZE);
// imshow(window_name, persp);

return 0;
}