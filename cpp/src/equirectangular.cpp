#include "equirectangular.hpp"

using namespace std;
using namespace cv;

Mat xyz2lonlat(const Mat& xyz) {
    auto t0 = chrono::high_resolution_clock::now();

    Mat norm;
    cv::norm(xyz, norm, NORM_L2, -1, true);
    auto t1 = chrono::high_resolution_clock::now();

    Mat xyz_norm = xyz / norm;
    auto t2 = chrono::high_resolution_clock::now();

    Mat x = xyz_norm.colRange(0, 1);
    Mat y = xyz_norm.colRange(1, 2);
    Mat z = xyz_norm.colRange(2, 3);
    auto t3 = chrono::high_resolution_clock::now();

    Mat lon, lat;
    cv::phase(x, z, lon);
    cv::asin(y, lat);
    auto t4 = chrono::high_resolution_clock::now();

    vector<Mat> lst = { lon, lat };
    Mat out;
    cv::hconcat(lst, out);
    auto t5 = chrono::high_resolution_clock::now();

    return out;
}

Mat lonlat2XY(const Mat& lonlat, const Size& shape) {
    Mat X = (lonlat.colRange(0, 1) / (x_horizon / 360.0 * CV_PI) + 0.5) * (shape.width - 1);
    Mat Y = (lonlat.colRange(1, 2) / (y_horizon / 360.0 * CV_PI) + 0.5) * (shape.height - 1);

    vector<Mat> lst = { X, Y };
    Mat out;
    cv::hconcat(lst, out);

    return out;
}

class EquirectangularModel {
public:
    EquirectangularModel(Size size = Size()) {
        _height = _width = -1;
        if (size != Size()) {
            _height = size.height;
            _width = size.width;
            get_funcs();
        }
    }

    void get_funcs() {
        if (y_func == nullptr) {
            y_func = [this](int x) {
                return static_cast<int>(((-5.625 * pow(x / static_cast<double>(_width) - 0.5, 2) + 0.47) * _height));
            };
            fov_func = [this](int x) {
                return 48 - 19 * abs(x - _width / 2) / (_width / 6);
            };
        }
    }

    Mat GetPerspective(const Mat& image) {
        get_size(image);
        Mat persp;
        cv::remap(image, persp, XY.col(0), XY.col(1), INTER_CUBIC);
        return persp;
    }

    Mat GetPerspective_cuda(const Mat& image) {
        get_size(image);

        cuda::GpuMat d_src(image);
        cuda::GpuMat d_map1(XY.col(0));
        cuda::GpuMat d_map2(XY.col(1));

        cuda::GpuMat dst;
        cuda::remap(d_src, dst, d_map1, d_map2, INTER_CUBIC);

        Mat persp;
        dst.download(persp);
        return persp;
    }

    void get_matrix(double THETA, double PHI, bool save = false) {
        auto t0 = chrono::high_resolution_clock::now();

        Mat y_axis = (Mat_<float>(3, 1) << 0.0, 1.0, 0.0);
        Mat x_axis = (Mat_<float>(3, 1) << 1.0, 0.0, 0.0);

        Mat R1, R2;
        Rodrigues(y_axis * radians(THETA), R1);
        Rodrigues(R1 * x_axis * radians(PHI), R2);
        Mat R = R2 * R1;
        Mat xyz_rotated = xyz * R.t();

        double rotation_angle = -THETA / 45.0 * 20;
        Mat rotate = (Mat_<float>(3, 3) << cos(radians(rotation_angle)), -sin(radians(rotation_angle)), 0,
            sin(radians(rotation_angle)), cos(radians(rotation_angle)), 0,
            0, 0, 1);
        xyz_rotated = xyz_rotated * rotate;

        Mat lonlat = xyz2lonlat(xyz_rotated);
        XY = lonlat2XY(lonlat, Size(_width, _height)).clone();

        if (save) {
            if (!filesystem::exists("matrices")) {
                filesystem::create_directory("matrices");
            }
            string filename = "matrices/mat_" + to_string(x) + ".npy";
            cv::FileStorage fs(filename, cv::FileStorage::WRITE);
            fs << "XY" << XY;
            fs.release();
        }
    }

    void set_funcs_with_init_settings(const vector<int>& left_most_setting, const vector<int>& middle_point_setting, const vector<int>& right_most_setting) {
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

    void get_xyz(double FOV) {
        int height = 1080, width = 1920;
        double f = 0.5 * width * 1 / tan(0.5 * FOV / 180.0 * CV_PI);
        double cx = (width - 1) / 2.0;
        double cy = (height - 1) / 2.0;
        Mat K = (Mat_<float>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
        Mat K_inv = K.inv();

        Mat x, y;
        meshgrid(Range(0, width), Range(0, height), x, y);
        Mat z = Mat::ones(x.size(), CV_32F);
        vector<Mat> xyz_vec = { x, y, z };
        Mat xyz;
        cv::merge(xyz_vec, xyz);
        xyz = xyz.reshape(1, xyz.total() / 3);
        xyz = xyz * K_inv.t();
        xyz = xyz.reshape(3, height);
        this->xyz = xyz;
    }

    void get_size(const Mat& image) {
        if (_height == -1) {
            _height = image.rows;
            _width = image.cols;
            get_funcs();
        }
    }

    double get_max_fov(int y, int min_y, int max_y, int y_length) {
        return min(abs(y - min_y) * 180.0 / y_length, abs(max_y - y) * 180.0 / y_length) - 5;
    }

    void get_cood_from_x(int x) {
        cout << x << " " << y_func(x) << " " << fov_func(x) << endl;
    }

    Mat GetPerspectiveFromCoordStupid(const Mat& image, int x, int y = -1, double fov = -1) {
        get_size(image);
        if (y == -1) {
            y = y_func(x);
        }
        double THETA = ((x - _width / 2.0) / (_width / 2.0)) * 90;
        double PHI = -((y - _height / 2.0) / (_height / 2.0)) * 45;

        this->x = x;
        this->y = y;
        if (fov == -1) {
            fov = fov_func(x);
        }
        get_xyz(fov);
        get_matrix(THETA, PHI, false);

        return GetPerspective(image);
    }

    Mat get_mat(int x, int y = -1, double fov = -1) {
        if (y == -1) {
            y = y_func(x);
        }
        double THETA = ((x - _width / 2.0) / (_width / 2.0)) * 90;
        double PHI = -((y - _height / 2.0) / (_height / 2.0)) * 45;
        if (fov == -1) {
            fov = fov_func(x);
        }
        get_xyz(fov);
        get_matrix(THETA, PHI);
        return XY;
    }

    Mat GetPerspectiveFromCoord(const Mat& image, int x, int y = -1, double fov = -1, bool fast_mode = false) {
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
                cv::FileStorage fs(filename, cv::FileStorage::READ);
                fs["XY"] >> XY;
                fs.release();
                auto t2 = chrono::high_resolution_clock::now();
            }
            else {
                double THETA = ((x - _width / 2.0) / (_width / 2.0)) * 90;
                double PHI = -((y - _height / 2.0) / (_height / 2.0)) * 45;
                if (fov == -1) {
                    fov = fov_func(x);
                }
                get_xyz(fov);
                auto t3 = chrono::high_resolution_clock::now();
                get_matrix(THETA, PHI);
                auto t4 = chrono::high_resolution_clock::now();
            }
        }
        auto t5 = chrono::high_resolution_clock::now();
        Mat result = GetPerspective(image);
        auto t6 = chrono::high_resolution_clock::now();
        return result;
    }

    pair<double, double> get_angles_from_points(int x, int y) {
        double THETA = ((x - _width / 2.0) / (_width / 2.0)) * 90;
        double PHI = -((y - _height / 2.0) / (_height / 2.0)) * 45;
        return make_pair(THETA, PHI);
    }

    void save_all_matrices(const vector<int>& x_range) {
        int tmp_x = x, tmp_y = y;
        for (int x : x_range) {
            int y = y_func(x);
            get_xyz(fov_func(x));
            auto [THETA, PHI] = get_angles_from_points(x, y);
            this->x = x;
            this->y = y;
            get_matrix(THETA, PHI, true);
        }
        this->x = tmp_x;
        this->y = tmp_y;
    }

private:
    int _height, _width;
    Mat xyz;
    Mat XY;
    function<int(int)> y_func;
    function<double(int)> fov_func;
    int x = -1, y = -1;

    double radians(double degrees) {
        return degrees * CV_PI / 180.0;
    }
};

int main() {
    // 打开图片
    string panorama_path = "path_to_your_image.jpg";
    Mat img = imread(panorama_path, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Error: Could not open or find the image!" << endl;
        return -1;
    }

    // 创建实例
    EquirectangularModel e(Size(img.cols, img.rows));

    // 设置初始位置
    int x_value = 1250;
    int y_value = 500;
    double fov = 34;

    // 测量 get_mat 函数的推理时间
    auto start = high_resolution_clock::now();
    Mat result = e.get_mat(x_value, y_value, fov);
    auto end = high_resolution_clock::now();

    // 计算并打印推理时间
    auto duration = duration_cast<milliseconds>(end - start).count();
    cout << "Inference time: " << duration << " ms" << endl;

    // 显示初始图片
    Mat persp = e.GetPerspectiveFromCoord(img, x_value, y_value, fov);
    string window_name = "Perspective View";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    imshow(window_name, persp);

    return 0;
}