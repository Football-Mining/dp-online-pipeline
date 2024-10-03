#include "equirectangular.hpp"

using namespace std;
using namespace cv;

double radians(double degrees) {
    return degrees * CV_PI / 180.0;
}

MatrixXf xyz2lonlat(const MatrixXf& xyz) {
    // Calculate the norm
    MatrixXf squared = xyz.array().square();
    VectorXf norm = squared.rowwise().sum().array().sqrt();

    // Normalize xyz
    MatrixXf xyz_norm = xyz.array().colwise() / norm.array();

    // Calculate longitude and latitude
    VectorXf lon = xyz_norm.col(0).binaryExpr(xyz_norm.col(2), [](float x, float z) { return std::atan2(x, z); });
    VectorXf lat = xyz_norm.col(1).unaryExpr([](float y) { return std::asin(y); });

    // Concatenate lon and lat
    MatrixXf out(xyz.rows(), 2);
    out.col(0) = lon;
    out.col(1) = lat;

    return out;
}

MatrixXd lonlat2XY(const MatrixXd& lonlat, const Vector2i& shape) {
    const double X_HORIZON = 360.0; // Define X_HORIZON appropriately
    const double Y_HORIZON = 180.0; // Define Y_HORIZON appropriately

    MatrixXd X = (lonlat.col(0) / (X_HORIZON / 360.0 * M_PI) + 0.5) * (shape.x() - 1);
    MatrixXd Y = (lonlat.col(1) / (Y_HORIZON / 360.0 * M_PI) + 0.5) * (shape.y() - 1);

    MatrixXd out(lonlat.rows(), 2);
    out << X, Y;

    return out;
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

    Matrix3d R1, R2;
    AngleAxisd rot1(radians(theta), y_axis);
    R1 = rot1.toRotationMatrix();
    AngleAxisd rot2(radians(phi), R1 * x_axis);
    R2 = rot2.toRotationMatrix();
    Matrix3d R = R2 * R1;

    MatrixXd xyz_rotated = (xyz.reshape(1, target_height * target_width) * R.transpose()).eval();

    double rotation_angle = -theta / 45.0 * 20;
    Matrix3d rotate;
    rotate << cos(radians(rotation_angle)), -sin(radians(rotation_angle)), 0,
              sin(radians(rotation_angle)),  cos(radians(rotation_angle)), 0,
              0, 0, 1;
    xyz_rotated = (xyz_rotated * rotate).eval();

    MatrixXd lonlat = xyz2lonlat(xyz_rotated.reshape(3, target_height));
    xy = lonlat2XY(lonlat, Size(_width, _height)).clone();

    if (save) {
        if (!filesystem::exists("matrices")) {
            filesystem::create_directory("matrices");
        }
        string filename = "matrices/mat_" + to_string(x) + ".npy";
        ofstream fs(filename, ios::binary);
        if (fs.is_open()) {
            fs.write(reinterpret_cast<const char*>(xy.data()), xy.size() * sizeof(double));
            fs.close();
        }
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
    float f = 0.5 * target_width * 1 / tan(0.5 * FOV / 180.0 * M_PI);
    float cx = (target_width - 1) / 2.0;
    float cy = (target_height - 1) / 2.0;

    Matrix3f K;
    K << f, 0, cx,
            0, f, cy,
            0, 0, 1;

    MatrixXf x = VectorXf::LinSpaced(width, 0, width - 1).replicate(1, height).transpose();
    MatrixXf y = VectorXf::LinSpaced(height, 0, height - 1).replicate(1, width);
    MatrixXf z = MatrixXf::Ones(height, width);

    MatrixXf xyz(target_height * target_width, 3);
    xyz << Map<MatrixXf>(x.data(), target_height * target_width, 1),
            Map<MatrixXf>(y.data(), target_height * target_width, 1),
            Map<MatrixXf>(z.data(), target_height * target_width, 1);

    Matrix3f K_inv = K.inverse();
    this->xyz = (xyz * K_inv.transpose()).reshaped(target_height, target_width, 3);
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