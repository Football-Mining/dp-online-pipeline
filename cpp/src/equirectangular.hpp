#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <filesystem>

using namespace std;
using namespace cv;

static constexpr int MATRIX_SAMPLE_RATE = 1;
static constexpr double X_HORIZON = 210;
static constexpr double Y_HORIZON = 70;

class EquirectangularModel {
public:

    EquirectangularModel(Size size = Size());

    void get_funcs();

    void get_matrix(double THETA, double PHI, bool save = false);

    void set_funcs_with_init_settings(const vector<int>& left_most_setting, const vector<int>& middle_point_setting, const vector<int>& right_most_setting);

    void get_xyz(double FOV);

    void get_size(const Mat& image);

    double get_max_fov(int y, int min_y, int max_y, int y_length);

    void get_cood_from_x(int x);

    Mat get_mat(int x, int y = -1, double fov = -1);

    pair<double, double> get_angles_from_points(int x, int y);

    void save_all_matrices(const vector<int>& x_range);

    Mat GetPerspectiveFromCoord(const Mat& image, int x, int y = -1, double fov = -1, bool fast_mode = false);

    Mat GetPerspectiveFromCoordStupid(const Mat& image, int x, int y = -1, double fov = -1);

    Mat GetPerspective(const Mat& image);

    Mat GetPerspective_cuda(const Mat& image);

private:
    int _height, _width;
    int target_height = 1080, target_width = 1920;
    Mat xyz;
    Mat xy;
    function<int(int)> y_func;
    function<double(int)> fov_func;
    int x = -1, y = -1;

    double cx = (target_width - 1) / 2.0;
    double cy = (target_height - 1) / 2.0;
};