#include <opencv2/core.hpp>
#include <string>
#include <fstream>
#include <stdexcept>
#include <filesystem>

class ImageTransformer {
public:
    cv::Mat transform(const cv::Mat& left_img, const cv::Mat& right_img, int x, int y = 0, int fov = 0);

private:
    std::string get_path(int x, const std::string& mat_name);

    cv::Mat get_local_mat(int x, const std::string& mat_name);
    
    cv::Mat load_mat_from_file(const std::string& path);
};