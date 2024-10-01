#include "paranoma.hpp"

cv::Mat ImageTransformer::get_path(int x, const std::string& mat_name) {
    std::string subfolder = std::to_string(x / 100);
    std::filesystem::path dir_path = std::filesystem::path("matrices") / mat_name / subfolder;
    if (!std::filesystem::exists(dir_path)) {
        std::filesystem::create_directories(dir_path);
    }
    return (dir_path / (std::to_string(x) + ".npy")).string();
}

cv::Mat ImageTransformer::get_local_mat(int x, const std::string& mat_name) {
    x = static_cast<int>(x);
    std::string path = get_path(x, mat_name);
    return load_mat_from_file(path);
}

cv::Mat ImageTransformer::load_mat_from_file(const std::string& path) {{
    cv::Mat mat;
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
    file.read(reinterpret_cast<char*>(mat.data), mat.total() * mat.elemSize());
    file.close();
    return mat;
}

cv::Mat ImageTransformer::transform(const cv::Mat& left_img, const cv::Mat& right_img, int x, int y = 0, int fov = 0) {
    cv::Mat left_mask = get_local_mat(x, "left_mask");
    cv::Mat right_mask = get_local_mat(x, "right_mask");
    cv::detail::MultiBandBlender blender;
    blender.setNumBands(5);
    blender.prepare(cv::Rect(0, 0, 1920, 1080));

    blender.feed(left_img.getUMat(cv::ACCESS_READ), left_mask.getUMat(cv::ACCESS_READ), cv::Point(0, 0));
    blender.feed(right_img.getUMat(cv::ACCESS_READ), right_mask.getUMat(cv::ACCESS_READ), cv::Point(0, 0));
    cv::UMat result, result_mask;
    blender.blend(result, result_mask);
    return result.getMat(cv::ACCESS_READ);
}

cv::Mat read_image_from_stdin() {
    std::vector<uchar> img_bytes;
    std::string line;
    std::getline(std::cin, line);  // Read metadata line
    int x, y, fov;
    std::stringstream ss(line);
    ss >> x >> y >> fov;

    // Read image bytes
    std::istreambuf_iterator<char> begin(std::cin), end;
    img_bytes.assign(begin, end);

    // Decode image bytes to cv::Mat
    cv::Mat img = cv::imdecode(img_bytes, cv::IMREAD_COLOR);
    return img;
}

void write_image_to_stdout(const cv::Mat& img) {
    std::vector<uchar> img_bytes;
    cv::imencode(".jpg", img, img_bytes);
    std::cout.write(reinterpret_cast<const char*>(img_bytes.data()), img_bytes.size());
    std::cout.flush();
}

int main() {
    ImageTransformer transformer;
    cv::Mat left_img, right_img;
    int x, y, fov;
    // while (true) {
    //     cv::Mat left_img = read_image_from_stdin();
    //     cv::Mat right_img = read_image_from_stdin();
    //     std::cin >> x >> y >> fov;
    //     cv::Mat result = transformer.transform(left_img, right_img, x, y, fov);
    //     write_image_to_stdout(result);
    // }
    cv::Mat left_img = read_image_from_stdin();
    cv::Mat right_img = read_image_from_stdin();
    std::cin >> x; // >> y >> fov;
    cv::Mat result = transformer.transform(left_img, right_img, x, y, fov);
    write_image_to_stdout(result);
}