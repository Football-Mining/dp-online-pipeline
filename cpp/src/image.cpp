#include "image.hpp"

Images::Images(double medium_megapix, double low_megapix, double final_megapix) {
    if (medium_megapix < low_megapix) {
        throw ImageError("Medium resolution megapix need to be greater or equal than low resolution megapix");
    }

    scalers_[Resolution::MEDIUM] = MegapixDownscaler(medium_megapix);
    scalers_[Resolution::LOW] = MegapixDownscaler(low_megapix);
    scalers_[Resolution::FINAL] = MegapixDownscaler(final_megapix);
    scales_set_ = false;
    sizes_set_ = false;
    names_set_ = false;
}

std::unique_ptr<Images> Images::of(const std::vector<cv::Mat>& images, double medium_megapix, double low_megapix, double final_megapix) {
    if (images.empty()) {
        throw ImageError("images must not be an empty list");
    }
    if (check_list_element_types(images)) {
        return std::make_unique<NumpyImages>(images, medium_megapix, low_megapix, final_megapix);
    } else {
        throw ImageError("invalid images list: must be numpy arrays (loaded images)");
    }
}

std::unique_ptr<Images> Images::of(const std::vector<std::string>& images, double medium_megapix, double low_megapix, double final_megapix) {
    if (images.empty()) {
        throw ImageError("images must not be an empty list");
    }
    if (check_list_element_types(images)) {
        return std::make_unique<FilenameImages>(images, medium_megapix, low_megapix, final_megapix);
    } else {
        throw ImageError("invalid images list: must be filename strings");
    }
}

void Images::set_scales(const cv::Size& size) {
    if (!scales_set_) {
        for (auto& scaler : scalers_) {
            scaler.second.set_scale_by_img_size(size);
        }
        scales_set_ = true;
    }
}

MegapixDownscaler& Images::get_scaler(Resolution resolution) {
    check_resolution(resolution);
    return scalers_[resolution];
}

std::vector<cv::Size> Images::get_scaled_img_sizes(Resolution resolution) {
    assert(scales_set_ && sizes_set_);
    check_resolution(resolution);
    std::vector<cv::Size> scaled_sizes;
    for (const auto& size : sizes_) {
        scaled_sizes.push_back(get_scaler(resolution).get_scaled_img_size(size));
    }
    return scaled_sizes;
}

double Images::get_ratio(Resolution from_resolution, Resolution to_resolution) {
    assert(scales_set_);
    check_resolution(from_resolution);
    check_resolution(to_resolution);
    return get_scaler(to_resolution).scale / get_scaler(from_resolution).scale;
}

cv::Mat Images::read_image(const std::string& img_name) {
    cv::Mat img = cv::imread(img_name);
    if (img.empty()) {
        throw ImageError("Cannot read image " + img_name);
    }
    return img;
}

cv::Size Images::get_image_size(const cv::Mat& img) {
    return cv::Size(img.cols, img.rows);
}

cv::Mat Images::resize_img_by_scaler(const MegapixDownscaler& scaler, const cv::Size& size, const cv::Mat& img) {
    cv::Size desired_size = scaler.get_scaled_img_size(size);
    cv::Mat resized_img;
    cv::resize(img, resized_img, desired_size, 0, 0, cv::INTER_LINEAR_EXACT);
    return resized_img;
}

void Images::check_resolution(Resolution resolution) {
    if (resolution != Resolution::MEDIUM && resolution != Resolution::LOW && resolution != Resolution::FINAL) {
        throw ImageError("Invalid resolution");
    }
}

std::vector<std::string> Images::resolve_wildcards(const std::vector<std::string>& img_names) {
    std::vector<std::string> resolved_names;
    for (const auto& name : img_names) {
        for (const auto& entry : fs::directory_iterator(name)) {
            if (!fs::is_directory(entry)) {
                resolved_names.push_back(entry.path().string());
            }
        }
    }
    return resolved_names;
}

bool Images::check_list_element_types(const std::vector<cv::Mat>& list) {
    return std::all_of(list.begin(), list.end(), [](const cv::Mat& element) { return !element.empty(); });
}

bool Images::check_list_element_types(const std::vector<std::string>& list) {
    return std::all_of(list.begin(), list.end(), [](const std::string& element) { return !element.empty(); });
}

cv::Mat Images::to_binary(const cv::Mat& img) {
    cv::Mat gray_img, binary_img;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    } else {
        gray_img = img;
    }
    cv::threshold(gray_img, binary_img, 0.5, 255.0, cv::THRESH_BINARY);
    return binary_img;
}