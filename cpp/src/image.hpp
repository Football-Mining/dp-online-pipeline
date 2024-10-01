#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include <memory>
#include <cassert>

namespace fs = std::filesystem;

class ImageError : public std::runtime_error {
public:
    explicit ImageError(const std::string& message) : std::runtime_error(message) {}
};

class MegapixDownscaler {
public:
    explicit MegapixDownscaler(double megapix) : megapix_(megapix) {}
    void set_scale_by_img_size(const cv::Size& size) {
        // Implementation for setting scale based on image size
    }
    cv::Size get_scaled_img_size(const cv::Size& size) const {
        // Implementation for getting scaled image size
        return size; // Placeholder
    }
    double scale;

private:
    double megapix_;
};

class Images {
public:
    enum class Resolution { MEDIUM = 0.6, LOW = 0.1, FINAL = -1 };

    static std::unique_ptr<Images> of(const std::vector<std::string>& images, double medium_megapix = 0.6, double low_megapix = 0.1, double final_megapix = -1);

    virtual ~Images() = default;

    virtual void subset(const std::vector<int>& indices) = 0;
    virtual std::vector<cv::Mat>::iterator begin() = 0;
    virtual std::vector<cv::Mat>::iterator end() = 0;

    std::vector<cv::Size> get_scaled_img_sizes(Resolution resolution);
    double get_ratio(Resolution from_resolution, Resolution to_resolution);

protected:
    Images(double medium_megapix, double low_megapix, double final_megapix);

    std::map<Resolution, MegapixDownscaler> scalers_;
    bool scales_set_;
    bool sizes_set_;
    bool names_set_;
    std::vector<cv::Size> sizes_;
    std::vector<std::string> names_;

    void set_scales(const cv::Size& size);
    MegapixDownscaler& get_scaler(Resolution resolution);

    static cv::Mat read_image(const std::string& img_name);
    static cv::Size get_image_size(const cv::Mat& img);
    static cv::Mat resize_img_by_scaler(const MegapixDownscaler& scaler, const cv::Size& size, const cv::Mat& img);
    static void check_resolution(Resolution resolution);
    static std::vector<std::string> resolve_wildcards(const std::vector<std::string>& img_names);
    static bool check_list_element_types(const std::vector<cv::Mat>& list);
    static bool check_list_element_types(const std::vector<std::string>& list);
    static cv::Mat to_binary(const cv::Mat& img);
};

class FilenameImages : public Images {
public:
    FilenameImages(const std::vector<std::string>& images, double medium_megapix, double low_megapix, double final_megapix);

    void subset(const std::vector<int>& indices) override;
    std::vector<cv::Mat>::iterator begin() override;
    std::vector<cv::Mat>::iterator end() override;

private:
    std::vector<std::string> names_;
    std::vector<cv::Mat> images_;
};

class FilenameImages : public Images {
public:
    FilenameImages(const std::vector<std::string>& images, float medium_megapix, float low_megapix, float final_megapix)
        : Images(images, medium_megapix, low_megapix, final_megapix), names_(resolveWildcards(images)), names_set_(true), sizes_set_(false) {
        if (names_.size() < 2) {
            throw StitchingError("2 or more Images needed");
        }
    }

    void subset(const std::vector<int>& indices) override {
        Images::subset(indices);
    }

    class Iterator {
    public:
        Iterator(FilenameImages& parent, size_t index) : parent_(parent), index_(index) {}

        bool operator!=(const Iterator& other) const {
            return index_ != other.index_;
        }

        cv::Mat operator*() {
            std::string name = parent_.names_[index_];
            cv::Mat img = Images::readImage(name);
            cv::Size size = Images::getImageSize(img);

            // ------
            // Attention for side effects!
            // the scalers are set on the first run
            parent_._setScales(size);

            // the original image sizes are set on the first run
            if (!parent_.sizes_set_) {
                parent_.sizes_.push_back(size);
                if (index_ + 1 == parent_.names_.size()) {
                    parent_.sizes_set_ = true;
                }
            }
            // ------

            return img;
        }

        Iterator& operator++() {
            ++index_;
            return *this;
        }

    private:
        FilenameImages& parent_;
        size_t index_;
    };

    Iterator begin() {
        return Iterator(*this, 0);
    }

    Iterator end() {
        return Iterator(*this, names_.size());
    }

private:
    std::vector<std::string> names_;
    bool names_set_;
    bool sizes_set_;
    std::vector<cv::Size> sizes_;

    void _setScales(const cv::Size& size) {
        // Implement scale setting logic here
    }
};