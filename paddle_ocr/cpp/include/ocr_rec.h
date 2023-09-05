#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include "utility.h"
#include "preprocess_op.h"
#include "postprocess_op.h"
#include <openvino/openvino.hpp>

using namespace std;
using namespace cv;

namespace PaddleOCR {

class Recognizer
{
public:
    explicit Recognizer(string model_path, const string &label_path);
    void Run(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results);

private:
    ov::InferRequest infer_request;
    string model_path;
    shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;

    std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
    bool is_scale_ = true;
    std::vector<std::string> label_list_;
    int rec_batch_num_ = 6;
    int rec_img_h_ = 48;
    int rec_img_w_ = 320;
    std::vector<int> rec_image_shape_ = {3, rec_img_h_, rec_img_w_};
    
    CrnnResizeImg resize_op_;
    Normalize normalize_op_;
    PermuteBatch permute_op_;
};
}
