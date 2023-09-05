#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include "args.h"
#include "paddleocr.h"
#include "paddlestructure.h"
#include <gflags/gflags.h>

using namespace PaddleOCR;

void check_params()
{
  if (FLAGS_type == "ocr")
  {
    if (FLAGS_det_model_dir.empty() || FLAGS_rec_model_dir.empty())
    {
      std::cout << "Need a path to detection and recogition model"
                   "[Usage] --det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ --rec_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << std::endl;
      exit(1);
    }
  }
  else if (FLAGS_type == "structure")
  {
    if (FLAGS_det_model_dir.empty() || FLAGS_rec_model_dir.empty() || FLAGS_lay_model_dir.empty() || FLAGS_tab_model_dir.empty())
    {
      std::cout << "Need a path to detection, recogition, layout and table model"
                   "[Usage] --det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ --rec_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ --lay_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ --tab_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << std::endl;
      exit(1);
    }
  }
}

int main(int argc, char *argv[])
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  check_params();

  // read image
  cv::Mat src_img = imread(FLAGS_input);

  if (FLAGS_type == "ocr")
  {
    PPOCR ppocr;
    std::vector<OCRPredictResult> ocr_result = ppocr.ocr(src_img);
    Utility::print_result(ocr_result);
    Utility::VisualizeBboxes(src_img, ocr_result,
                             "./ocr_result.jpg");
  }
}
