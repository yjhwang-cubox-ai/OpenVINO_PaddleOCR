//#include "ocr_cls.h"
//
//namespace PaddleOCR {
//
//Classifier::Classifier(std::string model_path)
//{
//    this->model_path = model_path;
//    ov::Core core;
//    this->model = core.read_model(this->model_path);
//    // dimension of batch size is dynamic
//    this->model->reshape({{ov::Dimension(1, 6), cls_image_shape[0], cls_image_shape[1], cls_image_shape[2]}});
//
//    // preprocessing API
//    ov::preprocess::PrePostProcessor prep(this->model);
//    // declare section of desired application's input format
//    prep.input().tensor()
//        .set_layout("NHWC")
//        .set_color_format(ov::preprocess::ColorFormat::BGR);
//    // specify actual model layout
//    prep.input().model()
//        .set_layout("NCHW");
//    prep.input().preprocess()
//        .mean(this->mean_)
//        .scale(this->scale_);
//    // dump preprocessor
//    std::cout << "Preprocessor: " << prep << std::endl;
//    this->model = prep.build();
//    this->compiled_model = core.compile_model(this->model, "CPU");
//    this->infer_request = compiled_model.create_infer_request();
//}
//
//void Classifier::Run(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results)
//{
//    std::vector<int> cls_labels(img_list.size(), 0);
//    std::vector<float> cls_scores(img_list.size(), 0);
//    std::vector<double> cls_times;
//
//    auto input_port = this->compiled_model.input();
//    int img_num = img_list.size();
//    for (int beg_img_no = 0; beg_img_no < img_num; beg_img_no += this->cls_batch_num_) {
//        int end_img_no = std::min(img_num, beg_img_no + this->cls_batch_num_);
//        int batch_num = end_img_no - beg_img_no;
//        
//        std::vector<ov::Tensor> batch_tensors;
//        ov::Shape intput_shape = {batch_num, cls_image_shape[1], cls_image_shape[2],3};
//        for (int ino = beg_img_no; ino < end_img_no; ino++) {
//            cv::Mat srcimg;
//            img_list[ino].copyTo(srcimg);
//            cv::Mat resize_img;
//            // preprocess 
//            this->resize_op_.Run(srcimg, resize_img, this->cls_image_shape);
//            resize_img.convertTo(resize_img, CV_32FC3, e);
//            if (resize_img.cols < cls_image_shape[2]) {
//                cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
//                                cls_image_shape[2] - resize_img.cols,
//                                cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
//            }
//            // prepare input tensor
//            ov::Tensor input_tensor(input_port.get_element_type(), intput_shape, (float*)resize_img.data);
//            batch_tensors.push_back(input_tensor);
//        }
//
//        // set batched input tensors
//        this->infer_request.set_input_tensors(batch_tensors);
//        // start inference
//        this->infer_request.infer();
//
//        // get output tensor
//        auto output = this->infer_request.get_output_tensor();
//        const float *out_data = output.data<const float>();
//        for (size_t batch_idx = 0; batch_idx < output.get_size() / 2; batch_idx++){
//            int label = int(
//                Utility::argmax(&out_data[batch_idx * 2],
//                            &out_data[(batch_idx + 1) * 2]));
//            float score = float(*std::max_element(
//                &out_data[batch_idx * 2],
//                &out_data[(batch_idx + 1) * 2]));
//            cls_labels[beg_img_no + batch_idx] = label;
//            cls_scores[beg_img_no + batch_idx] = score;
//        }
//    }
//
//    for (int i = 0; i < cls_labels.size(); i++) {
//        ocr_results[i].cls_label = cls_labels[i];
//        ocr_results[i].cls_score = cls_scores[i];
//    }
//}
//}
