#ifndef __CAFFE_MTCNN_HPP__
#define __CAFFE_MTCNN_HPP__

#include <string>
#include <vector>

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

#include <opencv2/opencv.hpp>

#include "mtcnn.hpp"
#include "comm_lib.hpp"

using namespace caffe;

class caffe_mtcnn: public mtcnn {

	public:
		caffe_mtcnn()=default;

		int load_model(const std::string& model_dir);

		void detect(cv::Mat& img, std::vector<face_box>& face_list);

		~caffe_mtcnn();

	protected:

		void copy_one_patch(const cv::Mat& img,face_box&input_box,float * data_to, int width, int height);

		int run_PNet(const cv::Mat& img, scale_window& win, std::vector<face_box>& box_list);


		void run_RNet(const cv::Mat& img,std::vector<face_box>& pnet_boxes, std::vector<face_box>& output_boxes);
		void run_ONet(const cv::Mat& img,std::vector<face_box>& rnet_boxes, std::vector<face_box>& output_boxes);


	private:

		Net<float> * PNet_;
		Net<float> * RNet_;
		Net<float> * ONet_;


};


#endif
