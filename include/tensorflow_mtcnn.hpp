#ifndef __TENSORFLOW_MTCNN_HPP__
#define __TENSORFLOW_MTCNN_HPP__

#include "tensorflow/c/c_api.h"
#include "mtcnn.hpp"
#include "comm_lib.hpp"

class tf_mtcnn: public mtcnn {

	public:
		tf_mtcnn()=default;

		int load_model(const std::string& model_dir);

		void detect(cv::Mat& img, std::vector<face_box>& face_list);

		~tf_mtcnn();


	protected:


		void run_PNet(const cv::Mat& img, scale_window& win, std::vector<face_box>& box_list);


		void run_RNet(const cv::Mat& img,std::vector<face_box>& pnet_boxes, std::vector<face_box>& output_boxes);
		void run_ONet(const cv::Mat& img,std::vector<face_box>& rnet_boxes, std::vector<face_box>& output_boxes);
	private:

		TF_Session * sess_;
		TF_Graph  *  graph_;

};


#endif
