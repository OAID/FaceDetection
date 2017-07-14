#ifndef __MXNET_MTCNN_HPP__
#define __MXNET_MTCNN_HPP__

#include <string>
#include <vector>

#include "mtcnn.hpp"
#include "mxnet/c_predict_api.h"
#include "comm_lib.hpp"

class mxnet_mtcnn: public mtcnn {

	public:
		mxnet_mtcnn():rnet_batch_bound_(10),onet_batch_bound_(10){};

		int load_model(const std::string& model_dir);

		void detect(cv::Mat& img, std::vector<face_box>& face_list);

                void set_batch_mode_bound(int r, int o) 
                 {
                     rnet_batch_bound_=r;
                     onet_batch_bound_=o;
                 }

		~mxnet_mtcnn();

	protected:

		void load_PNet(int h, int w);
		void free_PNet(void);

                void copy_one_patch(const cv::Mat& img,face_box&input_box,float * data_to, int width, int height);
                PredictorHandle load_RNet(int batch);
                PredictorHandle load_ONet(int batch);
                

		PredictorHandle load_mxnet_model(const std::string& param_file, const std::string& json_file, 
				int batch, int channel,int input_h, int input_w);

		void run_PNet(const cv::Mat& img, scale_window& win, std::vector<face_box>& box_list);

               	int run_preload_RNet(const cv::Mat& img, face_box& input_box, face_box& output_box);
		int run_preload_ONet(const cv::Mat& img, face_box&input_box, face_box& output_box); 

		void run_RNet(const cv::Mat& img,std::vector<face_box>& pnet_boxes, std::vector<face_box>& output_boxes);
		void run_ONet(const cv::Mat& img,std::vector<face_box>& rnet_boxes, std::vector<face_box>& output_boxes);


	private:

		std::string  model_dir_;
		PredictorHandle  PNet_;  //PNet_ will create and destroyed frequently
		PredictorHandle  RNet_;
		PredictorHandle  ONet_;
                int rnet_batch_bound_;
                int onet_batch_bound_;

};


#endif
