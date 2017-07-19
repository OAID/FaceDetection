/*
  Copyright (C) 2017 Open Intelligent Machines Co.,Ltd

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/
#include <fstream>
#include <utility>
#include <vector>
#include <assert.h>
#include <stdio.h>

#include "tensorflow/c/c_api.h"

#include "mtcnn.hpp"
#include "comm_lib.hpp"
#include "utils.hpp"
#include "tensorflow_mtcnn.hpp"

static int load_file(const std::string & fname, std::vector<char>& buf)
{
	std::ifstream fs(fname, std::ios::binary | std::ios::in);

	if(!fs.good())
	{
		std::cerr<<fname<<" does not exist"<<std::endl;
		return -1;
	}


	fs.seekg(0, std::ios::end);
	int fsize=fs.tellg();

	fs.seekg(0, std::ios::beg);
	buf.resize(fsize);
	fs.read(buf.data(),fsize);

	fs.close();

	return 0;

}


static TF_Session * load_graph(const char * frozen_fname, TF_Graph ** p_graph)
{
	TF_Status* s = TF_NewStatus();

	TF_Graph* graph = TF_NewGraph();

	std::vector<char> model_buf;

	if(load_file(frozen_fname,model_buf)<0)

           return nullptr;

	TF_Buffer graph_def = {model_buf.data(), model_buf.size(), nullptr};

	TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
	TF_ImportGraphDefOptionsSetPrefix(import_opts, "");
	TF_GraphImportGraphDef(graph, &graph_def, import_opts, s);

	if(TF_GetCode(s) != TF_OK)
	{
		printf("load graph failed!\n Error: %s\n",TF_Message(s));

		return nullptr;
	}

	TF_SessionOptions* sess_opts = TF_NewSessionOptions();
	TF_Session* session = TF_NewSession(graph, sess_opts, s);
	assert(TF_GetCode(s) == TF_OK);


	TF_DeleteStatus(s);


	*p_graph=graph;

	return session;
}

static void generate_bounding_box_tf(const float * confidence_data, int confidence_size,
		const float * reg_data, float scale, float threshold, 
		int feature_h, int feature_w, std::vector<face_box>&  output, bool transposed)
{

	int stride = 2;
	int cellSize = 12;

	int img_h= feature_h;
	int img_w = feature_w;


	for(int y=0;y<img_h;y++)
		for(int x=0;x<img_w;x++)
		{
			int line_size=img_w*2;

			float score=confidence_data[line_size*y+2*x+1];

			if(score>= threshold)
			{

				float top_x = (int)((x*stride + 1) / scale);
				float top_y = (int)((y*stride + 1) / scale);
				float bottom_x = (int)((x*stride + cellSize) / scale);
				float bottom_y = (int)((y*stride + cellSize) / scale);

				face_box box;

				box.x0 = top_x;
				box.y0 = top_y;
				box.x1 = bottom_x;
				box.y1 = bottom_y;

				box.score=score;

				int c_offset=(img_w*4)*y+4*x;

				if(transposed)
				{

					box.regress[1]=reg_data[c_offset];
					box.regress[0]=reg_data[c_offset+1]; 
					box.regress[3]=reg_data[c_offset+2];
					box.regress[2]= reg_data[c_offset+3];
				}
				else {

					box.regress[0]=reg_data[c_offset];
					box.regress[1]=reg_data[c_offset+1]; 
					box.regress[2]=reg_data[c_offset+2];
					box.regress[3]= reg_data[c_offset+3];
				}

				output.push_back(box);
			}

		}
}

/* To make tensor release happy...*/
static void dummy_deallocator(void* data, size_t len, void* arg)
{
}



tf_mtcnn::~tf_mtcnn()
{
	TF_Status* s = TF_NewStatus();

	if(sess_)
	{
		TF_CloseSession(sess_,s);
		TF_DeleteSession(sess_,s);
	}

	if(graph_)
		TF_DeleteGraph(graph_);

	TF_DeleteStatus(s);
}

int tf_mtcnn::load_model(const std::string& model_dir)
{

        std::string model_fname=model_dir+"/mtcnn_frozen_model.pb";
        
	sess_=load_graph(model_fname.c_str(),&graph_);


	if(sess_==nullptr)
		return -1;

	return 0;
}

void tf_mtcnn::run_PNet(const cv::Mat& img, scale_window& win, std::vector<face_box>& box_list)
{
	cv::Mat  resized;
	int scale_h=win.h;
	int scale_w=win.w;
	float scale=win.scale;

	cv::resize(img, resized, cv::Size(scale_w, scale_h),0,0);

	/* tensorflow related*/

	TF_Status * s= TF_NewStatus();

	std::vector<TF_Output> input_names;
	std::vector<TF_Tensor*> input_values;

	TF_Operation* input_name=TF_GraphOperationByName(graph_, "pnet/input");

	input_names.push_back({input_name, 0});

	const int64_t dim[4] = {1,scale_h,scale_w,3};

	TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT,dim,4,resized.ptr(),sizeof(float)*scale_w*scale_h*3,dummy_deallocator,nullptr);

	input_values.push_back(input_tensor);



	std::vector<TF_Output> output_names;

	TF_Operation* output_name = TF_GraphOperationByName(graph_,"pnet/conv4-2/BiasAdd");
	output_names.push_back({output_name,0});

	output_name = TF_GraphOperationByName(graph_,"pnet/prob1");
	output_names.push_back({output_name,0});

	std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);


	TF_SessionRun(sess_,nullptr,input_names.data(),input_values.data(),input_names.size(),
			output_names.data(),output_values.data(),output_names.size(),
			nullptr,0,nullptr,s);


	assert(TF_GetCode(s) == TF_OK);

	/*retrieval the forward results*/

	const float * conf_data=(const float *)TF_TensorData(output_values[1]);
	const float * reg_data=(const float *)TF_TensorData(output_values[0]);


	int feature_h=TF_Dim(output_values[0],1);
	int feature_w=TF_Dim(output_values[0],2);

	int conf_size=feature_h*feature_w*2;

	std::vector<face_box> candidate_boxes;

	generate_bounding_box_tf(conf_data,conf_size,reg_data, 
			scale,pnet_threshold_,feature_h,feature_w,candidate_boxes,true);


	nms_boxes(candidate_boxes, 0.5, NMS_UNION,box_list);

	TF_DeleteStatus(s);
	TF_DeleteTensor(output_values[0]);
	TF_DeleteTensor(output_values[1]);
	TF_DeleteTensor(input_tensor);

}



static void copy_one_patch(const cv::Mat& img,face_box&input_box,float * data_to, int height, int width)
{
	cv::Mat resized(height,width,CV_32FC3,data_to);


	cv::Mat chop_img = img(cv::Range(input_box.py0,input_box.py1),
			cv::Range(input_box.px0, input_box.px1));

	int pad_top = std::abs(input_box.py0 - input_box.y0);
	int pad_left = std::abs(input_box.px0 - input_box.x0);
	int pad_bottom = std::abs(input_box.py1 - input_box.y1);
	int pad_right = std::abs(input_box.px1-input_box.x1);

	cv::copyMakeBorder(chop_img, chop_img, pad_top, pad_bottom,pad_left, pad_right,  cv::BORDER_CONSTANT, cv::Scalar(0));

	cv::resize(chop_img,resized, cv::Size(width, height), 0, 0);
}


void tf_mtcnn::run_RNet(const cv::Mat& img, std::vector<face_box>& pnet_boxes, std::vector<face_box>& output_boxes)
{
	int batch=pnet_boxes.size();
	int channel = 3;
	int height = 24;
	int width = 24;


	/* prepare input image data */

	int  input_size=batch*height*width*channel;

	std::vector<float> input_buffer(input_size);

	float * input_data=input_buffer.data();

	for(int i=0;i<batch;i++)
	{
		int patch_size=width*height*channel;

		copy_one_patch(img,pnet_boxes[i], input_data,height,width);

		input_data+=patch_size;
	}


	/* tensorflow  related */

	TF_Status * s= TF_NewStatus();

	std::vector<TF_Output> input_names;
	std::vector<TF_Tensor*> input_values;

	TF_Operation* input_name=TF_GraphOperationByName(graph_, "rnet/input");

	input_names.push_back({input_name, 0});


	const int64_t dim[4] = {batch,height,width,channel};


	TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT,dim,4,input_buffer.data(),sizeof(float)*input_size,
			dummy_deallocator,nullptr);

	input_values.push_back(input_tensor);


	std::vector<TF_Output> output_names;

	TF_Operation* output_name = TF_GraphOperationByName(graph_,"rnet/conv5-2/conv5-2");
	output_names.push_back({output_name,0});

	output_name = TF_GraphOperationByName(graph_,"rnet/prob1");
	output_names.push_back({output_name,0});

	std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);


	TF_SessionRun(sess_,nullptr,input_names.data(),input_values.data(),input_names.size(),
			output_names.data(),output_values.data(),output_names.size(),
			nullptr,0,nullptr,s);


	assert(TF_GetCode(s) == TF_OK);

	/*retrieval the forward results*/

	const float * conf_data=(const float *)TF_TensorData(output_values[1]);
	const float * reg_data=(const float *)TF_TensorData(output_values[0]);


	for(int i=0;i<batch;i++)
	{

		if(conf_data[1]>rnet_threshold_)
		{
			face_box output_box;

			face_box& input_box=pnet_boxes[i];

			output_box.x0=input_box.x0;
			output_box.y0=input_box.y0;
			output_box.x1=input_box.x1;
			output_box.y1=input_box.y1;

			output_box.score = *(conf_data+1);

			/*Note: regress's value is swaped here!!!*/

			output_box.regress[0]=reg_data[1];
			output_box.regress[1]=reg_data[0];
			output_box.regress[2]=reg_data[3];
			output_box.regress[3]=reg_data[2];

			output_boxes.push_back(output_box);


		}

		conf_data+=2;
		reg_data+=4;

	}

	TF_DeleteStatus(s);
	TF_DeleteTensor(output_values[0]);
	TF_DeleteTensor(output_values[1]);
	TF_DeleteTensor(input_tensor);
}

void tf_mtcnn::run_ONet(const cv::Mat& img, std::vector<face_box>& rnet_boxes, std::vector<face_box>& output_boxes)
{
	int batch=rnet_boxes.size();
	int channel = 3;
	int height = 48;
	int width = 48;


	/* prepare input image data */

	int  input_size=batch*height*width*channel;

	std::vector<float> input_buffer(input_size);

	float * input_data=input_buffer.data();

	for(int i=0;i<batch;i++)
	{
		int patch_size=width*height*channel;

		copy_one_patch(img,rnet_boxes[i], input_data,height,width);

		input_data+=patch_size;
	}


	/* tensorflow  related */

	TF_Status * s= TF_NewStatus();

	std::vector<TF_Output> input_names;
	std::vector<TF_Tensor*> input_values;

	TF_Operation* input_name=TF_GraphOperationByName(graph_, "onet/input");

	input_names.push_back({input_name, 0});

	const int64_t dim[4] = {batch,height,width,channel};

	TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT,dim,4,input_buffer.data(),sizeof(float)*input_size,
			dummy_deallocator,nullptr);

	input_values.push_back(input_tensor);


	std::vector<TF_Output> output_names;

	TF_Operation* output_name = TF_GraphOperationByName(graph_,"onet/conv6-2/conv6-2");
	output_names.push_back({output_name,0});

	output_name = TF_GraphOperationByName(graph_,"onet/conv6-3/conv6-3");
	output_names.push_back({output_name,0});

	output_name = TF_GraphOperationByName(graph_,"onet/prob1");
	output_names.push_back({output_name,0});

	std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);


	TF_SessionRun(sess_,nullptr,input_names.data(),input_values.data(),input_names.size(),
			output_names.data(),output_values.data(),output_names.size(),
			nullptr,0,nullptr,s);


	assert(TF_GetCode(s) == TF_OK);

	/*retrieval the forward results*/

	const float * conf_data=(const float *)TF_TensorData(output_values[2]);
	const float * reg_data=(const float *)TF_TensorData(output_values[0]);
	const float * points_data=(const float *)TF_TensorData(output_values[1]);

	for(int i=0;i<batch;i++)
	{

		if(conf_data[1]>onet_threshold_)
		{
			face_box output_box;

			face_box& input_box=rnet_boxes[i];

			output_box.x0=input_box.x0;
			output_box.y0=input_box.y0;
			output_box.x1=input_box.x1;
			output_box.y1=input_box.y1;

			output_box.score = conf_data[1];

			output_box.regress[0]=reg_data[1];
			output_box.regress[1]=reg_data[0];
			output_box.regress[2]=reg_data[3];
			output_box.regress[3]=reg_data[2];

			/*Note: switched x,y points value too..*/
			for (int j = 0; j<5; j++){
				output_box.landmark.x[j] = *(points_data + j+5);
				output_box.landmark.y[j] = *(points_data + j);
			}

			output_boxes.push_back(output_box);


		}

		conf_data+=2;
		reg_data+=4;
		points_data+=10;
	}

	TF_DeleteStatus(s);
	TF_DeleteTensor(output_values[0]);
	TF_DeleteTensor(output_values[1]);
	TF_DeleteTensor(output_values[2]);
	TF_DeleteTensor(input_tensor);

}


void tf_mtcnn::detect(cv::Mat& img, std::vector<face_box>& face_list)
{
	cv::Mat working_img;

	float alpha=0.0078125;
	float mean=127.5;



	img.convertTo(working_img, CV_32FC3);

	working_img=(working_img-mean)*alpha;

	working_img=working_img.t();

	cv::cvtColor(working_img,working_img, cv::COLOR_BGR2RGB);

	int img_h=working_img.rows;
	int img_w=working_img.cols;


	std::vector<scale_window> win_list;

	std::vector<face_box> total_pnet_boxes;
	std::vector<face_box> total_rnet_boxes;
	std::vector<face_box> total_onet_boxes;


	cal_pyramid_list(img_h,img_w,min_size_,factor_,win_list);

	for(unsigned int i=0;i<win_list.size();i++)
	{
		std::vector<face_box>boxes;

		run_PNet(working_img,win_list[i],boxes);

		total_pnet_boxes.insert(total_pnet_boxes.end(),boxes.begin(),boxes.end());
	}


	std::vector<face_box> pnet_boxes;
	process_boxes(total_pnet_boxes,img_h,img_w,pnet_boxes);


        if(pnet_boxes.size()==0)
              return;

	// RNet
	std::vector<face_box> rnet_boxes;

	run_RNet(working_img, pnet_boxes,total_rnet_boxes);

	process_boxes(total_rnet_boxes,img_h,img_w,rnet_boxes);

        if(rnet_boxes.size()==0)
              return;

	//ONet
	run_ONet(working_img, rnet_boxes,total_onet_boxes);

	//calculate the landmark

	for(unsigned int i=0;i<total_onet_boxes.size();i++)
	{
		face_box& box=total_onet_boxes[i];

		float h=box.x1-box.x0+1;
		float w=box.y1-box.y0+1;

		for(int j=0;j<5;j++)
		{
			box.landmark.x[j]=box.x0+w*box.landmark.x[j]-1;
			box.landmark.y[j]=box.y0+h*box.landmark.y[j]-1;
		}

	}


	//Get Final Result
	regress_boxes(total_onet_boxes);
	nms_boxes(total_onet_boxes, 0.7, NMS_MIN,face_list);

	//switch x and y, since working_img is transposed

	for(unsigned int i=0;i<face_list.size();i++)
	{
		face_box& box=face_list[i];

		std::swap(box.x0,box.y0);
		std::swap(box.x1,box.y1);

		for(int l=0;l<5;l++)
		{
			std::swap(box.landmark.x[l],box.landmark.y[l]);
		}
	}

}



static mtcnn * tf_creator(void)
{
	return new tf_mtcnn();
}

REGISTER_MTCNN_CREATOR(tensorflow,tf_creator);

