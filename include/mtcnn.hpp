#ifndef __MTCNN_HPP__
#define __MTCNN_HPP__

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>


struct face_landmark
{
	float x[5];
	float y[5];
};

struct face_box
{
	float x0;
	float y0;
	float x1;
	float y1;

	/* confidence score */
	float score;

	/*regression scale */

	float regress[4];

	/* padding stuff*/
	float px0;
	float py0;
	float px1;
	float py1;

	face_landmark landmark;  
};



class mtcnn {
	public:
		mtcnn(void){
			min_size_=40;
			pnet_threshold_=0.6;
			rnet_threshold_=0.7;
			onet_threshold_=0.9;
			factor_=0.709;

		}

		void set_threshold(float p, float r, float o)
		{
			pnet_threshold_=p;
			rnet_threshold_=r;
			onet_threshold_=o;
		}

		void set_factor_min_size(float factor, float min_size)
		{
			factor_=factor;
			min_size_=min_size;   
		}


		virtual int load_model(const std::string& model_dir)=0;
		virtual void detect(cv::Mat& img, std::vector<face_box>& face_list)=0;
		virtual ~mtcnn(void){};

	protected:

		int min_size_;
		float pnet_threshold_;
		float rnet_threshold_;
		float onet_threshold_;
		float factor_;	 
};

/* factory part */

class mtcnn_factory
{
	public:

		typedef mtcnn * (*creator)(void);

		static void register_creator(const std::string& name, creator& create_func);
		static mtcnn * create_detector(const std::string& name);
                static std::vector<std::string> list(void);

	private:
		mtcnn_factory(){};


};

class  only_for_auto_register
{
	public:
		only_for_auto_register(std::string name, mtcnn_factory::creator func)
		{
			std::cout<<1<<"\n"<<name;
			if(func == NULL)
					std::cout<<"func null\n";
			mtcnn_factory::register_creator(name,func);
		}

};

#define REGISTER_MTCNN_CREATOR(name,func) \
	 static only_for_auto_register __attribute__((used)) dummy_mtcnn_creator_## name (#name, func)

#endif
