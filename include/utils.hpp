#ifndef __MTCNN_UTILS_HPP__
#define __MTCNN_UTILS_HPP__

/* get current time: in us */
unsigned long get_cur_time(void);

/* 
   for debug purpose, to save a image or float vector to file.
   the image should be in cv::Mat.
   To avoid OpenCV header file dependency, use void * instead of cv::Mat *
*/


void save_img(const char * name,void * p_img );  

void save_float(const char * name, const float * data, int size);


#endif

