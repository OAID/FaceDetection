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
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "mtcnn.hpp"
#include "utils.hpp"

#define DISP_WINNANE "camera"
#define QUIT_KEY     'q'
#define CAMID         0

int main(int argc, char * argv[])
{
    const char * type = "mxnet";
    double ftick, etick;
    double ticksPerUs;

    int res;

    while ((res = getopt(argc, argv, "t:")) != -1) {
        switch (res) {
            case 't':
                type = optarg;
                break;
            default:
                break;
        }
    }

    cv::VideoCapture camera(CAMID);

    if (!camera.isOpened()) {
        std::cerr << "failed to open camera" << std::endl;
        return 1;
    }

    std::string model_dir = "./models";

    mtcnn * p_mtcnn = mtcnn_factory::create_detector(type);

    if (p_mtcnn == nullptr) {
        std::cerr << type << " is not supported" << std::endl;
        std::cerr << "supported types: ";
        std::vector<std::string> type_list = mtcnn_factory::list();

        for (unsigned int i = 0; i < type_list.size(); i++)
            std::cerr << " " << type_list[i];

        std::cerr << std::endl;

        return 1;
    }

    ticksPerUs = cv::getTickFrequency() / 1000000;

    p_mtcnn->load_model(model_dir);
    cv::namedWindow(DISP_WINNANE, cv::WINDOW_AUTOSIZE);
    cv::Mat frame;
    std::vector<face_box> face_info;

    do {
            camera >> frame;

            if (!frame.data) {
                std::cerr << "Capture video failed" << std::endl;
                break;
            }

            ftick = cv::getCPUTickCount();
            p_mtcnn->detect(frame, face_info);
            etick = cv::getCPUTickCount();

            for (unsigned int i = 0; i < face_info.size(); i++) {
                face_box & box = face_info[i];

                /*draw box */
                cv::rectangle(frame, cv::Point(box.x0, box.y0),
                        cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 1);

                /* draw landmark */
                for (int l = 0; l < 5; l++) {
                    cv::circle(frame, cv::Point(box.landmark.x[l],
                        box.landmark.y[l]), 1, cv::Scalar(0, 0, 255), 1.8);
                }
            }

            std::cout<<"total detected: "<<face_info.size()<<" faces. used "<<(etick - ftick)/ticksPerUs<<" us"<<std::endl;

            cv::imshow(DISP_WINNANE, frame);
            face_info.clear();
    } while (QUIT_KEY != cv::waitKey(1));

    return 0;
}
