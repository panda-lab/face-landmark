#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_io.h"

#include <time.h>

using namespace caffe;
using namespace std;
using namespace cv;
using namespace dlib;

int main()
{
    string network = "/Users/CaffeMac/CaffeMac/Models/landmark_deploy.prototxt";
    string weights = "/Users/CaffeMac/CaffeMac/Models/landmark.caffemodel";
    string img = "/Users/CaffeMac/CaffeMac/test.jpg";
    Net<float> *net = new Net<float>(network,TEST);
    
    net->CopyTrainedLayersFrom(weights);
    
    Caffe::set_mode(Caffe::CPU);
    
    Mat image = imread(img);
    
    frontal_face_detector detector = get_frontal_face_detector();
    
    
    array2d< bgr_pixel> arrimg(image.rows, image.cols);
    for(int i=0; i < image.rows; i++)
    {
        
        for(int j=0; j < image.cols; j++)
        {
            
            arrimg[i][j].blue = image.at< cv::Vec3b>(i,j)[0];
            arrimg[i][j].green=image.at< cv::Vec3b>(i,j)[1];
            arrimg[i][j].red = image.at< cv::Vec3b>(i,j)[2];
        }
    }
    
    //开始检测，返回一系列的边界框
    clock_t start;
    int cnt = 0;
    start = clock();
    std::vector<dlib::rectangle> dets = detector(arrimg);
    start = clock() -start;
    printf("detection time is: %f ms\n", (double(start))/CLOCKS_PER_SEC*1000);
    //cout<<"time is  "<<start/10e3<<endl;
    //cout << cnt++ <<endl;
    for(int i = 0;i < dets.size();i++)
    {
        //Bbox tmp = DetBox[i];
        dlib::rectangle tmp = dets[i];
        //cv::rectangle(image, Point(tmp.y1, tmp.x1), Point(tmp.y2, tmp.x2), Scalar(0,0,255), 1,4,0);
        //Mat srcROI(image, Rect(tmp.y1,tmp.x1,tmp.y2 - tmp.y1,tmp.x2-tmp.x1));
        cv::rectangle(image, Point(tmp.left(), tmp.top()), Point(tmp.right(), tmp.bottom()), Scalar(0,0,255), 1,4,0);
        Mat srcROI(image, Rect(tmp.left(),tmp.top(),tmp.right()-tmp.left(),tmp.bottom() - tmp.top()));
        
        Mat img2;
        cvtColor(srcROI,img2,CV_RGB2GRAY);
        
        img2.convertTo(img2, CV_32FC1);
        Size dsize = Size(60,60);
        Mat img3 = Mat(dsize, CV_32FC1);
        resize(img2, img3, dsize, 0,0,INTER_CUBIC);
        
        Mat tmp_m, tmp_sd;
        double m = 0, sd = 0;
        meanStdDev(img3, tmp_m, tmp_sd);
        m = tmp_m.at<double>(0,0);
        sd = tmp_sd.at<double>(0,0);
        
        img3 = (img3 - m)/(0.000001 + sd);
        
        if (img3.channels() * img3.rows * img3.cols != net->input_blobs()[0]->count())
            LOG(FATAL) << "Incorrect " << image << ", resize to correct dimensions.\n";
        // prepare data into array
        float *data = (float*)malloc( img3.rows * img3.cols * sizeof(float));
        
        int pix_count = 0;
        
        for (int i = 0; i < img3.rows; ++i) {
            for (int j = 0; j < img3.cols; ++j) {
                float pix = img3.at<float>(i, j);
                float* p = (float*)(data);
                p[pix_count] = pix;
                ++pix_count;
            }
        }
        
        std::vector<Blob<float>*> in_blobs = net->input_blobs();
        in_blobs[0]->Reshape(1, 1, img3.rows, img3.cols);
        net->Reshape();
        
        in_blobs[0]->set_cpu_data((float*)data);
        Timer total_timer;
        total_timer.Start();
        
        net->Forward();
        cout << " total time = " << total_timer.MicroSeconds() / 1000 <<endl;
        const boost::shared_ptr<Blob<float> > feature_blob = net->blob_by_name("Dense3");//获取该层特征
        
        float feat_dim = feature_blob->count() / feature_blob->num();//计算特征维度
        cout << feat_dim << endl;
        const float* data_ptr = (const float *)feature_blob->cpu_data();//特征块数据
        
        
        std::vector<float> feat2;
        
        for (int i = 0; i < feat_dim; i++)
        {
            feat2.push_back(*data_ptr);
            if (i < feat_dim - 1)
                data_ptr++;
        }
        
        for(int i = 0;i < feat_dim/2;i++)
        {
            Point x = Point(int(feat2[2*i]*(tmp.right() - tmp.left()) + tmp.left()),int(feat2[2*i + 1]*(tmp.bottom() - tmp.top()) + tmp.top()));
            cv::circle(image, x, 0.1, Scalar(0, 0, 255), 4, 8, 0);
        }
        free(data);    
    }
    imshow("result", image);
    imwrite("/Users/CaffeMac/CaffeMac/result.jpg", image);
    free(net);
    
    image.release();
    return 0;
    
}
