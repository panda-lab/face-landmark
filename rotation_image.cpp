#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include <fstream>
#include <dirent.h>
using namespace std;
using namespace cv;
#define BYTE unsigned char

Mat ImageRotate(Mat & src, const CvPoint &_center, double angle)
{
    CvPoint2D32f center;
    center.x = float(_center.x);
    center.y = float(_center.y);
    
    //计算二维旋转的仿射变换矩阵
    Mat M = getRotationMatrix2D(center, angle, 1);
    
    // rotate
    Mat dst;
    warpAffine(src, dst, M, cvSize(src.cols, src.rows), CV_INTER_LINEAR);
    return dst;
}

// 获取指定像素点放射变换后的新的坐标位置
CvPoint getPointAffinedPos(const CvPoint &src, const CvPoint center, double angle)
{
    CvPoint dst;
    int x = src.x - center.x;
    int y = src.y - center.y;
    
    dst.x = cvRound(x * cos(angle) + y * sin(angle) + center.x);
    dst.y = cvRound(-x * sin(angle) + y * cos(angle) + center.y);
    return dst;
}




int main(int argc, const char * argv[])
{
    struct dirent *ptr;
    DIR *dir;
    string PATH = "/home/workspace/Dataset/68/helen/trainset/";
    dir=opendir(PATH.c_str());
    vector<string> files;
    cout << "文件列表: "<< endl;
    while((ptr=readdir(dir))!=NULL)
    {
        
        //跳过'.'和'..'两个目录
        if(ptr->d_name[0] == '.')
            continue;
        //cout << ptr->d_name << endl;
        string tmp = ptr->d_name;
        long k = tmp.find(".png",0);
        long m = tmp.find(".jpg",0);
        if(k > 0 || m > 0)
        {
            files.push_back(ptr->d_name);
        }
    }
    closedir(dir);
    
    for (int i = 0; i < files.size(); ++i)
    {
        cout << files[i] << endl;
        string filename = "/home/workspace/Dataset/68/helen/trainset/" + files[i];
        Mat img = imread(filename);
        cout << filename << endl;
        CvPoint center;
        center.x = img.cols / 2;
        center.y = img.rows / 2;
    
    
        double angle = 9;
    
        Mat dst = ImageRotate(img, center, angle);
    
    
        //读取pts文件
        long idx = filename.find(".", 0);
	printf("---- %ld ----\n",filename.size());
        string pre_path = filename.erase(idx);
        pre_path.insert(idx, ".pts");
        ifstream ifs(pre_path);
 
    
        string rotate_pts = "/home/workspace/Dataset/train(helen+ibug+lfpw+afw)/img_09/" + files[i];
	long idx2 = rotate_pts.find(".", 0);
        rotate_pts.erase(idx2);

        string out_pts = rotate_pts.insert(idx2, "_09.pts");
        ofstream outfile(out_pts,ios::out);
    
        float x;
        float y;
        char drop;
        int cnt = 0;
        string temp;
        getline(ifs,temp);  //跳过前三行
        outfile << temp << endl;
        getline(ifs,temp);
        outfile << temp << endl;
        getline(ifs,temp);
        outfile << temp << endl;
    
        while (!ifs.eof())
        {
            if (ifs>>x>>y)
            {
                CvPoint tmp;
                tmp.x = x;
                tmp.y = y;
            
                CvPoint l2 = getPointAffinedPos(tmp, center, angle * CV_PI / 180);
                //circle(dst,l2,1,CV_RGB(0,255,255),2);
                outfile.precision(3);
                outfile.setf(ios::fixed);
                outfile << l2.x << " " << l2.y << endl;
            }
            else
            {
                ifs.clear();
                ifs.sync();
                ifs>>drop;
            }
        
            cnt++;
            if(cnt >= 68)
            {
                cnt = 0;
                break;
            }
        }
    
        outfile.close();
        //imshow("src", dst);
        string rotate_img = "/home/workspace/Dataset/train(helen+ibug+lfpw+afw)/img_09/" + files[i];
        rotate_img.erase(idx2);
        string out_img = rotate_img.insert(idx2, "_09.jpg");

        imwrite(out_img,dst);
    
        img.release();
        dst.release();

        //waitKey(0);

    }
    
  
    return 0;
}
