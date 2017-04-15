#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{

    Mat img1, img2;
    img1 = imread("images/tsukuba_left.png", CV_LOAD_IMAGE_COLOR);
    img2 = imread("images/tsukuba_right.png", CV_LOAD_IMAGE_COLOR);

    if (!img1.data || !img2.data) {
        std::cout <<  "INPUT ERROR: Could not open or find images" << std::endl;
		system("PAUSE");
		return -1;
    }

    Mat imgAbsDiff;
    absdiff(img1, img2, imgAbsDiff); // element-wise absolute difference

    namedWindow("OpenCV", WINDOW_AUTOSIZE);
    imshow("OpenCV", imgAbsDiff);

    waitKey(0); // wait for a keystroke in the window
    
    return 0;
    
}

