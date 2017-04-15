#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void computeCostVolume(const Mat &imgLeft, const Mat &imgRight,
                       std::vector<Mat> &costVolumeLeft, std::vector<Mat> &costVolumeRight,
                       int windowSize, int maxDisparity);

int main()
{

    Mat imgLeft, imgRight;
    imgLeft  = imread("images/tsukuba_left.png", CV_LOAD_IMAGE_COLOR);
    imgRight = imread("images/tsukuba_right.png", CV_LOAD_IMAGE_COLOR);
    if (!imgLeft.data || !imgRight.data) {
        std::cout <<  "INPUT ERROR: Could not open or find images" << std::endl;
        return -1;
    }

    std::vector<Mat> costVolumeLeft;
    std::vector<Mat> costVolumeRight;
    int windowSize = 5; // must be uneven
    int maxDisparity = 15;
    computeCostVolume(imgLeft, imgRight, costVolumeLeft, costVolumeRight, windowSize, maxDisparity);

    namedWindow("OpenCV", WINDOW_AUTOSIZE);
    imshow("OpenCV", costVolumeRight[5]);

    waitKey(0); // wait for a keystroke in the window
    
    return 0;
    
}


// given two rectified stereo images left and right we want to find the disparity,
// i.e. the horizontal left shift for the right image so that it best matches the left image
// based on the absolute pixel color difference (cost) in a given window
void computeCostVolume(const Mat &imgLeft, const Mat &imgRight, 
                       std::vector<Mat> &costVolumeLeft, std::vector<Mat> &costVolumeRight, 
                       int windowSize, int maxDisparity)
{

    Mat costMat = Mat(imgLeft.rows, imgRight.cols, CV_8UC1, 0.); // 8 bit unsigned char, one channel (grayscale), filled with zeros

    for (int disparity = 0; disparity <= maxDisparity; ++disparity) {

        for (int y = windowSize/2 + disparity; y < imgLeft.rows - windowSize/2; ++y) {
        for (int x = windowSize/2 + disparity; x < imgLeft.cols - windowSize/2; ++x) {

            // take windows as image submatrices making sure they do not exceed the image frame
            // for right image shift sample position x to left by disparity (no y shift needed since rectified)
            Mat windowLeft = imgLeft.rowRange(y-windowSize/2, y+windowSize/2)
                                    .colRange(x-windowSize/2, x+windowSize/2);
            Mat windowRight = imgRight.rowRange(y-windowSize/2, y+windowSize/2)
                                      .colRange(x-windowSize/2-disparity, x+windowSize/2-disparity);

            // take elementwise absolute color differences between all pixels in left and right window
            Mat windowAbsDiff;
            absdiff(windowLeft, windowRight, windowAbsDiff);

            // the sum of the absolute color differences in the window is the cost at this pixel
            // note Mat::at takes y (row) as first argument, then x (col)
            Scalar sumChannel = sum(windowAbsDiff); // Scalar is used to pass pixel values, it is a vector of color channels, but not a vector of pixels
            costMat.at<uchar>(y, x) = sumChannel[0] + sumChannel[1] + sumChannel[2];

        }
        }

        // append costMat for this disparity to costVolume and try with next disparity
        costVolumeRight.push_back(costMat);

    }

}

