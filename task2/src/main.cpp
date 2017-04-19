#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void computeCostVolume(const Mat &imgLeft, const Mat &imgRight,
                       std::vector<Mat> &costVolumeLeft, std::vector<Mat> &costVolumeRight,
                       int windowSize, int maxDisparity);
void selectDisparity(cv::Mat &dispLeft, cv::Mat &dispRight, std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight);


int main()
{

    Mat imgLeft, imgRight;
    imgLeft  = imread("images/tsukuba_left.png", CV_LOAD_IMAGE_COLOR);
    imgRight = imread("images/tsukuba_right.png", CV_LOAD_IMAGE_COLOR);
    if (!imgLeft.data || !imgRight.data) {
        std::cout << "INPUT ERROR: Could not open or find images" << std::endl;
        system("PAUSE");
        return -1;
    }

    std::vector<Mat> costVolumeLeft, costVolumeRight;
    cv::Mat dispLeft, dispRight;
    int windowSize = 5; // must be uneven
    int maxDisparity = 15;

    computeCostVolume(imgLeft, imgRight, costVolumeLeft, costVolumeRight, windowSize, maxDisparity);

    namedWindow("Cost Map Right for disparity 5", WINDOW_AUTOSIZE);
    imshow("Cost Map Right for disparity 5", costVolumeRight[5]);
    namedWindow("Cost Map Left for disparity 5", WINDOW_AUTOSIZE);
    imshow("Cost Map Left for disparity 5", costVolumeLeft[5]);

    selectDisparity(dispLeft, dispRight, costVolumeLeft, costVolumeRight);

    namedWindow("Disparity Map Right", WINDOW_AUTOSIZE);
    imshow("Disparity Map Right", dispRight);
    namedWindow("Disparity Map Left", WINDOW_AUTOSIZE);
    imshow("Disparity Map Left", dispLeft);

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

    for (int disparity = 0; disparity <= maxDisparity; ++disparity) {

        Mat costMatRight = Mat(imgLeft.rows, imgRight.cols, CV_32SC1, 0.); // 32 bit signed integer, one channel (grayscale), filled with zeros
        Mat costMatLeft = Mat(imgLeft.rows, imgRight.cols, CV_32SC1, 0.);

        for (int y = 0 + disparity; y < imgLeft.rows - windowSize - disparity; ++y) {
            for (int x = 0 + disparity; x < imgLeft.cols - windowSize - disparity; ++x) {

                // COSTVOLUMERIGHT

                // take windows as image submatrices making sure they do not exceed the image frame
                // for right image shift sample position x to left by disparity (no y shift needed since rectified)
                Mat windowLeft = imgLeft.rowRange(y, y + windowSize).colRange(x, x + windowSize);
                Mat windowRight = imgRight.rowRange(y, y + windowSize).colRange(x - disparity, x + windowSize - disparity);

                // the sum of the absolute color differences in the window is the cost at this pixel
                // note Mat::at takes y (row) as first argument, then x (col)
                Mat windowAbsDiff;
                absdiff(windowLeft, windowRight, windowAbsDiff);
                Scalar sumChannel = sum(windowAbsDiff); // Scalar is used to pass pixel values, it is a vector of color channels, but not a vector of pixels
                costMatRight.at<int>(y, x) = sumChannel[0] + sumChannel[1] + sumChannel[2];

                // COSTVOLUMELEFT

                // take windows as image submatrices making sure they do not exceed the image frame
                // for left image shift sample position x to right by disparity (no y shift needed since rectified)
                windowLeft = imgLeft.rowRange(y, y + windowSize).colRange(x + disparity, x + windowSize + disparity);
                windowRight = imgRight.rowRange(y, y + windowSize).colRange(x, x + windowSize);

                // the sum of the absolute color differences in the window is the cost at this pixel
                // note Mat::at takes y (row) as first argument, then x (col)
                absdiff(windowLeft, windowRight, windowAbsDiff);
                sumChannel = sum(windowAbsDiff); // Scalar is used to pass pixel values, it is a vector of color channels, but not a vector of pixels
                costMatLeft.at<int>(y, x) = sumChannel[0] + sumChannel[1] + sumChannel[2];

            }
        }

        // append costMat for this disparity to costVolume and calculate costs with next disparity
        costVolumeRight.push_back(costMatRight);
        costVolumeLeft.push_back(costMatLeft);
    }

}

// compute left and right disparity maps from cost volumes (containing costs for each pixel and each given disparity shift)
// for each pixel the disparity with lowest cost is used
// disparities are then normalized for visualization (e.g. if costs calculated for 16 different disperities, map 15 to 255)
void selectDisparity(cv::Mat &dispMatLeft, cv::Mat &dispMatRight,
                     std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight)
{

    if (costVolumeLeft.size() != costVolumeRight.size() || costVolumeRight.size() == 0) {
        std::cout << "ERROR: Matching Cost Volumes have unequal or zero size" << std::endl;
        return;
    }

    dispMatRight = Mat(costVolumeRight[0].size(), CV_8UC1, 0.);
    dispMatLeft = Mat(costVolumeLeft[0].size(), CV_8UC1, 0.);

    for (int y = 0; y < costVolumeRight[0].rows; ++y) {
        for (int x = 0; x < costVolumeRight[0].cols; ++x) {

            int minCostRight = INT32_MAX;
            int minCostLeft = INT32_MAX;
            int disparityRight = costVolumeRight.size();
            int disparityLeft = costVolumeLeft.size();

            for (int disparity = 0; disparity < costVolumeRight.size(); ++disparity) {

                // if we find a disparity with lower cost in right cost volume, update candidate for right disparity map
                int costRight = costVolumeRight[disparity].at<int>(y, x);
                if (costRight < minCostRight) {
                    minCostRight = costRight;
                    disparityRight = disparity;
                }

                // if we find a disparity with lower cost in left cost volume, update candidate for left disparity map
                int costLeft = costVolumeLeft[disparity].at<int>(y, x);
                if (costLeft < minCostLeft) {
                    minCostLeft = costLeft;
                    disparityLeft = disparity;
                }
            }

            // normalize disparities for visualization (e.g. if costs calculated for 16 different disperities, map 15 to 255)
            int numDisparities = costVolumeRight.size()-1;
            dispMatRight.at<uchar>(y, x) = disparityRight / (float)(numDisparities) * 255.f;
            dispMatLeft.at<uchar>(y, x) = disparityLeft / (float)(numDisparities) * 255.f;
        }
    }
}
