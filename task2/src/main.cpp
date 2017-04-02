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
        std::cout <<  "INPUT ERROR: Could not open or find images" << std::endl;
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

        Mat costMatRight = Mat(imgLeft.rows, imgRight.cols, CV_32SC1, 0.); // 32 bit signed char, one channel (grayscale), filled with zeros
        Mat costMatLeft = Mat(imgLeft.rows, imgRight.cols, CV_32SC1, 0.);

        for (int y = 0 + disparity; y < imgLeft.rows - windowSize - disparity; ++y) {
            for (int x = 0 + disparity; x < imgLeft.cols - windowSize - disparity; ++x) {

                // COSTVOLUMERIGHT

                // take windows as image submatrices making sure they do not exceed the image frame
                // for right image shift sample position x to left by disparity (no y shift needed since rectified)
                Mat windowLeft =   imgLeft.rowRange(y, y + windowSize).colRange(x, x + windowSize);
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
                windowLeft =   imgLeft.rowRange(y, y + windowSize).colRange(x + disparity, x + windowSize + disparity);
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


void selectDisparity(cv::Mat &dispLeft, cv::Mat &dispRight, std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight)
{

    if (costVolumeLeft.size() == costVolumeRight.size()) {
        if (costVolumeRight.size() > 0) {

            dispRight = Mat(costVolumeRight[0].size(), CV_8UC1, 0.);
            dispLeft = Mat(costVolumeLeft[0].size(), CV_8UC1, 0.);

            for (int x = 0; x < costVolumeRight[0].cols; x++) {
                for (int y = 0; y < costVolumeRight[0].rows; y++) {

                    int minimumCostRight = INT32_MAX;
                    int minimumCostLeft = INT32_MAX;
                    int disparityRight = costVolumeRight.size();
                    int disparityLeft = costVolumeLeft.size();

                    for (int d = 0; d < costVolumeRight.size(); d++) {

                        //get cost value for current disparity for right and left cost volume
                        auto costRight = costVolumeRight[d].at<int>(y, x);
                        auto costLeft = costVolumeLeft[d].at<int>(y, x);

                        //check if current right cost value is smaller than the current minimum cost value of right image
                        if (costRight < minimumCostRight) {
                            //if current cost value is smaller, update minimal cost and store the current disparity value for the right disparity map
                            minimumCostRight = costRight;
                            disparityRight = d;
                        }

                        //check if current left cost value is smaller than the current minimum cost value of left image
                        if (costLeft < minimumCostLeft) {
                            //if current cost value is smaller, update minimal cost and store the current disparity value for the right disparity map
                            minimumCostLeft = costLeft;
                            disparityLeft = d;
                        }
                    }

                    //update the disparity map for right and left image and multiply with disparity value + 1
                    //note: +1 not necessary because size() already returns max disparity value + 1
                    dispRight.at<uchar>(y, x) = disparityRight *costVolumeRight.size();
                    dispLeft.at<uchar>(y, x) = disparityLeft *costVolumeLeft.size();
                }
            }
        }
    }

}
