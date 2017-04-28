#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void computeCostVolume(const Mat &imgLeft, const Mat &imgRight,
                       std::vector<Mat> &costVolumeLeft, std::vector<Mat> &costVolumeRight,
                       int windowSize, int maxDisparity);
void selectDisparity(cv::Mat &dispLeft, cv::Mat &dispRight,
                     std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight);


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
    cv::Mat dispMatLeft, dispMatRight;

    // the following parameters, especially the windowSize have to be tweaked for each pair of input images
    int windowSize = 7; // must be uneven. increase if result too noisy, decrease if result too blurry
    int maxDisparity = 15; // higher means slower, estimate based on how many pixels frontmost objects are apart. use small images!!

    computeCostVolume(imgLeft, imgRight, costVolumeLeft, costVolumeRight, windowSize, maxDisparity);

    selectDisparity(dispMatLeft, dispMatRight, costVolumeLeft, costVolumeRight);

    // display results
    namedWindow("Disparity Map Right", WINDOW_AUTOSIZE);
    imshow("Disparity Map Right", dispMatRight);
    namedWindow("Disparity Map Left", WINDOW_AUTOSIZE);
    imshow("Disparity Map Left", dispMatLeft);

    /*/ display ground truth
    Mat imgGTLeft, imgGTRight;
    imgGTLeft  = imread("images/tsukuba_displeft.png", CV_LOAD_IMAGE_COLOR);
    imgGTRight = imread("images/tsukuba_dispright.png", CV_LOAD_IMAGE_COLOR);
    namedWindow("Disparity Map GT Right", WINDOW_AUTOSIZE);
    imshow("Disparity Map GT Right", imgGTRight);
    namedWindow("Disparity Map GT Left", WINDOW_AUTOSIZE);
    imshow("Disparity Map GT Left", imgGTLeft);
    //*/

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

        // add borders / zero padding around images
        // to ensure that windows placed around original pixels and shifted by disparity never leave the image
        // without having to adaptively adjust the window size
        Mat imgLeftBorder, imgRightBorder;
        int borderSize = windowSize/2 + disparity;
        copyMakeBorder(imgLeft, imgLeftBorder, borderSize, borderSize, borderSize, borderSize, BORDER_CONSTANT, 0);
        copyMakeBorder(imgRight, imgRightBorder, borderSize, borderSize, borderSize, borderSize, BORDER_CONSTANT, 0);

        for (int y = borderSize; y < imgLeftBorder.rows - borderSize; ++y) {
            for (int x = borderSize; x < imgLeftBorder.cols - borderSize; ++x) {

                // COSTVOLUMERIGHT

                // take windows as image submatrices making sure they do not exceed the image frame
                // for right image shift sample position x to left by disparity (no y shift needed since rectified)
                Mat windowLeft = imgLeftBorder.rowRange(y-windowSize/2, y+windowSize/2).colRange(x-windowSize/2, x+windowSize/2);
                Mat windowRight = imgRightBorder.rowRange(y-windowSize/2, y+windowSize/2).colRange(x-windowSize/2-disparity, x+windowSize/2-disparity);

                // the sum of the absolute color differences in the window is the cost at this pixel
                // note Mat::at takes y (row) as first argument, then x (col)
                Mat windowAbsDiff;
                absdiff(windowLeft, windowRight, windowAbsDiff);
                Scalar sumChannel = sum(windowAbsDiff); // Scalar is used to pass pixel values, it is a vector of color channels, but not a vector of pixels
                costMatRight.at<int>(y-borderSize, x-borderSize) = sumChannel[0] + sumChannel[1] + sumChannel[2];

                // COSTVOLUMELEFT

                // take windows as image submatrices making sure they do not exceed the image frame
                // for left image shift sample position x to right by disparity (no y shift needed since rectified)
                windowLeft = imgLeftBorder.rowRange(y-windowSize/2, y+windowSize/2).colRange(x-windowSize/2+disparity, x+windowSize/2+disparity);
                windowRight = imgRightBorder.rowRange(y-windowSize/2, y+windowSize/2).colRange(x-windowSize/2, x+windowSize/2);

                // the sum of the absolute color differences in the window is the cost at this pixel
                // note Mat::at takes y (row) as first argument, then x (col)
                absdiff(windowLeft, windowRight, windowAbsDiff);
                sumChannel = sum(windowAbsDiff); // Scalar is used to pass pixel values, it is a vector of color channels, but not a vector of pixels
                costMatLeft.at<int>(y-borderSize, x-borderSize) = sumChannel[0] + sumChannel[1] + sumChannel[2];

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
