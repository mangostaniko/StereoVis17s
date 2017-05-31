#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void computeCostVolume(const Mat &imgLeft, const Mat &imgRight,
                       std::vector<Mat> &costVolumeLeft, std::vector<Mat> &costVolumeRight,
                       int windowSize, int maxDisparity);
void selectDisparity(cv::Mat &dispLeft, cv::Mat &dispRight,
                     std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight);

void calculateWeight(cv::Mat &windowLeft);
float calculateWeightColorDifferance(int halfWindowSize, Mat window, float sumSpatialDiff);


float gamma = 14.0;
float gammaC = gamma*(1.0 / 2.0);
float gammaP = 36.0;

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
    int windowSize = 5; // must be uneven. increase if result too noisy, decrease if result too blurry
    int maxDisparity = 11; // higher means slower, estimate based on how many pixels frontmost objects are apart. use small images!!

    computeCostVolume(imgLeft, imgRight, costVolumeLeft, costVolumeRight, windowSize, maxDisparity);

    //selectDisparity(dispMatLeft, dispMatRight, costVolumeLeft, costVolumeRight);


	namedWindow("Cost Volume Left", WINDOW_AUTOSIZE);
	imshow("Cost Volume Left", costVolumeLeft[7]);
	namedWindow("Cost Volume Right", WINDOW_AUTOSIZE);
	imshow("Cost Volume Right", costVolumeRight[7]);

    // display results
    /*namedWindow("Disparity Map Right", WINDOW_AUTOSIZE);
    imshow("Disparity Map Right", dispMatRight);
    namedWindow("Disparity Map Left", WINDOW_AUTOSIZE);
    imshow("Disparity Map Left", dispMatLeft);*/

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

	//calculate spatial diff outside the for loop because it is every time the same window size, so the differences between the spatial values is every time the same range
	int halfWindowSize = floor(windowSize/2);
	float sumSpatialDiff = 0;

	Vec2i p = Vec2i(halfWindowSize, halfWindowSize);
	for (int u = 1; u <= windowSize; u++) {
		for (int v = 1; v <= windowSize; v++) {
			Vec2i q = Vec2i(u, v);
			Vec2i diff;
			absdiff(p, q, diff);
			diff = diff.mul(diff);
			float spatialDiff = sqrt(diff[0]+diff[1])/gammaP;
			sumSpatialDiff += spatialDiff;
		}
	}

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
                Mat windowLeft = imgLeftBorder.rowRange(y-halfWindowSize, y+windowSize-halfWindowSize).colRange(x- halfWindowSize, x+windowSize-halfWindowSize);
                Mat windowRight = imgRightBorder.rowRange(y-halfWindowSize, y+windowSize-halfWindowSize).colRange(x-halfWindowSize-disparity, x+windowSize-halfWindowSize-disparity);


				float weightLeft = calculateWeightColorDifferance(halfWindowSize, windowLeft, sumSpatialDiff);
				float weightRight = calculateWeightColorDifferance(halfWindowSize, windowRight, sumSpatialDiff);

				double bothWeights = weightLeft*weightRight;

                // the sum of the absolute color differences in the window is the cost at this pixel
                // note Mat::at takes y (row) as first argument, then x (col)
                Mat windowAbsDiff;
                absdiff(windowLeft, windowRight, windowAbsDiff);
                Scalar sumChannel = sum(windowAbsDiff); // Scalar is used to pass pixel values, it is a vector of color channels, but not a vector of pixels
				costMatRight.at<int>(y - borderSize, x - borderSize) = ((sumChannel[0] + sumChannel[1] + sumChannel[2]) *bothWeights) / bothWeights;

                // COSTVOLUMELEFT

                // take windows as image submatrices making sure they do not exceed the image frame
                // for left image shift sample position x to right by disparity (no y shift needed since rectified)
                windowLeft = imgLeftBorder.rowRange(y-halfWindowSize, y+windowSize- halfWindowSize).colRange(x- halfWindowSize +disparity, x+windowSize-halfWindowSize +disparity);
                windowRight = imgRightBorder.rowRange(y- halfWindowSize, y+windowSize-halfWindowSize).colRange(x- halfWindowSize, x+windowSize-halfWindowSize);
				
				weightLeft = calculateWeightColorDifferance(halfWindowSize, windowLeft, sumSpatialDiff);
				weightRight = calculateWeightColorDifferance(halfWindowSize, windowRight, sumSpatialDiff);

				bothWeights = weightLeft*weightRight;

                // the sum of the absolute color differences in the window is the cost at this pixel
                // note Mat::at takes y (row) as first argument, then x (col)
                absdiff(windowLeft, windowRight, windowAbsDiff);
                sumChannel = sum(windowAbsDiff); // Scalar is used to pass pixel values, it is a vector of color channels, but not a vector of pixels
				costMatLeft.at<int>(y - borderSize, x - borderSize) = ((sumChannel[0] + sumChannel[1] + sumChannel[2]));// *bothWeights) / bothWeights;

            }
        }

        // append costMat for this disparity to costVolume and calculate costs with next disparity
        costVolumeRight.push_back(costMatRight);
        costVolumeLeft.push_back(costMatLeft);
    }

}


float calculateWeightColorDifferance(int halfWindowSize, Mat window, float sumSpatialDiff) {
	Vec3b middleColor = window.at<Vec3b>(halfWindowSize, halfWindowSize);
	vector<Mat> channels(3);
	split(window, channels);
	Mat ch1Diff, ch2Diff, ch3Diff;

	absdiff(channels[0], middleColor[0], ch1Diff);
	ch1Diff = ch1Diff.mul(ch1Diff);
	absdiff(channels[1], middleColor[1], ch2Diff);
	ch2Diff = ch2Diff.mul(ch2Diff);
	absdiff(channels[2], middleColor[2], ch3Diff);
	ch3Diff = ch2Diff.mul(ch3Diff);

	Scalar sumCol = sum(ch1Diff) + sum(ch2Diff) + sum(ch3Diff);
	float colDiff = sqrt(sumCol[0] + sumCol[1] + sumCol[2]) / gammaC;
	return exp(-(colDiff + sumSpatialDiff));
}

//calculates the weight for one window
void calculateWeight(cv::Mat &window) {




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
