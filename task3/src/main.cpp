#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void computeCostVolume(const Mat &imgLeft, const Mat &imgRight,
                       std::vector<Mat> &costVolumeLeft, std::vector<Mat> &costVolumeRight,
                       int windowSize, int maxDisparity);
void calculateWindowWeights(const Mat &window, Mat &resultWeightMat);

void selectDisparity(const std::vector<Mat> &costVolumeLeft, const std::vector<Mat> &costVolumeRight, Mat &dispLeft, Mat &dispRight);
void refineDisparity(Mat &dispMatLeft, Mat &dispMatRight, int numDisparities);


#define GAMMA_C 7.f;
#define GAMMA_P 36.f;

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
	Mat dispMatLeft, dispMatRight;

    // the following parameters, especially the windowSize have to be tweaked for each pair of input images
	int windowSize = 7; // must be uneven. increase if result too noisy, decrease if result too blurry
	int maxDisparity = 15; // higher means slower, estimate based on how many pixels frontmost objects are apart. use small images!!

	std::cout << "COMPUTING COST VOLUMES (WINDOW MATCHING COST FOR EACH DISPARITY)..." << std::endl;
    computeCostVolume(imgLeft, imgRight, costVolumeLeft, costVolumeRight, windowSize, maxDisparity);

	/*/ display one slice of cost volume
	namedWindow("Cost Volume Left", WINDOW_AUTOSIZE);
	imshow("Cost Volume Left", costVolumeLeft[7]);
	namedWindow("Cost Volume Right", WINDOW_AUTOSIZE);
	imshow("Cost Volume Right", costVolumeRight[7]);
	//*/

	std::cout << "SELECTING BEST MATCHING DISPARITY..." << std::endl;
	selectDisparity(costVolumeLeft, costVolumeRight, dispMatLeft, dispMatRight);

	// convert back from float to 8 bit image
	Mat dispMatLeftConverted, dispMatRightConverted;
	dispMatLeft.convertTo(dispMatLeftConverted, CV_8UC1, 255.0f);
	dispMatRight.convertTo(dispMatRightConverted, CV_8UC1, 255.0f);

	// write results to file
	imwrite("images/result_disparity_left.png", dispMatLeftConverted);
	imwrite("images/result_disparity_right.png", dispMatRightConverted);
	std::cout << "Results saved to file: <build>/images/result_disparity_[left|right].png." << std::endl;

    // display results
	std::cout << "Displaying results." << std::endl;
	namedWindow("Disparity Map Left", WINDOW_AUTOSIZE);
	imshow("Disparity Map Left", dispMatLeftConverted);
    namedWindow("Disparity Map Right", WINDOW_AUTOSIZE);
	imshow("Disparity Map Right", dispMatRightConverted);

	std::cout << "REFINING DISPARITY MAP..." << std::endl;
	refineDisparity(dispMatLeft, dispMatRight, maxDisparity);

	// convert back from float to 8 bit image
	dispMatLeft.convertTo(dispMatLeftConverted, CV_8UC1, 255.0f);
	dispMatRight.convertTo(dispMatRightConverted, CV_8UC1, 255.0f);

	// write results to file
	imwrite("images/result_disparity_left_refined.png", dispMatLeftConverted);
	imwrite("images/result_disparity_right_refined.png", dispMatRightConverted);
	std::cout << "Results saved to file: <build>/images/result_disparity_[left|right]_refined.png." << std::endl;

	// display results
	std::cout << "Displaying refined results." << std::endl;
	namedWindow("Disparity Map Refined Left", WINDOW_AUTOSIZE);
	imshow("Disparity Map Refined Left", dispMatLeftConverted);
	namedWindow("Disparity Map Refined Right", WINDOW_AUTOSIZE);
	imshow("Disparity Map Refined Right", dispMatRightConverted);

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
// i.e. the horizontal left shift for the right image so that it best matches the left image (and vice versa).
// for each pixel, correspondence windows around pixel position in one image and displaced postion in other are compared pixelwise.
// matching cost of each pixel is based on sum of weighted absolute pixel color differences in the correspondence windows.
// the pixel absolute differences in each window are weighted based on spatial and intensity differences to window center,
// based on assumption that pixels of same object have similar color and are close to each other.
void computeCostVolume(const Mat &imgLeft, const Mat &imgRight, 
                       std::vector<Mat> &costVolumeLeft, std::vector<Mat> &costVolumeRight,
                       int windowSize, int maxDisparity)
{
	int halfWindowSize = windowSize/2; // integer division truncates, e.g. 5/2 = 2

    for (int disparity = 0; disparity <= maxDisparity; ++disparity) {

		Mat costMatRight = Mat(imgLeft.rows, imgRight.cols, CV_32FC1, 0.); // 32 bit float, one channel (grayscale), filled with zeros
		Mat costMatLeft = Mat(imgLeft.rows, imgRight.cols, CV_32FC1, 0.);

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
				Mat windowLeft = imgLeftBorder.rowRange(y-halfWindowSize, y+halfWindowSize+1).colRange(x-halfWindowSize, x+halfWindowSize+1);
				Mat windowRight = imgRightBorder.rowRange(y-halfWindowSize, y+halfWindowSize+1).colRange(x-halfWindowSize-disparity, x+halfWindowSize+1-disparity);

				Mat weightsLeft = Mat(windowSize, windowSize, CV_32FC1, 0.);
				Mat weightsRight = Mat(windowSize, windowSize, CV_32FC1, 0.);
				calculateWindowWeights(windowLeft, weightsLeft);
				calculateWindowWeights(windowRight, weightsRight);
				auto text = sum(weightsLeft.mul(weightsRight));
				double sumWindowWeights = sum(weightsLeft.mul(weightsRight))[0]; // sum() always returns Scalar, i.e. vector of channels

				// the weighted sum of the absolute color differences in the window is the cost at this pixel
                // note Mat::at takes y (row) as first argument, then x (col)
				Mat windowAbsDiffs;
				absdiff(windowLeft, windowRight, windowAbsDiffs);
				transform(windowAbsDiffs, windowAbsDiffs, Matx13f(1,1,1)); // hack to elementwise add all three color channels into single channel (CV_8UC3 -> CV_8UC1)
				windowAbsDiffs.convertTo(windowAbsDiffs, CV_32FC1, 1); // convert CV_8UC1 [0,255] image to CV_32FC1 float 32 bit single channel
				windowAbsDiffs = windowAbsDiffs.mul(weightsLeft).mul(weightsRight, 1/sumWindowWeights); // multiply each abs diff with its two weights in each of the two windows and normalize by total weights
				Scalar sumWeightedAbsDiffs = sum(windowAbsDiffs); // sum up all elements in the single channel. note: Scalar is used for pixel values, it is a vector of color channels
				costMatRight.at<float>(y - borderSize, x - borderSize) = sumWeightedAbsDiffs[0];

                // COSTVOLUMELEFT

                // take windows as image submatrices making sure they do not exceed the image frame
                // for left image shift sample position x to right by disparity (no y shift needed since rectified)
				windowLeft = imgLeftBorder.rowRange(y-halfWindowSize, y+halfWindowSize+1).colRange(x-halfWindowSize+disparity, x+halfWindowSize+1+disparity);
				windowRight = imgRightBorder.rowRange(y-halfWindowSize, y+halfWindowSize+1).colRange(x-halfWindowSize, x+halfWindowSize+1);

				// the weighted sum of the absolute color differences in the window is the cost at this pixel
                // note Mat::at takes y (row) as first argument, then x (col)
				absdiff(windowLeft, windowRight, windowAbsDiffs);
				transform(windowAbsDiffs, windowAbsDiffs, Matx13f(1,1,1)); // hack to elementwise add all three color channels into single channel (CV_8UC3 -> CV_8UC1)
				windowAbsDiffs.convertTo(windowAbsDiffs, CV_32FC1, 1); // convert CV_8UC1 [0,255] image to CV_32FC1 float 32 bit single channel
				windowAbsDiffs = windowAbsDiffs.mul(weightsLeft).mul(weightsRight, 1/sumWindowWeights); // multiply each abs diff with its two weights in each of the two windows and normalize by total weights
				sumWeightedAbsDiffs = sum(windowAbsDiffs); // sum up all elements in the single channel. note: Scalar is used for pixel values, it is a vector of color channels
				costMatLeft.at<float>(y - borderSize, x - borderSize) = sumWeightedAbsDiffs[0];

            }
        }

        // append costMat for this disparity to costVolume and calculate costs with next disparity
        costVolumeRight.push_back(costMatRight);
        costVolumeLeft.push_back(costMatLeft);
    }

}

// for window based density estimation we assume pixels in window having same disparity (belong to same object)
// since this is not always the case we weight pixels in window based on color and spatial difference from center.
// if dissimilar color and farther from center, they likely belong to other object than center pixel.
// window should be a square matrix with uneven number of rows/cols
// resultWeightMat is an empty matrix to which weights for all pixels in window will be written
void calculateWindowWeights(const Mat &window, Mat &resultWeightMat) {

	Vec2i windowCenterPos = Vec2i(window.rows/2, window.cols/2);
	Vec3b windowCenterColor = window.at<Vec3b>(windowCenterPos);

	for (int i = 0; i < window.rows; ++i) {
		for (int j = 0; j < window.cols; ++j) {

			Vec2i pixelPos = Vec2i(i, j);
			Vec3b pixelColor = window.at<Vec3b>(pixelPos);

			Vec2i spatialDiffVector;
			absdiff(pixelPos, windowCenterPos, spatialDiffVector); // abs difference of pixel position vectors
			spatialDiffVector = spatialDiffVector.mul(spatialDiffVector); // squared components x and y
			float spatialDiff = sqrt(spatialDiffVector[0] + spatialDiffVector[1]) / GAMMA_P; // euclidean distance divided by constant

			Vec3b colorDiffVector;
			absdiff(pixelColor, windowCenterColor, colorDiffVector); // abs difference of pixel color vectors
			colorDiffVector = colorDiffVector.mul(colorDiffVector); // squared components r, g, b
			float colorDiff = sqrt(colorDiffVector[0] + colorDiffVector[1] + colorDiffVector[2]) / GAMMA_C;

			resultWeightMat.at<float>(i, j) = exp(-(colorDiff + spatialDiff));
		}
	}

}


// compute left and right disparity maps from cost volumes (containing costs for each pixel and each given disparity shift)
// for each pixel the disparity with lowest cost is used
// disparities are then normalized for visualization (e.g. if costs calculated for 16 different disperities, map 15 to 1)
void selectDisparity(const std::vector<Mat> &costVolumeLeft, const std::vector<Mat> &costVolumeRight,
                     Mat &dispMatLeft, Mat &dispMatRight)
{

    if (costVolumeLeft.size() != costVolumeRight.size() || costVolumeRight.size() == 0) {
        std::cout << "ERROR: Matching Cost Volumes have unequal or zero size" << std::endl;
        return;
    }

	dispMatRight = Mat(costVolumeRight[0].size(), CV_32FC1, 0.);
	dispMatLeft = Mat(costVolumeLeft[0].size(), CV_32FC1, 0.);

	for (int y = 0; y < costVolumeLeft[0].rows; ++y) {
		for (int x = 0; x < costVolumeLeft[0].cols; ++x) {

			float minCostRight = FLT_MAX;
			float minCostLeft = FLT_MAX;
			int disparityRight = costVolumeRight.size();
			int disparityLeft = costVolumeLeft.size();

			for (int disparity = 0; disparity < costVolumeLeft.size(); ++disparity) {

                // if we find a disparity with lower cost in right cost volume, update candidate for right disparity map
				float costRight = costVolumeRight[disparity].at<float>(y, x);
                if (costRight < minCostRight) {
                    minCostRight = costRight;
                    disparityRight = disparity;
                }

                // if we find a disparity with lower cost in left cost volume, update candidate for left disparity map
				float costLeft = costVolumeLeft[disparity].at<float>(y, x);
                if (costLeft < minCostLeft) {
                    minCostLeft = costLeft;
                    disparityLeft = disparity;
                }
            }

			// normalize disparities for visualization (e.g. if costs calculated for 16 different disperities, map 15 to 1)
			int numDisparities = costVolumeLeft.size()-1;
			dispMatRight.at<float>(y, x) = disparityRight / (float)(numDisparities);
			dispMatLeft.at<float>(y, x) = disparityLeft / (float)(numDisparities);
        }
    }
}

// refine disparity map by filling in inconsistent pixels in occluded or mismatched regions
// check for inconsistent pixels by comparing each pixel from one image with its corresponding pixel in the other image.
// those pixels are invalid whose corresponding disparity differs by more than one.
void refineDisparity(Mat &dispMatLeft, Mat &dispMatRight, int numDisparities)
{
	float lastValidDisparity = 0.f;

	for (int y = 0; y < dispMatLeft.rows; ++y) {
		for (int x = 0; x < dispMatLeft.cols; ++x) {

			// check for inconsistent pixels by comparing each pixel from one image with its corresponding pixel in the other image.
			// those pixels are invalid whose corresponding disparity differs by more than one.
			// in our float image one disparity step is mapped to a step of 1/numDisparities (numDisparities is mapped to 1)
			if (abs(dispMatLeft.at<float>(y, x) - dispMatRight.at<float>(y, x)) > 1/(float)(numDisparities)) {

				// fill those invalid pixels with the minimum of their closest valid left or right neighbor’s disparity.
				dispMatLeft.at<float>(y, x) = lastValidDisparity;
				dispMatRight.at<float>(y, x) = lastValidDisparity;

			} else { // valid
				lastValidDisparity = dispMatLeft.at<float>(y, x);
			}

		}
	}
}
