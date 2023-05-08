#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

vector<vector<bool>> magMask;

double diceCoefficient(Mat &mask1, Mat &mask2) {
	double intersectionArea = 0, mask1Area = 0, mask2Area = 0;
    for (int i = 0; i < mask1.rows; i += 1) {
        for (int j = 0; j < mask1.cols; j += 1) {
			uchar &msk1 = mask1.at<uchar>(i, j);
			uchar &msk2 = mask2.at<uchar>(i, j);
			if (msk1 > 128 && msk2 == 255) intersectionArea++;
			if (msk1 > 128) mask1Area++;
			if (msk2 == 255) mask2Area++;
		}
	}

    double dice = (2 * intersectionArea) / (mask1Area + mask2Area);
    return dice;
}

pair<int, int> transformation(int y, int x, float magn, float angle) {
    int i = y + sin(angle * 2 * CV_PI / 360.0 ) * magn,
        j = x + cos(angle * 2 * CV_PI / 360.0 ) * magn;
	i = min(max(i, 0), (int)magMask.size() - 1);
	j = min(max(j, 0), (int)magMask[0].size() - 1);
    return { i, j };
}

Mat flowVizualization(Mat &orig, Mat &flow, Mat &mask) {

	GaussianBlur(flow, flow, Size(5, 5), 0);

	Mat flow_magnitude, angle;
    vector<Mat> flow_channels;
    split(flow, flow_channels);
    cartToPolar(flow_channels[0], flow_channels[1], flow_magnitude, angle, true);

    Mat flow_magnitude_1d = flow_magnitude.reshape(1, 1);
    Mat labels, centers;
    kmeans(flow_magnitude_1d, 2, labels, TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    float thresh = (centers.at<float>(0) + centers.at<float>(1)) / 2;

    Mat flow_thresh;
    threshold(flow_magnitude, flow_thresh, thresh, 1, THRESH_BINARY);

	Mat img(orig);
	vector<vector<bool>> magMaskCopy(magMask.size(), vector(magMask[0].size(), false));
	
	vector<vector<int>> result(orig.rows, vector<int>(orig.cols, 0));


	for (int i = 0; i < orig.rows; i += 1) {
        for (int j = 0; j < orig.cols; j += 1) {
			Vec3b &org = img.at<Vec3b>(i, j);
			float &ang = angle.at<float>(i, j);
			float &thr = flow_thresh.at<float>(i, j);
			float &mag = flow_magnitude.at<float>(i, j);
			if (thr > 0 || (magMask[i][j] && mag > thresh / 8)) {
				result[i][j] = 1;
            }
        }
    }

	vector<vector<int>> lf(orig.rows, vector<int>(orig.cols, 0));
	vector<vector<int>> up(orig.rows, vector<int>(orig.cols, 0));

	for (int i = 0; i < orig.rows; ++i) lf[i][0] = result[i][0];
	for (int j = 0; j < orig.cols; ++j) up[0][j] = result[0][j];

	for (int i = 0; i < orig.rows; ++i) {
        for (int j = 1; j < orig.cols; ++j) {
			lf[i][j] = result[i][j] + lf[i][j - 1];
		}
	}
	for (int j = 0; j < orig.cols; ++j) {
		for (int i = 1; i < orig.rows; ++i) {
			up[i][j] = result[i][j] + up[i - 1][j];
		}
	}

	for (int i = 0; i < orig.rows; i += 1) {
        for (int j = 0; j < orig.cols; j += 1) {
			if (!result[i][j]) {
				int t = 0;
				int l = max(j - 50, 0), r = min(j + 50, orig.cols - 1),
					u = max(i - 50, 0), d = min(i + 50, orig.rows - 1);
				if (lf[i][j] - lf[i][l] != 0) t++;
				if (lf[i][j] - lf[i][r] != 0) t++;
				if (lf[i][j] - lf[u][j] != 0) t++;
				if (lf[i][j] - lf[d][j] != 0) t++;
				if (t >= 4) result[i][j] = true;
			} 
		}
	}

	for (int i = 0; i < orig.rows; i += 1) {
        for (int j = 0; j < orig.cols; j += 1) {
			uchar &msk = mask.at<uchar>(i, j);
			Vec3b &org = img.at<Vec3b>(i, j);
			float &ang = angle.at<float>(i, j);
			float &mag = flow_magnitude.at<float>(i, j);
			if (result[i][j]) {
				msk = 255;
                org[0] *= 0.7; org[1] *= 0.7; org[2] = org[2] * 0.7 + 255 * 0.3;
                pair<int,int> tr(transformation(i, j, mag, ang));
                magMaskCopy[tr.first][tr.second] = true;
            }
        }
    }

	magMask = magMaskCopy;
	return img;

    // angle *= ((1.f / 360.f) * (180.f / 255.f));
    // Mat _hsv[3], hsv, hsv8, bgr;
    // _hsv[0] = angle;
    // _hsv[1] = Mat::ones(angle.size(), CV_32F);
    // _hsv[2] = magn_norm;
    // merge(_hsv, 3, hsv);
    // hsv.convertTo(hsv8, CV_8U, 255.0);
    // cvtColor(hsv8, bgr, COLOR_HSV2BGR);
    // return bgr;
}

Mat FarnebackFlow(Mat &frameOld, Mat &frame, Mat &mask) {
    Mat prvs, next;
    cvtColor(frameOld, prvs, COLOR_BGR2GRAY);
    cvtColor(frame, next, COLOR_BGR2GRAY);

    Mat flow;
    calcOpticalFlowFarneback(prvs, next, flow, 0.9, 10, 30, 20, 5, 1.2, 0);
    
    return flowVizualization(frame, flow, mask);
}

Mat LucasKanadeFlow(Mat &frameOld, Mat &frame, Mat &mask) {
    Mat flow;
    optflow::calcOpticalFlowSparseToDense(frameOld, frame, flow);

    return flowVizualization(frame, flow, mask);
}

Mat RobustLocalFlow(Mat &frameOld, Mat &frame, Mat &mask) {
    Mat flow;
    optflow::calcOpticalFlowDenseRLOF(frameOld, frame, flow, Ptr<optflow::RLOFOpticalFlowParameter>(), 0.5f,
                                        Size(4,4), cv::optflow::InterpolationType::INTERP_EPIC,
                                        128, 0.05f, 1000.0f, 5, 100, true, 500.0f, 1.5f, false);
 
    return flowVizualization(frame, flow, mask);
}

Mat DualTVL1Flow(Mat &frameOld, Mat &frame, Mat &mask) {
    Mat_<Point2f> flow;
    Ptr<optflow::DualTVL1OpticalFlow> tvl1 = optflow::DualTVL1OpticalFlow::create();

    Mat prvs, next;
    cvtColor(frameOld, prvs, COLOR_BGR2GRAY);
    cvtColor(frame, next, COLOR_BGR2GRAY);

    tvl1->calc(prvs, next, flow);
    return flowVizualization(frame, flow, mask);
}

Mat DeepFlow(Mat &frameOld, Mat &frame, Mat &mask) {
	Mat flow;
	Ptr<DenseOpticalFlow> algo = optflow::createOptFlow_DeepFlow();

	Mat prvs, next;
    cvtColor(frameOld, prvs, COLOR_BGR2GRAY);
    cvtColor(frame, next, COLOR_BGR2GRAY);

	algo->calc(prvs, next, flow);
	return flowVizualization(frame, flow, mask);
}

Mat HornSchunckFlow(Mat &frameOld, Mat &frame) {
    int iterations = 100, avg_window = 5; double alpha = 1;

    Mat img_gray, img2_gray;
    cvtColor(frameOld, img_gray, COLOR_BGR2GRAY);
	cvtColor(frame, img2_gray, COLOR_BGR2GRAY);

    Mat img_gray_db, img2_gray_db;
    img_gray.convertTo(img_gray_db, CV_64FC1, 1.0 / 255.0);
	img2_gray.convertTo(img2_gray_db, CV_64FC1, 1.0 / 255.0);

    Mat I_t = img2_gray_db - img_gray_db;
    Mat I_x, I_y;
	int ddepth = -1;
    Sobel(img_gray_db, I_x, ddepth, 1, 0, 3);
	Sobel(img_gray_db, I_y, ddepth, 0, 1, 3);

    Mat U = Mat::zeros(I_t.rows, I_t.cols, CV_64FC1);
	Mat V = Mat::zeros(I_t.rows, I_t.cols, CV_64FC1);
    Mat kernel = Mat::ones(avg_window, avg_window, CV_64FC1) / pow(avg_window, 2);

    for (int i = 0; i < iterations; i++) {
		
		Mat U_avg, V_avg;

		Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);

		filter2D(U, U_avg, U.depth(), kernel, anchor, 0, BORDER_CONSTANT);
		filter2D(V, V_avg, V.depth(), kernel, anchor, 0, BORDER_CONSTANT);

		Mat C_prod1, C_prod2, I_x_squared, I_y_squared, I_x_C, I_y_C, C;

		multiply(I_x, U_avg, C_prod1);
		multiply(I_y, V_avg, C_prod2);
		multiply(I_x, I_x, I_x_squared);
		multiply(I_y, I_y, I_y_squared);

		Mat C_num = C_prod1 + C_prod2 + I_t; 
		Mat C_den = pow(alpha, 2) + I_x_squared + I_y_squared;
		
		divide(C_num, C_den, C);

		multiply(I_x, C, I_x_C);
		multiply(I_y, C, I_y_C);

		U = U_avg - I_x_C;
		V = V_avg - I_y_C;
	}

    Mat img;

    int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;

	int num_cols = RY + YG + GC + CB + BM + MR;
	int col = 0;

	Mat color_wheel = Mat::zeros(num_cols, 3, CV_64FC1);

	//RY calculation
	for (int i = 0; i < RY; i++) {
		color_wheel.at<double>(i, 0) = 255;
		color_wheel.at<double>(i, 1) = floor(255 * i / RY);
	}
	col += RY;

	//YG calculation
	for (int i = 0; i < YG; i++) {
		color_wheel.at<double>(col + i, 0) = 255 - floor(255 * i / YG);
		color_wheel.at<double>(col + i, 1) = 255;
	}
	col += YG;

	//GC calculation
	for (int i = 0; i < GC; i++) {
		color_wheel.at<double>(col + i, 1) = 255;
		color_wheel.at<double>(col + i, 2) = floor(255 * i / GC);
	}
	col += GC;

	//CB calculation
	for (int i = 0; i < CB; i++) {
		color_wheel.at<double>(col + i, 1) = 255 - floor(255 * i / CB);
		color_wheel.at<double>(col + i, 2) = 255;
	}
	col += CB;

	//BM calculation
	for (int i = 0; i < BM; i++) {
		color_wheel.at<double>(col + i, 2) = 255;
		color_wheel.at<double>(col + i, 0) = floor(255 * i / BM);
	}
	col += BM;

	//MR calculation
	for (int i = 0; i < MR; i++) {
		color_wheel.at<double>(col + i, 2) = 255 - floor(255 * i / MR);
		color_wheel.at<double>(col + i, 0) = 255;
	}

	Mat U_squared, V_squared, rad;
	cv::pow(U, 2, U_squared);
	cv::pow(V, 2, V_squared);
	cv::sqrt(U_squared + V_squared, rad);

	Mat a = Mat::zeros(U.rows, U.cols, CV_64FC1);

	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			double v_element = V.at<double>(i, j);
			double u_element = U.at<double>(i, j);
			a.at<double>(i, j) = atan2(-v_element, -u_element) / M_PI;
		}
	}

	Mat fk = (a + 1) / 2 * (num_cols - 1);
	Mat k0 = Mat::zeros(U.rows, U.cols, CV_64FC1);

	for (int i = 0; i < k0.rows; i++) {
		for (int j = 0; j < k0.cols; j++) {
			k0.at<double>(i, j) = floor(fk.at<double>(i, j));
		}
	}

	Mat k1 = k0 + 1;

	for (int i = 0; i < k1.rows; i++) {
		for (int j = 0; j < k1.cols; j++) {
			if (k1.at<double>(i, j) == num_cols)
				k1.at<double>(i, j) = 0;
		}
	}

	Mat f = fk - k0;
	Mat f_prime = 1 - f;

	vector<cv::Mat> channels;

	for (int i = 0; i < color_wheel.cols; i++) {
		Mat col0 = Mat::zeros(k0.rows, k0.cols, CV_64FC1);
		Mat col1 = Mat::zeros(k1.rows, k1.cols, CV_64FC1);

		for (int j = 0; j < k0.rows; j++) {
			for (int k = 0; k < k0.cols; k++) {

				double col0_index = k0.at<double>(j, k);
				col0.at<double>(j, k) = color_wheel.at<double>(col0_index, i) / 255.0;

				double col1_index = k1.at<double>(j, k);
				col1.at<double>(j, k) = color_wheel.at<double>(col1_index, i) / 255.0;
			}
		}

		Mat col_first, col_second, col;

		multiply(f_prime, col0, col_first);
		multiply(f, col1, col_second);

		col = col_first + col_second;

		for (int l = 0; l < col.rows; l++) {
			for (int m = 0; m < col.cols; m++) {
				if (rad.at<double>(l, m) <= 1) {
					double col_val = 1 - rad.at<double>(l, m) * (1 - col.at<double>(l, m));
					col.at<double>(l, m) = col_val;
				}
				else
					col.at<double>(l, m) *= 0.75;
			}
		}
		channels.push_back(col);
	}
	reverse(channels.begin(), channels.end());
	cv::merge(channels, img);

    img.convertTo(img, CV_8U, 255.0);
	bitwise_not(img, img);
    return img;
}

int main() {
    VideoCapture cap("video/31.mp4");
	VideoCapture capMask("video/31Mask.mp4");
    VideoWriter res("video/test2.mp4", VideoWriter::fourcc('m','p','4','v'),
                    cap.get(CAP_PROP_FPS), Size(cap.get(CAP_PROP_FRAME_WIDTH),
                                                cap.get(CAP_PROP_FRAME_HEIGHT)));
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat frameOld;
    int frameNum = 0;
	double dice = 0;
    while(true) {
        Mat frame, frame_mask;
        cap >> frame;
		capMask >> frame_mask;
		cvtColor(frame_mask, frame_mask, COLOR_BGR2GRAY);
        if (frame.empty()) break;
        if (frameNum) {
			Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8U);
            res.write(RobustLocalFlow(frameOld, frame, mask));
			cout << diceCoefficient(frame_mask, mask) << ' ';
			dice += diceCoefficient(frame_mask, mask);
            // imshow("text", RobustLocalFlow(frameOld, frame));
            // waitKey();
        } else {
            magMask = vector(frame.rows, vector(frame.cols, false));
        }
        frameOld = frame;

        cout << frameNum << endl;
        if (frameNum == 50) break;

		frameNum++;
    }
	cout << "Result Dice: " << dice / frameNum;
    cap.release();
    res.release();
}