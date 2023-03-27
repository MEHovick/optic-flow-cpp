#include <vector>
#include <iostream>
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

pair<int, int> transformation(int x, int y, float magn, float angle) {
    int i = x + sin(angle * 2 * CV_PI / 360.0 ) * magn,
        j = y + cos(angle * 2 * CV_PI / 360.0 ) * magn;
    if (i < 0) i = 0;
    else if (i >= magMask.size()) i = magMask.size() - 1;
    if (j < 0) j = 0;
    else if (j >= magMask[0].size()) j = magMask[0].size() - 1;
    return { i, j };
}

Mat flowVizualization(Mat &orig, Mat &flow) {
    Mat flow_parts[2];
    split(flow, flow_parts);
    Mat magnitude, angle, magn_norm;
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);

    vector<vector<bool>> magMaskCopy(magMask.size(), vector(magMask[0].size(), false));

    float maxMag = 0, maxRealMag = 0;
    for (int i = 0; i < magn_norm.rows; i++) {
        for (int j = 0; j < magn_norm.cols; j++) {
            maxMag = max(maxMag, magn_norm.at<float>(i, j));
            maxRealMag = max(maxRealMag, magnitude.at<float>(i, j));
        }
    }

    Mat img(orig);
    for (int i = 0; i < magn_norm.rows; i += 1) {
        for (int j = 0; j < magn_norm.cols; j += 1) {
            float mg_norm = magn_norm.at<float>(i, j);
            float mg_orig = magnitude.at<float>(i, j);
            float angl = angle.at<float>(i, j);
            Vec3b &org = img.at<Vec3b>(i, j);
            if (maxMag - mg_norm < 0.70 || (magMask[i][j] && maxMag - mg_norm < 0.95)) {
                // Point start(j, i), backend(j + cos(angl * 2 * CV_PI / 360.0 ) * 20 * mg_norm,
                //                        i + sin(angl * 2 * CV_PI / 360.0 ) * 20 * mg_norm);
                // line(img, start, start, Scalar(0, 255, 0), 2);
                org[0] *= 0.7; org[1] *= 0.7; org[2] = org[2] * 0.7 + 255 * 0.3;
                pair<int,int> tr(transformation(i, j, mg_orig, angl));
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

Mat FarnebackFlow(Mat &frameOld, Mat &frame) {
    Mat prvs, next;
    cvtColor(frameOld, prvs, COLOR_BGR2GRAY);
    cvtColor(frame, next, COLOR_BGR2GRAY);

    Mat flow;
    calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    
    return flowVizualization(frame, flow);
}

Mat LucasKanadeFlow(Mat &frameOld, Mat &frame) {
    Mat flow;
    optflow::calcOpticalFlowSparseToDense(frameOld, frame, flow);

    return flowVizualization(frame, flow);
}

Mat RobustLocalFlow(Mat &frameOld, Mat &frame) {
    Mat flow;
    optflow::calcOpticalFlowDenseRLOF(frameOld, frame, flow, Ptr<optflow::RLOFOpticalFlowParameter>(), 0.5f,
                                        Size(6,6), cv::optflow::InterpolationType::INTERP_EPIC,
                                        128, 0.05f, 1000.0f, 5, 100, true, 500.0f, 1.5f, false);
 
    return flowVizualization(frame, flow);
}

Mat DualTVL1Flow(Mat &frameOld, Mat &frame) {
    Mat_<Point2f> flow;
    Ptr<optflow::DualTVL1OpticalFlow> tvl1 = optflow::DualTVL1OpticalFlow::create();

    Mat prvs, next;
    cvtColor(frameOld, prvs, COLOR_BGR2GRAY);
    cvtColor(frame, next, COLOR_BGR2GRAY);

    tvl1->calc(prvs, next, flow);
    return flowVizualization(frame, flow);
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
    return img;
}

int main() {
    VideoCapture cap("video/orig.mp4");
    VideoWriter res("video/test.mp4", VideoWriter::fourcc('m','p','4','v'),
                    cap.get(CAP_PROP_FPS), Size(cap.get(CAP_PROP_FRAME_WIDTH),
                                                cap.get(CAP_PROP_FRAME_HEIGHT)));
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat frameOld;
    int frameNum = 0;
    while(true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        if (frameNum) {
            res.write(RobustLocalFlow(frameOld, frame));
            // imshow("text", DualTVL1Flow(frameOld, frame));
            // waitKey();
        } else {
            magMask = vector(frame.rows, vector(frame.cols, false));
        }
        frameOld = frame;
        frameNum++;

        cout << frameNum << endl;
        if (frameNum == 150) break;
    }
    cap.release();
    res.release();
}