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

Mat flowVizualization(Mat &flow) {
    // visualization
    Mat flow_parts[2];
    split(flow, flow_parts);
    Mat magnitude, angle, magn_norm;
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));
    // build hsv image
    Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, bgr, COLOR_HSV2BGR);
    return bgr;
}

Mat simpleFlow(Mat &frameOld, Mat&frame) {
    Mat flow;
    optflow::calcOpticalFlowSF(frameOld, frame, flow, 3, 2, 4);

    return flowVizualization(flow);
}

Mat FarnebackFlow(Mat &frameOld, Mat &frame) {
    Mat prvs, next;
    cvtColor(frameOld, prvs, COLOR_BGR2GRAY);
    cvtColor(frame, next, COLOR_BGR2GRAY);

    Mat flow;
    calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    
    return flowVizualization(flow);
}

Mat LucasKanadeFlow(Mat &frameOld, Mat &frame) {
    Mat flow;
    optflow::calcOpticalFlowSparseToDense(frameOld, frame, flow);

    return flowVizualization(flow);
}

Mat RobustLocalFlow(Mat &frameOld, Mat &frame) {
    Mat flow;
    optflow::calcOpticalFlowDenseRLOF(frameOld, frame, flow, Ptr<optflow::RLOFOpticalFlowParameter>(), 0.5f,
                                        Size(6,6), cv::optflow::InterpolationType::INTERP_EPIC,
                                        128, 0.05f, 1000.0f, 5, 100, true, 500.0f, 1.5f, false);
 
    return flowVizualization(flow);
}

Mat DualTVL1Flow(Mat &frameOld, Mat &frame) {
    Mat prvs, next;
    cvtColor(frameOld, prvs, COLOR_BGR2GRAY);
    cvtColor(frame, next, COLOR_BGR2GRAY);

    Mat_<Point2f> flow;
    Ptr<optflow::DualTVL1OpticalFlow> tvl1 = optflow::DualTVL1OpticalFlow::create();
    tvl1->calc(prvs, next, flow);

    Mat dst; float maxmotion = -1;
    dst.create(flow.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flow.rows; ++y)
        {
            for (int x = 0; x < flow.cols; ++x)
            {
                Point2f u = flow(y, x);

                if (!cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9)
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flow.rows; ++y)
    {
        for (int x = 0; x < flow.cols; ++x)
        {
            Point2f u = flow(y, x);

            if (!cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9) {
                float fx = u.x / maxrad, fy = u.y / maxrad;

                    static bool first = true;

                    // relative lengths of color transitions:
                    // these are chosen based on perceptual similarity
                    // (e.g. one can distinguish more shades between red and yellow
                    //  than between yellow and green)
                    const int RY = 15;
                    const int YG = 6;
                    const int GC = 4;
                    const int CB = 11;
                    const int BM = 13;
                    const int MR = 6;
                    const int NCOLS = RY + YG + GC + CB + BM + MR;
                    static Vec3i colorWheel[NCOLS];

                    if (first)
                    {
                        int k = 0;

                        for (int i = 0; i < RY; ++i, ++k)
                            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

                        for (int i = 0; i < YG; ++i, ++k)
                            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

                        for (int i = 0; i < GC; ++i, ++k)
                            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

                        for (int i = 0; i < CB; ++i, ++k)
                            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

                        for (int i = 0; i < BM; ++i, ++k)
                            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

                        for (int i = 0; i < MR; ++i, ++k)
                            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

                        first = false;
                    }

                    const float rad = sqrt(fx * fx + fy * fy);
                    const float a = atan2(-fy, -fx) / (float)CV_PI;

                    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
                    const int k0 = static_cast<int>(fk);
                    const int k1 = (k0 + 1) % NCOLS;
                    const float f = fk - k0;

                    Vec3b pix;

                    for (int b = 0; b < 3; b++)
                    {
                        const float col0 = colorWheel[k0][b] / 255.f;
                        const float col1 = colorWheel[k1][b] / 255.f;

                        float col = (1 - f) * col0 + f * col1;

                        if (rad <= 1)
                            col = 1 - rad * (1 - col); // increase saturation with radius
                        else
                            col *= .75; // out of range

                        pix[2 - b] = static_cast<uchar>(255.f * col);
                    }

                dst.at<Vec3b>(y, x) = pix;
            }
        }
    }
    return dst;
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
            res.write(DualTVL1Flow(frameOld, frame));
            // imshow("text", DualTVL1Flow(frameOld, frame));
            // waitKey();
        }
        frameOld = frame;
        frameNum++;

        cout << frameNum << endl;
        if (frameNum == 100) break;
    }
    cap.release();
    res.release();
}