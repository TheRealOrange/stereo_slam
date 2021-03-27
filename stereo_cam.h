#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>

class StereoCam {
private:
    cv::Ptr<cv::StereoBM> left_matcher;
    cv::Ptr<cv::StereoMatcher> right_matcher;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;

    cv::Mat imgL_downsample;
    cv::Mat imgR_downsample;

    cv::Mat undistort_L;
    cv::Mat undistort_R;
    cv::Mat imgDisparity_L;
    cv::Mat imgDisparity_R;

    cv::Mat imgDisparity;
    cv::Mat dmap, dmap_valid;
    cv::Size imageSize;
    cv::Mat map1_L, map2_L, map1_R, map2_R;
    bool Q_calculated = false;

    void calculateQ();
    void getCameraData(std::string left_file, std::string right_file, std::string info_file);

public:
    int max_disp;
    int wsize;
    double lambda;
    double sigma;
    double downsample;

    cv::Mat K1, D1, R1, P1;
    cv::Mat K2, D2, R2, P2;
    cv::Mat Q, R;
    cv::Vec3d T;
    cv::Rect valid_roi_L, valid_roi_R, valid_disp_roi;

    cv::Mat disparityVis;

    StereoCam(const std::string& left_file, const std::string& right_file, const std::string& info_file);
    StereoCam(const std::string& left_file, const std::string& right_file, const std::string& info_file,
              int max_disp, int wsize, double lambda, double sigma, double downsample);
    StereoCam(const std::string& left_file, const std::string& right_file, const std::string& info_file,
              int max_disp, int wsize, double lambda, double sigma, double downsample, const cv::Size& imgsize);

    void process(cv::Mat img_L, cv::Mat img_R, cv::Mat& disparity_map);
    void getDisparityVisualisation(cv::Mat& disparity_vis, double vis_mult);
    void getUndistortedImages(cv::Mat& undistort_L_crop, cv::Mat& undistort_R_crop);
    void getValidImage(cv::Mat& valid_undistort);

    void getPointCloud(cv::Mat& xyz);

};