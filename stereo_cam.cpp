#include "stereo_cam.h"

void StereoCam::getCameraData(std::string left_file, std::string right_file, std::string info_file) {
    cv::FileStorage leftFs(left_file, cv::FileStorage::READ);
    cv::FileStorage rightFs(right_file, cv::FileStorage::READ);
    cv::FileStorage infoFs(info_file, cv::FileStorage::READ);

    leftFs["camera_matrix"] >> this->K1;
    leftFs["distortion_coefficients"] >> this->D1;

    rightFs["camera_matrix"] >> this->K2;
    rightFs["distortion_coefficients"] >> this->D2;

    infoFs["R"] >> this->R;
    infoFs["T"] >> this->T;

    leftFs.release();
    rightFs.release();
    infoFs.release();
}

StereoCam::StereoCam(const std::string& left_file, const std::string& right_file, const std::string& info_file,
                     int max_disp, int wsize, double lambda, double sigma, double downsample) {
    getCameraData(left_file, right_file, info_file);
    this->max_disp = max_disp;
    this->wsize = wsize;
    this->lambda = lambda;
    this->sigma = sigma;
    this->downsample = downsample;

    this->max_disp/=2;
    if(this->max_disp%16!=0)
        this->max_disp += 16-(this->max_disp%16);

    this->left_matcher = cv::StereoBM::create(this->max_disp, this->wsize);
    this->wls_filter = cv::ximgproc::createDisparityWLSFilter(this->left_matcher);
}

StereoCam::StereoCam(const std::string& left_file, const std::string& right_file, const std::string& info_file,
                     int max_disp, int wsize, double lambda, double sigma, double downsample, const cv::Size& imgsize) {
    StereoCam(left_file, right_file, info_file, max_disp, wsize, lambda, sigma, downsample);
    this->imageSize = imgsize;
    this->calculateQ();
}

StereoCam::StereoCam(const std::string& left_file, const std::string& right_file, const std::string& info_file) {
    StereoCam(left_file, right_file, info_file, 300, 15, 8000.0, 1.5, 0.8);
}

void StereoCam::calculateQ() {
    if (!this->Q_calculated) {
        stereoRectify(this->K1, this->D1, this->K2, this->D2, this->imageSize, this->R, this->T, this->R1, this->R2, this->P1, this->P2, this->Q, cv::CALIB_SAME_FOCAL_LENGTH, -1, cv::Size(), &this->valid_roi_L, &this->valid_roi_R);
        this->left_matcher->setROI1(this->valid_roi_L); this->left_matcher->setROI2(this->valid_roi_R);
        this->right_matcher = cv::ximgproc::createRightMatcher(this->left_matcher);
        this->Q_calculated = true;
    }
}

void StereoCam::process(cv::Mat img_L, cv::Mat img_R, cv::Mat& disparity_map) {
    assert(img_L.rows == img_R.rows && img_L.cols == img_R.cols);
    cv::Size inputSize(img_L.cols, img_L.rows);
    if (inputSize != this->imageSize) {
        this->Q_calculated = false;
        this->imageSize = inputSize;
        calculateQ();
    }

    cv::undistort(img_L, this->undistort_L, this->K1, this->D1);
    cv::undistort(img_R, this->undistort_R, this->K2, this->D2);

    cv::resize(this->undistort_L, this->imgL_downsample, cv::Size(), this->downsample, this->downsample, cv::INTER_LINEAR_EXACT);
    cv::resize(this->undistort_R, this->imgR_downsample, cv::Size(), this->downsample, this->downsample, cv::INTER_LINEAR_EXACT);

    cv::cvtColor(this->imgL_downsample,  this->imgL_downsample,  cv::COLOR_BGR2GRAY);
    cv::cvtColor(this->imgR_downsample, this->imgR_downsample, cv::COLOR_BGR2GRAY);

    this->left_matcher->compute(this->imgL_downsample, this->imgR_downsample, this->imgDisparity_L);
    this->right_matcher->compute(this->imgR_downsample, this->imgL_downsample, this->imgDisparity_R);

    this->wls_filter->setLambda(this->lambda);
    this->wls_filter->setSigmaColor(this->sigma);

    this->wls_filter->filter(this->imgDisparity_L, this->undistort_L, this->imgDisparity, this->imgDisparity_R);

    this->imgDisparity.convertTo(this->dmap, CV_32F, 1.0/16.0, 0.0);
    disparity_map = dmap.clone();
}

void StereoCam::getDisparityVisualisation(cv::Mat &disparity_vis, double vis_mult) {
    cv::ximgproc::getDisparityVis(this->imgDisparity, this->disparityVis, vis_mult);
}

void StereoCam::undistorted(cv::Mat& undistort_L_crop, cv::Mat& undistort_R_crop) {
    undistort_L_crop = this->undistort_L.clone();
    undistort_R_crop = this->undistort_R.clone();
}

void StereoCam::getPointCloud(cv::Mat &xyz) {
    reprojectImageTo3D(this->dmap, xyz, this->Q, false, CV_32F);
}

