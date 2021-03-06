#include "stereo_cam.h"

void StereoCam::getCameraData(std::string left_file, std::string right_file, std::string info_file) {
    cv::FileStorage leftFs(left_file, cv::FileStorage::READ);
    cv::FileStorage rightFs(right_file, cv::FileStorage::READ);
    cv::FileStorage infoFs(info_file, cv::FileStorage::READ);

    int cols, rows;
    leftFs["image_width"] >> cols;
    leftFs["image_height"] >> rows;

    this->calibSize = cv::Size(cols, rows);

    leftFs["camera_matrix"] >> this->K1;
    leftFs["distortion_coefficients"] >> this->D1;
    //leftFs["rectification_matrix"] >> this->R1;
    //leftFs["projection_matrix"] >> this->P1;

    rightFs["camera_matrix"] >> this->K2;
    rightFs["distortion_coefficients"] >> this->D2;
    //rightFs["rectification_matrix"] >> this->R2;
    //rightFs["projection_matrix"] >> this->P2;

    infoFs["R"] >> this->R;
    infoFs["T"] >> this->T;
    //infoFs["Q"] >> this->Q;

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
        stereoRectify(this->K1, this->D1, this->K2, this->D2, this->calibSize, this->R, this->T, this->R1, this->R2, this->P1, this->P2, this->Q, cv::CALIB_ZERO_DISPARITY, 1, this->imageSize, &this->valid_roi_L, &this->valid_roi_R);
        this->left_matcher->setROI1(this->valid_roi_L); this->left_matcher->setROI2(this->valid_roi_R);
        this->right_matcher = cv::ximgproc::createRightMatcher(this->left_matcher);

        this->K1_optim = cv::getOptimalNewCameraMatrix(this->K1, this->D1, this->calibSize, 1, this->imageSize);
        this->K2_optim = cv::getOptimalNewCameraMatrix(this->K2, this->D2, this->calibSize, 1, this->imageSize);

        cv::initUndistortRectifyMap(this->K1, this->D1, this->R1, this->K1_optim, this->imageSize, CV_32FC1, this->map1_L, this->map2_L);
        cv::initUndistortRectifyMap(this->K2, this->D2, this->R2, this->K2_optim, this->imageSize, CV_32FC1, this->map1_R, this->map2_R);

        this->valid_disp_roi = cv::getValidDisparityROI(this->valid_roi_L, this->valid_roi_R, left_matcher->getMinDisparity(), left_matcher->getNumDisparities(), left_matcher->getSmallerBlockSize());
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

    cv::remap(img_L, this->undistort_L, this->map1_L, this->map2_L, cv::INTER_LINEAR);
    cv::remap(img_R, this->undistort_R, this->map1_R, this->map2_R, cv::INTER_LINEAR);

    cv::resize(this->undistort_L, this->imgL_downsample, cv::Size(), this->downsample, this->downsample, cv::INTER_LINEAR_EXACT);
    cv::resize(this->undistort_R, this->imgR_downsample, cv::Size(), this->downsample, this->downsample, cv::INTER_LINEAR_EXACT);

    cv::cvtColor(this->imgL_downsample,  this->imgL_downsample,  cv::COLOR_BGR2GRAY);
    cv::cvtColor(this->imgR_downsample, this->imgR_downsample, cv::COLOR_BGR2GRAY);

    this->left_matcher->compute(this->imgL_downsample, this->imgR_downsample, this->imgDisparity_L);
    this->right_matcher->compute(this->imgR_downsample, this->imgL_downsample, this->imgDisparity_R);

    this->wls_filter->setLambda(this->lambda);
    this->wls_filter->setSigmaColor(this->sigma);

    this->wls_filter->filter(this->imgDisparity_L, this->undistort_L, this->imgDisparity, this->imgDisparity_R);

    this->imgDisparity.convertTo(this->dmap, CV_32FC3, 1.0/16.0, 0.0);
    this->dmap_valid = dmap(this->valid_disp_roi).clone();
    disparity_map = this->dmap_valid.clone();
}

void StereoCam::getDisparityVisualisation(cv::Mat &disparity_vis, double vis_mult) {
    cv::ximgproc::getDisparityVis(this->imgDisparity, this->disparityVis, vis_mult);
    disparity_vis = this->disparityVis(this->valid_disp_roi).clone();
}

void StereoCam::getUndistortedImages(cv::Mat& undistort_L_crop, cv::Mat& undistort_R_crop) {
    undistort_L_crop = this->undistort_L.clone();
    undistort_R_crop = this->undistort_R.clone();
}

void StereoCam::getPointCloud(cv::Mat &xyz) {
    reprojectImageTo3D(this->dmap_valid, xyz, this->Q, true, CV_32FC3);
}

void StereoCam::getValidImage(cv::Mat &valid_undistort) {
    valid_undistort = this->undistort_L(this->valid_disp_roi).clone();
}

