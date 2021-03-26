#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include <iostream>
#include <chrono>
#include <pcl/point_types.h>

void getCameraData(std::string left_file, std::string right_file, std::string info_file,
                   cv::Mat& K1, cv::Mat& D1,
                   cv::Mat& K2, cv::Mat& D2,
                   cv::Mat& R, cv::Vec3d& T) {
    cv::FileStorage leftFs(left_file, cv::FileStorage::READ);
    cv::FileStorage rightFs(right_file, cv::FileStorage::READ);
    cv::FileStorage infoFs(info_file, cv::FileStorage::READ);

    leftFs["camera_matrix"] >> K1;
    leftFs["distortion_coefficients"] >> D1;

    rightFs["camera_matrix"] >> K2;
    rightFs["distortion_coefficients"] >> D2;

    infoFs["R"] >> R;
    infoFs["T"] >> T;

    leftFs.release();
    rightFs.release();
    infoFs.release();
}

int main() {
    //pcl::visualization::CloudViewer viewer("justin is retarded");
    cv::VideoCapture capWebcam(1);   // declare a VideoCapture object to associate webcam, 0 means use 1st (default) webcam

    if (!capWebcam.isOpened())  //  To check if object was associated to webcam successfully
    {
        std::cout << "error: Webcam connect unsuccessful\n"; // if not then print error message
        return(0);            // and exit program
    }

    int max_disp = 300;
    int wsize = 15;
    double lambda = 8000.0;
    double sigma = 1.5;
    double vis_mult = 1.0;
    double downsample = 0.8;

    max_disp/=2;
    if(max_disp%16!=0)
        max_disp += 16-(max_disp%16);

    cv::Mat img_input;        // input image
    cv::Mat img_L, undistort_L, undistort_L_crop;
    cv::Mat img_R, undistort_R, undistort_R_crop;
    cv::Mat imgDisparity_L;
    cv::Mat imgDisparity_R;

    cv::Mat imgDisparity;
    cv::Mat disparityVis;

    cv::Mat dmap;

    cv::Mat K1, D1, R1, P1;
    cv::Mat K2, D2, R2, P2;
    cv::Mat Q, R;
    cv::Vec3d T;
    cv::Rect valid_roi_L, valid_roi_R;
    bool Q_calculated = false;

    getCameraData("../left.yml", "../right.yml", "../bleh.yml", K1, D1, K2, D2, R, T);

    std::cout << "left_camera" << std::endl;
    std::cout << "K1: " << K1 << std::endl;
    std::cout << "D1: " << D1 << std::endl << std::endl;
    std::cout << "right_camera" << std::endl;
    std::cout << "K2: " << K1 << std::endl;
    std::cout << "D2: " << D1 << std::endl << std::endl;

    cv::Mat imgL_downsample;
    cv::Mat imgR_downsample;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;

    cv::Ptr<cv::StereoBM> left_matcher = cv::StereoBM::create(max_disp, wsize);
    wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
    cv::Ptr<cv::StereoMatcher> right_matcher;

    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);

    char charCheckForEscKey = 0;

    while (charCheckForEscKey != 27 && capWebcam.isOpened()) {// until the Esc key is pressed or webcam connection is lost

        bool blnFrameReadSuccessfully = capWebcam.read(img_input);   // get next frame

        if (!blnFrameReadSuccessfully || img_input.empty()) {    // if frame read unsuccessfully
            std::cout << "error: frame can't read \n";      // print error message
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();
        cv::Rect leftROI(0, 0, img_input.cols / 2, img_input.rows);
        cv::Rect rightROI(leftROI.width, 0, img_input.cols - leftROI.width, img_input.rows);
        img_L = img_input(leftROI).clone();
        img_R = img_input(rightROI).clone();

        cv::undistort(img_L, undistort_L, K1, D1);
        cv::undistort(img_R, undistort_R, K2, D2);

        if (!Q_calculated) {
            cv::Size imageSize(img_L.cols, img_L.rows);
            stereoRectify(K1, D1, K2, D2, imageSize, R, T, R1, R2, P1, P2, Q, cv::CALIB_SAME_FOCAL_LENGTH, -1, cv::Size(), &valid_roi_L, &valid_roi_R);
            left_matcher->setROI1(valid_roi_L); left_matcher->setROI2(valid_roi_R);
            right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
            Q_calculated = true;
        }

        cv::resize(undistort_L, imgL_downsample, cv::Size(), downsample, downsample, cv::INTER_LINEAR_EXACT);
        cv::resize(undistort_R, imgR_downsample, cv::Size(), downsample, downsample, cv::INTER_LINEAR_EXACT);

        cv::cvtColor(imgL_downsample,  imgL_downsample,  cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgR_downsample, imgR_downsample, cv::COLOR_BGR2GRAY);

        left_matcher->compute(imgL_downsample, imgR_downsample, imgDisparity_L);
        right_matcher->compute(imgR_downsample, imgL_downsample, imgDisparity_R);

        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);

        wls_filter->filter(imgDisparity_L, undistort_L, imgDisparity, imgDisparity_R);

        cv::ximgproc::getDisparityVis(imgDisparity, disparityVis, vis_mult);

        imgDisparity.convertTo(dmap, CV_32F, 1.0/16.0, 0.0);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new   pcl::PointCloud<pcl::PointXYZRGB>());
        cv::Mat xyz;
        reprojectImageTo3D(dmap, xyz, Q, false, CV_32F);
        pointcloud->width = static_cast<uint32_t>(dmap.cols);
        pointcloud->height = static_cast<uint32_t>(dmap.rows);
        pointcloud->is_dense = false;
        pcl::PointXYZRGB point;
        for (int i = 0; i < dmap.rows; ++i) {
            auto* rgb_ptr = undistort_L.ptr<uchar>(i);
            auto* dmap_ptr = dmap.ptr<uchar>(i);
            auto* xyz_ptr = xyz.ptr<double>(i);

            for (int j = 0; j < dmap.cols; ++j)
            {
                uchar d = dmap_ptr[j];
                if (d == 0) continue;
                cv::Point3f p = xyz.at<cv::Point3f>(i, j);

                point.z = p.z;   // I have also tried p.z/16
                point.x = p.x;
                point.y = p.y;

                point.b = rgb_ptr[3 * j];
                point.g = rgb_ptr[3 * j + 1];
                point.r = rgb_ptr[3 * j + 2];
                pointcloud->points.push_back(point);
            }
        }
        //viewer.showCloud(pointcloud);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "time: " << duration.count() << "\r" << std::flush;

        undistort_L_crop = undistort_L.clone(); cv::rectangle(undistort_L_crop, valid_roi_L, cv::Scalar(0,255,0), 1, 8, 0);
        undistort_R_crop = undistort_R.clone(); cv::rectangle(undistort_R_crop, valid_roi_R, cv::Scalar(0,255,0), 1, 8, 0);

        cv::namedWindow("imgOriginal", cv::WINDOW_NORMAL);
        cv::namedWindow("imgUndistortL", cv::WINDOW_NORMAL);
        cv::namedWindow("imgUndistortR", cv::WINDOW_NORMAL);
        cv::namedWindow("disparityVis", cv::WINDOW_NORMAL);
        // show windows
        cv::imshow("imgOriginal", img_input);
        cv::imshow("imgUndistortL", undistort_L_crop);
        cv::imshow("imgUndistortR", undistort_R_crop);
        cv::imshow("disparityVis", disparityVis);

        charCheckForEscKey = cv::waitKey(50);        // delay and get key press
    }

    return(0);
}
