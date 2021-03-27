#include "stereo_cam.h"

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>

#include <iostream>
#include <chrono>
#include <pcl/point_types.h>

int main() {
    pcl::visualization::CloudViewer viewer("justin is retarded");
    cv::VideoCapture capWebcam(1);   // declare a VideoCapture object to associate webcam, 0 means use 1st (default) webcam
    capWebcam.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    capWebcam.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    //capWebcam.set(cv::CAP_PROP_FPS, 10.0);
    //capWebcam.set(cv::CAP_PROP_EXPOSURE, -8);

    if (!capWebcam.isOpened()) {
        // To check if object was associated to webcam successfully
        std::cout << "error: Webcam connect unsuccessful\n"; // if not then print error message
        return(0);            // and exit program
    }

    double vis_mult = 1.0;

    StereoCam stereo_camera("../left.yml", "../right.yml", "../bleh.yml",
                            150, 13, 8000.0, 1.5, 0.8);


    std::cout << "left_camera" << std::endl;
    std::cout << "K1: " << stereo_camera.K1 << std::endl;
    std::cout << "D1: " << stereo_camera.D1 << std::endl << std::endl;
    std::cout << "right_camera" << std::endl;
    std::cout << "K2: " << stereo_camera.K1 << std::endl;
    std::cout << "D2: " << stereo_camera.D1 << std::endl << std::endl;

    cv::Mat img_input, img_L, img_R, undistort_L, undistort_R, undistort_valid;
    cv::Mat disparityVis;

    cv::Mat dmap, xyz;

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

        stereo_camera.process(img_L, img_R, dmap);
        stereo_camera.getDisparityVisualisation(disparityVis, vis_mult);
        stereo_camera.getUndistortedImages(undistort_L, undistort_R);
        stereo_camera.getValidImage(undistort_valid);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        stereo_camera.getPointCloud(xyz);
        pointcloud->width = static_cast<uint32_t>(dmap.cols);
        pointcloud->height = static_cast<uint32_t>(dmap.rows);
        pointcloud->is_dense = false;
        for (int i = 0; i < dmap.rows; ++i) {
            auto* rgb_ptr = undistort_valid.ptr<uchar>(i);
            auto* dmap_ptr = dmap.ptr<uchar>(i);
            auto* xyz_ptr = xyz.ptr<double>(i);

            for (int j = 0; j < dmap.cols; ++j) {
                pcl::PointXYZRGB point;
                uchar d = dmap_ptr[j];
                point.z = 0; point.x = 0; point.y = 0; point.b = 0; point.g = 0; point.r = 0;
                if (d <= 10) { pointcloud->points.push_back(point); continue; }
                //std::cout << (int)d << std::endl;
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
        viewer.showCloud(pointcloud);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "total time: " << duration.count() << "\r" << std::flush;

        cv::rectangle(undistort_L, stereo_camera.valid_roi_L, cv::Scalar(0,255,0), 2, 8, 0);
        cv::rectangle(undistort_R, stereo_camera.valid_roi_R, cv::Scalar(0,255,0), 2, 8, 0);

        cv::namedWindow("imgOriginal", cv::WINDOW_NORMAL);
        cv::namedWindow("imgUndistortL", cv::WINDOW_NORMAL);
        cv::namedWindow("imgUndistortR", cv::WINDOW_NORMAL);
        cv::namedWindow("disparityVis", cv::WINDOW_NORMAL);
        // show windows
        cv::imshow("imgOriginal", img_input);
        cv::imshow("imgUndistortL", undistort_L);
        cv::imshow("imgUndistortR", undistort_R);
        cv::imshow("disparityVis", disparityVis);

        //std::string writePath = "../cloud.ply";
        //pcl::io::savePLYFileBinary(writePath, *pointcloud);

        //break;

        charCheckForEscKey = cv::waitKey(50);        // delay and get key press
    }

    return(0);
}
