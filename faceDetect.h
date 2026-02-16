/*
  Nihal Sandadi
  9/28/25

  header file for faceDetect.cpp, establishes the functions for detecting a face
*/

#ifndef FACEDETECT_H
#define FACEDETECT_H

#include <opencv2/opencv.hpp>

#define FACE_CASCADE_FILE "C:/Users/Nihal Sandadi/Desktop/computer vision/hw1/VideoDisplay/VideoDisplay/haarcascade_frontalface_alt2.xml"

int detectFaces(cv::Mat& grey, std::vector<cv::Rect>& faces);
int drawBoxes(cv::Mat& frame, std::vector<cv::Rect>& faces, int minWidth = 50, float scale = 1.0);

#endif