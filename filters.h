/*
  Nihal Sandadi
  9/28/25

  List/Definition of all the filter functions
*/
#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>
#include <vector>

int greyscale(cv::Mat& src, cv::Mat& dst);
int sepiaTone(cv::Mat& src, cv::Mat& dst);
int blur5x5(cv::Mat& src, cv::Mat& dst);
int sobelX3x3(cv::Mat& src, cv::Mat& dst);
int sobelY3x3(cv::Mat& src, cv::Mat& dst);
int gradientMagnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels);
int greyscaleKeepGreen(cv::Mat& src, cv::Mat& dst);
int sepiaToneFaces(cv::Mat& src, cv::Mat& dst, std::vector<cv::Rect>& faces);
int sobelXExceptFaces(cv::Mat& src, cv::Mat& dst, std::vector<cv::Rect>& faces);
int sepiaGradientMagnitude(cv::Mat& src, cv::Mat& dst);
int blurQuantizeGradient(cv::Mat& src, cv::Mat& dst, int levels);

#endif