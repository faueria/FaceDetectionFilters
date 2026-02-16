/*
  Nihal Sandadi
  9/28/25

  Header file for vidDisplay
*/

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <iomanip>
#include "filters.h"
#include "faceDetect.h"
#include "DA2Network.hpp"

cv::Mat applyDisplayMode(cv::Mat& frame, char displayMode, int quantizeLevels, DA2Network*& da_net, 
    bool& depthInitialized, cv::Mat& depthFrame, float depthScaleFactor, std::vector<cv::Rect>& faces);

void processFaceDetection(cv::Mat& displayFrame, std::vector<cv::Rect>& faces, bool faceDetectionEnabled, 
    bool showDistance, char displayMode, bool depthInitialized, const cv::Mat& depthFrame);

void displayModeText(cv::Mat& displayFrame, char displayMode, int quantizeLevels, bool faceDetectionEnabled, 
    bool showDistance, size_t faceCount);

void handleKeyboardInput(char key, char& displayMode, bool& faceDetectionEnabled, bool& showDistance, 
    int& quantizeLevels, int& imageCount, cv::Mat& processedFrame, std::vector<cv::Rect>& faces);