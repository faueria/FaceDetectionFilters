/*
  Nihal Sandadi
  9/28/25

  Used to help detect faces in a video stream
*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "faceDetect.h"

/*
  cv::Mat& grey : greyscale image which represents the input image to detect faces in
  std::vector<cv::Rect>& faces : reference vector for the box where the faces are detected
  Detects faces in the given frame
*/
int detectFaces(cv::Mat& grey, std::vector<cv::Rect>& faces) {
    static cv::Mat half;
    static cv::CascadeClassifier face_cascade;
    static cv::String face_cascade_file(FACE_CASCADE_FILE);

    if (face_cascade.empty()) {
        if (!face_cascade.load(face_cascade_file)) {
            printf("Unable to load face cascade file: %s\n", FACE_CASCADE_FILE);
            printf("Terminating\n");
            exit(-1);
        }
    }

    faces.clear();
    cv::resize(grey, half, cv::Size(grey.cols / 2, grey.rows / 2));
    cv::equalizeHist(half, half);
    face_cascade.detectMultiScale(half, faces);

    for (int i = 0; i < faces.size(); i++) {
        faces[i].x *= 2;
        faces[i].y *= 2;
        faces[i].width *= 2;
        faces[i].height *= 2;
    }

    return 0;
}

/*
  cv::Mat& frame : image frame which represents the input image
  std::vector<cv::Rect>& faces : vector of all the faces in the picture with coords
  int minWidth : minimum width for an identified face
  float scale : used to adjust the rectangle to image
  draws the boxes directly on the frame
*/
int drawBoxes(cv::Mat& frame, std::vector<cv::Rect>& faces, int minWidth, float scale) {
    cv::Scalar wcolor(170, 120, 110);

    for (int i = 0; i < faces.size(); i++) {
        if (faces[i].width > minWidth) {
            cv::Rect face(faces[i]);
            face.x *= scale;
            face.y *= scale;
            face.width *= scale;
            face.height *= scale;
            cv::rectangle(frame, face, wcolor, 3);
        }
    }

    return 0;
}