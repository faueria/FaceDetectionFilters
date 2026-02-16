/*
  Nihal Sandadi
  9/28/25

  List of all the filters avaiable to the user
*/

#include "filters.h"
#include <opencv2/opencv.hpp>
#include <cmath>

/*
  cv::Mat& src : reference image which passes in the unmodified frame
  cv::Mat& dst : reference to the destination output image, where the filter will be stored
  greyscale filter which will convert a color image to a greyscale one according to certain 
  values (0.2R, 0.3G, 0.5B)
*/
int greyscale(cv::Mat& src, cv::Mat& dst) {
    dst.create(src.size(), src.type());

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
            uchar grayValue = static_cast<uchar>(
                0.2 * pixel[2] +  // Red
                0.3 * pixel[1] +  // Green  
                0.5 * pixel[0]    // Blue
                );
            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(grayValue, grayValue, grayValue);
        }
    }
    return 0;
}

/*
  cv::Mat& src : reference image which passes in the unmodified frame
  cv::Mat& dst : reference to the destination output image, where the filter will be stored
  sepia filter which tints color images based on the original values and without mixing 
  new values with old values when calculating
*/
int sepiaTone(cv::Mat& src, cv::Mat& dst) {
    dst.create(src.size(), src.type());

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
            uchar B_original = pixel[0];
            uchar G_original = pixel[1];
            uchar R_original = pixel[2];

            float newB = 0.131 * B_original + 0.534 * G_original + 0.272 * R_original;
            float newG = 0.168 * B_original + 0.686 * G_original + 0.349 * R_original;
            float newR = 0.189 * B_original + 0.769 * G_original + 0.393 * R_original;

            newB = (newB > 255) ? 255 : (newB < 0) ? 0 : newB;
            newG = (newG > 255) ? 255 : (newG < 0) ? 0 : newG;
            newR = (newR > 255) ? 255 : (newR < 0) ? 0 : newR;

            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(newB),
                static_cast<uchar>(newG),
                static_cast<uchar>(newR)
            );
        }
    }
    return 0;
}

/*
  cv::Mat& src : reference image which passes in the unmodified frame
  cv::Mat& dst : reference to the destination output image, where the filter will be stored
  guassian blur filter which efficiently blurs an image according to this gaussian [1, 2, 4, 2, 1]
*/
int blur5x5(cv::Mat& src, cv::Mat& dst) {
    dst.create(src.size(), src.type());
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    int kernel[5] = { 1, 2, 4, 2, 1 };
    int kernelSum = 10;

    // horizontal filter pass
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3i sum(0, 0, 0);

            for (int k = -2; k <= 2; k++) {
                int xk = x + k;
                if (xk < 0) xk = -xk;
                if (xk >= src.cols) xk = 2 * src.cols - xk - 2;

                cv::Vec3b pixel = src.at<cv::Vec3b>(y, xk);
                int weight = kernel[k + 2];

                sum[0] += pixel[0] * weight;
                sum[1] += pixel[1] * weight;
                sum[2] += pixel[2] * weight;
            }

            temp.at<cv::Vec3s>(y, x) = cv::Vec3s(
                static_cast<short>(sum[0]),
                static_cast<short>(sum[1]),
                static_cast<short>(sum[2])
            );
        }
    }

    // vertical filter pass
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3i sum(0, 0, 0);

            for (int k = -2; k <= 2; k++) {
                int yk = y + k;
                if (yk < 0) yk = -yk;
                if (yk >= src.rows) yk = 2 * src.rows - yk - 2;

                cv::Vec3s pixel = temp.at<cv::Vec3s>(yk, x);
                int weight = kernel[k + 2];

                sum[0] += pixel[0] * weight;
                sum[1] += pixel[1] * weight;
                sum[2] += pixel[2] * weight;
            }

            int blue = sum[0] / (kernelSum * kernelSum);
            int green = sum[1] / (kernelSum * kernelSum);
            int red = sum[2] / (kernelSum * kernelSum);

            blue = (blue > 255) ? 255 : (blue < 0) ? 0 : blue;
            green = (green > 255) ? 255 : (green < 0) ? 0 : green;
            red = (red > 255) ? 255 : (red < 0) ? 0 : red;

            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(blue),
                static_cast<uchar>(green),
                static_cast<uchar>(red)
            );
        }
    }

    return 0;
}

/*
  cv::Mat& src : reference image which passes in the unmodified frame
  cv::Mat& dst : reference to the destination output image, where the filter will be stored
  detects vertical edges with vertical smoothing [1, 2, 1] and the horizontal gradient [-1, 0, 1].
  More efficient to break it into two than to do one pass of a 3x3 matrix.
*/
int sobelX3x3(cv::Mat& src, cv::Mat& dst) {
    dst.create(src.size(), CV_16SC3);
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    // vertical smoothing with [1, 2, 1]
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            for (int c = 0; c < 3; c++) {
                short sum = 0;
                for (int k = -1; k <= 1; k++) {
                    int yk = y + k;
                    if (yk < 0) yk = 0;
                    if (yk >= src.rows) yk = src.rows - 1;

                    uchar pixel = src.at<cv::Vec3b>(yk, x)[c];
                    int weight = (k == -1) ? 1 : (k == 0) ? 2 : 1;
                    sum += pixel * weight;
                }
                temp.at<cv::Vec3s>(y, x)[c] = sum;
            }
        }
    }

    // horizontal gradient with [-1, 0, 1]
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            for (int c = 0; c < 3; c++) {
                short sum = 0;
                for (int k = -1; k <= 1; k++) {
                    int xk = x + k;
                    if (xk < 0) xk = 0;
                    if (xk >= src.cols) xk = src.cols - 1;

                    short pixel = temp.at<cv::Vec3s>(y, xk)[c];
                    int weight = (k == -1) ? -1 : (k == 0) ? 0 : 1;
                    sum += pixel * weight;
                }
                dst.at<cv::Vec3s>(y, x)[c] = sum;
            }
        }
    }

    return 0;
}

/*
  cv::Mat& src : reference image which passes in the unmodified frame
  cv::Mat& dst : reference to the destination output image, where the filter will be stored
  detects horizontal edges with horizontal smoothing [1, 2, 1] and the vertical gradient [1, 0, -1].
  More efficient to break it into two than to do one pass of a 3x3 matrix.
*/
int sobelY3x3(cv::Mat& src, cv::Mat& dst) {
    dst.create(src.size(), CV_16SC3);
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    // horizontal smoothing with [1, 2, 1]
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            for (int c = 0; c < 3; c++) {
                short sum = 0;
                for (int k = -1; k <= 1; k++) {
                    int xk = x + k;
                    if (xk < 0) xk = 0;
                    if (xk >= src.cols) xk = src.cols - 1;

                    uchar pixel = src.at<cv::Vec3b>(y, xk)[c];
                    int weight = (k == -1) ? 1 : (k == 0) ? 2 : 1;
                    sum += pixel * weight;
                }
                temp.at<cv::Vec3s>(y, x)[c] = sum;
            }
        }
    }

    // vertical gradient with [1, 0, -1]
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            for (int c = 0; c < 3; c++) {
                short sum = 0;
                for (int k = -1; k <= 1; k++) {
                    int yk = y + k;
                    if (yk < 0) yk = 0;
                    if (yk >= src.rows) yk = src.rows - 1;

                    short pixel = temp.at<cv::Vec3s>(yk, x)[c];
                    int weight = (k == -1) ? 1 : (k == 0) ? 0 : -1;
                    sum += pixel * weight;
                }
                dst.at<cv::Vec3s>(y, x)[c] = sum;
            }
        }
    }

    return 0;
}

/*
  cv::Mat& sx : the vertical edges from sobel x
  cv::Mat& sy : the horizontal edges from sobel y
  cv::Mat& dst : the output image with the combined edge strength
  combines a sobel x and a sobel y to calculate the total edge strength
  and get the gradient magnitude
*/
int gradientMagnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
    if (sx.size() != sy.size() || sx.type() != sy.type()) {
        std::cout << "Error: Sobel images must have same size and type" << std::endl;
        return -1;
    }

    if (sx.type() != CV_16SC3) {
        std::cout << "Error: Input images must be CV_16SC3 type" << std::endl;
        return -1;
    }

    dst.create(sx.size(), CV_8UC3);

    for (int y = 0; y < sx.rows; y++) {
        for (int x = 0; x < sx.cols; x++) {
            cv::Vec3b magnitudePixel;

            for (int c = 0; c < 3; c++) {
                short gx = sx.at<cv::Vec3s>(y, x)[c];
                short gy = sy.at<cv::Vec3s>(y, x)[c];

                double mag = std::sqrt(static_cast<double>(gx) * gx + static_cast<double>(gy) * gy);
                uchar displayValue = static_cast<uchar>(std::min(255.0, mag / 4.0));

                magnitudePixel[c] = displayValue;
            }

            dst.at<cv::Vec3b>(y, x) = magnitudePixel;
        }
    }

    return 0;
}

/*
  cv::Mat& src : reference image which passes in the unmodified frame
  cv::Mat& dst : reference to the destination output image, where the filter will be stored
  int levels : the number of color values we want per channel
  this smooths out an image with a guassian blur and then reduices the color to a limited 
  number of levels, which give a more cartoon effect
*/
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels) {
    if (levels <= 0) {
        std::cout << "Error: Levels must be positive" << std::endl;
        return -1;
    }

    if (levels == 1) {
        return blur5x5(src, dst);
    }

    cv::Mat blurred;
    blur5x5(src, blurred);

    dst.create(src.size(), src.type());

    float bucketSize = 255.0f / static_cast<float>(levels);

    for (int y = 0; y < blurred.rows; y++) {
        for (int x = 0; x < blurred.cols; x++) {
            cv::Vec3b pixel = blurred.at<cv::Vec3b>(y, x);
            cv::Vec3b quantizedPixel;

            for (int c = 0; c < 3; c++) {
                int bucketIndex = static_cast<int>(pixel[c] / bucketSize);
                bucketIndex = std::max(0, std::min(levels - 1, bucketIndex));
                quantizedPixel[c] = static_cast<uchar>(bucketIndex * bucketSize);
            }

            dst.at<cv::Vec3b>(y, x) = quantizedPixel;
        }
    }

    return 0;
}

/*
  cv::Mat& src : reference image which passes in the unmodified frame
  cv::Mat& dst : reference to the destination output image, where the filter will be stored
  Similar to grey scale function earlier but preserves the color of green values that are 
  bigger than red and blue by more than 20 values and the green value is bigger than 50
*/
int greyscaleKeepGreen(cv::Mat& src, cv::Mat& dst) {
    dst.create(src.size(), src.type());

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
            uchar B = pixel[0];
            uchar G = pixel[1];
            uchar R = pixel[2];

            uchar grayValue = static_cast<uchar>(0.2 * R + 0.3 * G + 0.5 * B);

            bool isGreen = (G > R && G > B) && (G > 50) && (G > R + 20 || G > B + 20);

            if (isGreen) {
                dst.at<cv::Vec3b>(y, x) = pixel;
            }
            else {
                dst.at<cv::Vec3b>(y, x) = cv::Vec3b(grayValue, grayValue, grayValue);
            }
        }
    }
    return 0;
}

/*
  cv::Mat& src : reference image which passes in the unmodified frame
  cv::Mat& dst : reference to the destination output image, where the filter will be stored
  std::vector<cv::Rect>& faces : vector of the coords of all the faces
  for all the face coords, apply the sepia tones to the bounds but leave the rest of the
  image untouched
*/
int sepiaToneFaces(cv::Mat& src, cv::Mat& dst, std::vector<cv::Rect>& faces) {
    src.copyTo(dst);

    for (const auto& face : faces) {
        cv::Rect safeFace = face & cv::Rect(0, 0, src.cols, src.rows);

        if (safeFace.area() > 0) {
                        cv::Mat faceRegion = dst(safeFace);

            cv::Mat sepiaFace;
            sepiaTone(faceRegion, sepiaFace);

            sepiaFace.copyTo(faceRegion);

            cv::rectangle(dst, safeFace, cv::Scalar(100, 80, 60), 2);
        }
    }

    return 0;
}

/*
  cv::Mat& src : reference image which passes in the unmodified frame
  cv::Mat& dst : reference to the destination output image, where the filter will be stored
  std::vector<cv::Rect>& faces : vector of the coords of all the faces
  sets up sobel x edge detection everywhere except the face, keeping the face box untouched
*/
int sobelXExceptFaces(cv::Mat& src, cv::Mat& dst, std::vector<cv::Rect>& faces) {
    cv::Mat sobelXResult, sobelDisplay;

    sobelX3x3(src, sobelXResult);
    cv::convertScaleAbs(sobelXResult, sobelDisplay);

    src.copyTo(dst);

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            bool isInFace = false;
            cv::Point pt(x, y);

            for (const auto& face : faces) {
                if (face.contains(pt)) {
                    isInFace = true;
                    break;
                }
            }

            if (!isInFace) {
                dst.at<cv::Vec3b>(y, x) = sobelDisplay.at<cv::Vec3b>(y, x);
            }
        }
    }

    for (const auto& face : faces) {
        cv::rectangle(dst, face, cv::Scalar(0, 255, 0), 2);
    }

    return 0;
}

/*
  cv::Mat& src : reference image which passes in the unmodified frame
  cv::Mat& dst : reference to the destination output image, where the filter will be stored

  This filter applies sepia tone first, then calculates gradient magnitude on the sepia result.
  This creates a vintage image with edge detection applied to the sepia-toned version.
*/
int sepiaGradientMagnitude(cv::Mat& src, cv::Mat& dst) {
    // Apply sepia tone first
    cv::Mat sepiaResult;
    sepiaTone(src, sepiaResult);

    // Calculate gradient magnitude ON THE SEPIA RESULT
    cv::Mat sobelXResult, sobelYResult, magnitudeResult;
    sobelX3x3(sepiaResult, sobelXResult);
    sobelY3x3(sepiaResult, sobelYResult);
    gradientMagnitude(sobelXResult, sobelYResult, magnitudeResult);

    // Use the gradient magnitude result directly
    dst = magnitudeResult;

    return 0;
}

/*
  cv::Mat& src : reference image which passes in the unmodified frame
  cv::Mat& dst : reference to the destination output image, where the filter will be stored
  int levels : the number of color values we want per channel

  This filter applies blur+quantize first, then calculates gradient magnitude on that result.
  This creates edge detection applied to the cartoonized/blur-quantized image.
*/
int blurQuantizeGradient(cv::Mat& src, cv::Mat& dst, int levels) {
    if (levels <= 0) {
        std::cout << "Error: Levels must be positive" << std::endl;
        return -1;
    }

    // Apply blur and quantize first
    cv::Mat blurQuantResult;
    blurQuantize(src, blurQuantResult, levels);

    // Calculate gradient magnitude ON THE BLUR+QUANTIZE RESULT
    cv::Mat sobelXResult, sobelYResult, magnitudeResult;
    sobelX3x3(blurQuantResult, sobelXResult);
    sobelY3x3(blurQuantResult, sobelYResult);
    gradientMagnitude(sobelXResult, sobelYResult, magnitudeResult);

    // Use the gradient magnitude result directly
    dst = magnitudeResult;

    return 0;
}