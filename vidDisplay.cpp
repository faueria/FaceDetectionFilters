/*
  Nihal Sandadi
  9/28/25

  Main user loop/file which handles user inputs and video streaming
*/

#include "vidDisplay.h"

/*
  int argc : number of command line arguments
  char* argv[] : array of command line arguments

  starts the video streaming and dictates what effects, based on user input, are applied 
  to the current video. Actively tracks the current state of the video plus its filters.
*/
int main(int argc, char* argv[]) {
    // initialize all the variables
    cv::VideoCapture* capdev;

    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    cv::Size refS(static_cast<int>(capdev->get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(capdev->get(cv::CAP_PROP_FRAME_HEIGHT)));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1);
    cv::Mat frame;
    cv::Mat processedFrame;
    cv::Mat displayFrame;

    cv::Mat depthFrame;
    DA2Network* da_net = nullptr;
    bool depthInitialized = false;
    const float depthReduction = 0.5;
    float depthScaleFactor = 256.0f / (refS.height * depthReduction);

    std::vector<cv::Rect> faces;
    bool faceDetectionEnabled = false;
    bool showDistance = false;

    int imageCount = 0;
    char displayMode = 'c';
    int quantizeLevels = 10;

    // user instructions on what to do and how to switch modes
    printf("Controls:\n");
    printf("  'c' - Color mode\n");
    printf("  'g' - OpenCV Greyscale mode\n");
    printf("  'h' - Custom Greyscale mode\n");
    printf("  's' - Sepia Tone mode\n");
    printf("  'b' - Blur mode\n");
    printf("  'x' - Sobel X mode (vertical edges)\n");
    printf("  'y' - Sobel Y mode (horizontal edges)\n");
    printf("  'm' - Gradient Magnitude mode\n");
    printf("  'l' - Blur + Quantize mode (%d levels)\n", quantizeLevels);
    printf("  'z' - Sobel X + Face Detection mode\n");
    printf("  'k' - Greyscale Keep Green mode\n");
    printf("  't' - Sepia Tone Faces mode\n");
    printf("  'f' - Toggle Face Detection\n");
    printf("  'd' - Depth Estimation mode\n");
    printf("  'D' - Toggle Distance Display\n");
    printf("  '+' - Increase quantization levels\n");
    printf("  '-' - Decrease quantization levels\n");
    printf("  'S' - Save current frame\n");
    printf("  'q' - Quit application\n");
    printf("  'u' - Sepia + Gradient Magnitude mode\n");
    printf("  'v' - Blur+Quantize + Gradient Magnitude mode\n");

    // Main loop
    for (;;) {
        *capdev >> frame;
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        displayFrame = applyDisplayMode(frame, displayMode, quantizeLevels,
            da_net, depthInitialized, depthFrame,
            depthScaleFactor, faces);

        displayFrame.copyTo(processedFrame);

        processFaceDetection(displayFrame, faces, faceDetectionEnabled,
            showDistance, displayMode, depthInitialized, depthFrame);

        displayModeText(displayFrame, displayMode, quantizeLevels,
            faceDetectionEnabled, showDistance, faces.size());

        cv::imshow("Video", displayFrame);

        // we checkk for the key input here
        char key = cv::waitKey(10);
        if (key == 'q') break;

        handleKeyboardInput(key, displayMode, faceDetectionEnabled, showDistance,
            quantizeLevels, imageCount, processedFrame, faces);
    }

    // Clean up variables here
    if (da_net != nullptr) {
        delete da_net;
    }
    delete capdev;
    return 0;
}

/*
  cv::Mat& frame : the current frame of the video in BGR format
  char displayMode : what display mode is currently active
  int quantizeLevels : the number of color values we want per channel
  this smooths out an image with a guassian blur and then reduices 
  the color to a limited number of levels, which give a more cartoon effect
  DA2Network*& da_net : the reference pointer to depth estimation neural network
  bool& depthInitialized : reference tracking to see if depth network is loaded
  cv::Mat& depthFrame : output storage for only depth estimation results
  float depthScaleFactor : also for depth network, scaling factor for it
  std::vector<cv::Rect>& faces : coords of detected faces
  cv::Mat : output type, returns the processed frame

  main calls this function to determine what filters should be applied to 
  the current frame based on the display mode
*/
cv::Mat applyDisplayMode(cv::Mat& frame, char displayMode, int quantizeLevels,
    DA2Network*& da_net, bool& depthInitialized,
    cv::Mat& depthFrame, float depthScaleFactor,
    std::vector<cv::Rect>& faces) {
    cv::Mat result;
    cv::Mat sobelXResult, sobelYResult, magnitudeResult;
    cv::Mat depthInput, depthVis;

    switch (displayMode) {
    case 'g': {
        cv::cvtColor(frame, result, cv::COLOR_BGR2GRAY);
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
        break;
    }

    case 'h': {
        greyscale(frame, result);
        break;
    }

    case 's': {
        sepiaTone(frame, result);
        break;
    }

    case 'b': {
        blur5x5(frame, result);
        break;
    }

    case 'x': {
        sobelX3x3(frame, sobelXResult);
        cv::convertScaleAbs(sobelXResult, result);
        break;
    }

    case 'y': {
        sobelY3x3(frame, sobelYResult);
        cv::convertScaleAbs(sobelYResult, result);
        break;
    }

    case 'm': {
        sobelX3x3(frame, sobelXResult);
        sobelY3x3(frame, sobelYResult);
        gradientMagnitude(sobelXResult, sobelYResult, magnitudeResult);
        result = magnitudeResult;
        break;
    }

    case 'l': {
        blurQuantize(frame, result, quantizeLevels);
        break;
    }

    case 'z': {
        result = frame.clone();
        break;
    }

    case 'k': {
        greyscaleKeepGreen(frame, result);
        break;
    }

    case 't': {
        result = frame.clone();
        break;
    }
    case 'u': {
        sepiaGradientMagnitude(frame, result);
        break;
    }

    case 'v': {
        blurQuantizeGradient(frame, result, quantizeLevels);
        break;
    }

    case 'd': {
        if (!depthInitialized) {
            try {
                da_net = new DA2Network("model_fp16.onnx");
                depthInitialized = true;
                printf("Depth estimation initialized\n");
            }
            catch (const std::exception& e) {
                printf("Failed to initialize depth estimation: %s\n", e.what());
                printf("Make sure model_fp16.onnx is in the executable directory\n");
                result = frame.clone();
                break;
            }
        }

        if (depthInitialized && da_net != nullptr) {
            cv::resize(frame, depthInput, cv::Size(), 0.5, 0.5);
            da_net->set_input(depthInput, depthScaleFactor);
            da_net->run_network(depthFrame, depthInput.size());
            cv::applyColorMap(depthFrame, depthVis, cv::COLORMAP_INFERNO);
            cv::resize(depthVis, result, frame.size());
        }
        else {
            result = frame.clone();
        }
        break;
    }

    default: {
        result = frame.clone();
        break;
    }
    }

    return result;
}

/*
  cv::Mat& displayFrame : the current frame of the video by reference
  std::vector<cv::Rect>& faces : vector coords of detected faces
  bool faceDetectionEnabled : face detection flag
  bool showDistance : show distance to a face/faces flag
  char displayMode : the current display mode
  bool depthInitialized : flag to see if depth network is loaded
  const cv::Mat& depthFrame : this is the depth map from the ai approximation

  this function specifically handles face detection and face based filters, 
  along with the distance measurement display
*/
void processFaceDetection(cv::Mat& displayFrame, std::vector<cv::Rect>& faces,
    bool faceDetectionEnabled, bool showDistance, char displayMode, 
    bool depthInitialized, const cv::Mat& depthFrame) {
    if (!faceDetectionEnabled) return;

    cv::Mat grey;
    cv::cvtColor(displayFrame, grey, cv::COLOR_BGR2GRAY);
    detectFaces(grey, faces);

    if (displayMode == 'z' && !faces.empty()) {
        sobelXExceptFaces(displayFrame, displayFrame, faces);
    }
    else if (displayMode == 't' && !faces.empty()) {
        sepiaToneFaces(displayFrame, displayFrame, faces);
    }
    else {
        drawBoxes(displayFrame, faces);
    }

    if (showDistance && !faces.empty()) {
        for (int i = 0; i < faces.size(); i++) {
            float estimatedDistance = 0.0f;

            if (displayMode == 'd' && depthInitialized && !depthFrame.empty()) {
                cv::Point faceCenter(faces[i].x + faces[i].width / 2,
                    faces[i].y + faces[i].height / 2);

                float scaleX = (float)depthFrame.cols / displayFrame.cols;
                float scaleY = (float)depthFrame.rows / displayFrame.rows;
                cv::Point depthCenter(faceCenter.x * scaleX, faceCenter.y * scaleY);

                depthCenter.x = std::max(0, std::min(depthFrame.cols - 1, depthCenter.x));
                depthCenter.y = std::max(0, std::min(depthFrame.rows - 1, depthCenter.y));

                uchar depthValue = depthFrame.at<uchar>(depthCenter);
                float normalizedDepth = 1.0f - (depthValue / 255.0f);
                estimatedDistance = 0.3f + normalizedDepth * 3.0f;
            }
            else {
                // Fall back to size-based estimation
                float faceArea = faces[i].width * faces[i].height;
                float frameArea = displayFrame.cols * displayFrame.rows;
                float relativeSize = faceArea / frameArea;
                estimatedDistance = 0.8f / sqrt(relativeSize);
            }

            estimatedDistance = std::max(0.3f, std::min(5.0f, estimatedDistance));

            std::stringstream distanceStream;
            distanceStream << std::fixed << std::setprecision(1) << estimatedDistance << "m";

            if (displayMode == 'd' && depthInitialized) {
                distanceStream << " (depth)";
            }
            else {
                distanceStream << " (size)";
            }

            std::string distanceText = distanceStream.str();
            cv::Point textPos(faces[i].x, faces[i].y - 10);
            cv::Scalar textColor = (displayMode == 'd' && depthInitialized)
                ? cv::Scalar(0, 255, 0)  // Green for depth-based
                : cv::Scalar(0, 255, 255); // Yellow for size-based

            cv::putText(displayFrame, distanceText, textPos,
                cv::FONT_HERSHEY_SIMPLEX, 0.5, textColor, 2);
        }
    }
}

/*
  cv::Mat& displayFrame : the reference video for the current frame
  char displayMode : the current active filter/mode
  int quantizeLevels : the current quantization level for the blur +
                       quantize filter
  bool faceDetectionEnabled : flag for face detection 
  bool showDistance : flag if we need to show distance
  size_t faceCount : number of currently detected faces

  this function shows the user the current display mode
*/
void displayModeText(cv::Mat& displayFrame, char displayMode, int quantizeLevels,
    bool faceDetectionEnabled, bool showDistance, size_t faceCount) {
    std::string modeText;
    cv::Scalar textColor;

    // Set mode text and color based on current display mode
    switch (displayMode) {
    case 'g': modeText = "OPENCV GREYSCALE"; textColor = cv::Scalar(200, 200, 200); break;
    case 'h': modeText = "CUSTOM GREYSCALE"; textColor = cv::Scalar(150, 150, 255); break;
    case 's': modeText = "SEPIA TONE"; textColor = cv::Scalar(30, 100, 180); break;
    case 'b': modeText = "BLUR"; textColor = cv::Scalar(200, 100, 50); break;
    case 'x': modeText = "SOBEL X (VERTICAL EDGES)"; textColor = cv::Scalar(0, 255, 0); break;
    case 'y': modeText = "SOBEL Y (HORIZONTAL EDGES)"; textColor = cv::Scalar(0, 0, 255); break;
    case 'm': modeText = "GRADIENT MAGNITUDE"; textColor = cv::Scalar(255, 255, 0); break;
    case 'l': modeText = "BLUR + QUANTIZE (" + std::to_string(quantizeLevels) + " levels)";
        textColor = cv::Scalar(255, 100, 200); break;
    case 'z': modeText = "SOBEL X + FACE DETECTION"; textColor = cv::Scalar(255, 150, 0); break;
    case 'k': modeText = "GREYSCALE KEEP GREEN"; textColor = cv::Scalar(0, 255, 0); break;
    case 't': modeText = "SEPIA TONE FACES"; textColor = cv::Scalar(150, 100, 50); break;
    case 'd': modeText = "DEPTH ESTIMATION"; textColor = cv::Scalar(255, 0, 255); break;
    default: modeText = "COLOR"; textColor = cv::Scalar(0, 255, 255); break;
    case 'u': modeText = "SEPIA + GRADIENT MAGNITUDE"; textColor = cv::Scalar(180, 120, 80); break;
    case 'v': modeText = "BLUR+QUANTIZE + GRADIENT MAGNITUDE (" + std::to_string(quantizeLevels) + " levels)"; 
        textColor = cv::Scalar(200, 150, 255); break;
    }

    if (faceDetectionEnabled) {
        modeText += " + FACE DETECTION";
        if (showDistance) {
            modeText += " + DISTANCE";
            std::string method = (displayMode == 'd') ? " (depth-based)" : " (size-based)";
            modeText += method;
        }

        cv::putText(displayFrame, "Faces: " + std::to_string(faceCount),
            cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(0, 255, 255), 2);
    }

    cv::putText(displayFrame, modeText, cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2);
}

/*
  char key : they keyboard press by the user
  char& displayMode : the current display mode (by reference)
  bool& faceDetectionEnabled : flag for face detection
  bool& showDistance : flag if we need to calculate distance
  int& quantizeLevels : the amount of levels for quantize
  int& imageCount : the counter for the amount of saved images
  cv::Mat& processedFrame : reference to the current processed frame
  std::vector<cv::Rect>& faces : vector for coords of all faces

  this function handles the keyboard input and updates the application state
*/
// Handle keyboard input
void handleKeyboardInput(char key, char& displayMode, bool& faceDetectionEnabled,
    bool& showDistance, int& quantizeLevels, int& imageCount,
    cv::Mat& processedFrame, std::vector<cv::Rect>& faces) {
    switch (key) {
    case 'S': { // Save frame
        std::string filename = "image_" + std::to_string(imageCount) + ".jpg";
        bool success = cv::imwrite(filename, processedFrame);
        if (success) {
            printf("Image saved as: %s\n", filename.c_str());
            imageCount++;
        }
        else {
            printf("Failed to save image!\n");
        }
        break;
    }

    case 'g': displayMode = 'g'; printf("Switched to OpenCV Greyscale mode\n"); break;
    case 'h': displayMode = 'h'; printf("Switched to Custom Greyscale mode\n"); break;
    case 's': displayMode = 's'; printf("Switched to Sepia Tone mode\n"); break;
    case 'b': displayMode = 'b'; printf("Switched to Blur mode\n"); break;
    case 'x': displayMode = 'x'; printf("Switched to Sobel X mode\n"); break;
    case 'y': displayMode = 'y'; printf("Switched to Sobel Y mode\n"); break;
    case 'm': displayMode = 'm'; printf("Switched to Gradient Magnitude mode\n"); break;
    case 'l': displayMode = 'l'; printf("Switched to Blur + Quantize mode (%d levels)\n", quantizeLevels); break;
    case 'z': displayMode = 'z'; printf("Switched to Sobel X + Face Detection mode\n"); break;
    case 'k': displayMode = 'k'; printf("Switched to Greyscale Keep Green mode\n"); break;
    case 't': displayMode = 't'; printf("Switched to Sepia Tone Faces mode\n"); break;
    case 'u': displayMode = 'u'; printf("Switched to Sepia + Gradient Magnitude mode\n"); break;
    case 'v': displayMode = 'v'; printf("Switched to Blur+Quantize + Gradient Magnitude mode\n"); break;
    case 'f':
        faceDetectionEnabled = !faceDetectionEnabled;
        printf("Face detection %s\n", faceDetectionEnabled ? "ENABLED" : "DISABLED");
        if (!faceDetectionEnabled) faces.clear();
        break;
    case 'd': displayMode = 'd'; printf("Switched to Depth Estimation mode\n"); break;
    case 'D':
        showDistance = !showDistance;
        printf("Distance display %s\n", showDistance ? "ENABLED" : "DISABLED");
        break;
    case '+':
        if (displayMode == 'l') {
            quantizeLevels = std::min(64, quantizeLevels + 1);
            printf("Increased quantization levels to: %d\n", quantizeLevels);
        }
        break;
    case '-':
        if (displayMode == 'l') {
            quantizeLevels = std::max(2, quantizeLevels - 1);
            printf("Decreased quantization levels to: %d\n", quantizeLevels);
        }
        break;
    case 'c': displayMode = 'c'; printf("Switched to Color mode\n"); break;
    }
}