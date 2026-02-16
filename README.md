# Project 1: Video Special Effects

To run the program, make sure you add onnxruntime to your dependencies as well as making sure you have the neural network content also installed. You don't need to add any arguments to run and all the files should be in the same file, with perhaps small changes to the paths of various files based on your own files system.

A prerequisite to running the program is that you also need to have a working camera and to run you only need to run the core files.

vidDisplay is in charge of the main loop, handling keyboard inputs, and setting up the appropriate filters.
filters have functions for all the available filters group there.
faceDetect is a separate file to set up the logic around face detection and using the neural network.

da2 is outside code that is an example of the depth filter, use this file to test if the depth filter is working first.

### Controls
'c' - Color mode
'g' - OpenCV Greyscale mode
'h' - Custom Greyscale mode
's' - Sepia Tone mode
'b' - Blur mode
'x' - Sobel X mode (vertical edges)
'y' - Sobel Y mode (horizontal edges)
'm' - Gradient Magnitude mode
'l' - Blur + Quantize mode (quantizeLevels levels)
'z' - Sobel X + Face Detection mode
'k' - Greyscale Keep Green mode
't' - Sepia Tone Faces mode
'f' - Toggle Face Detection
'd' - Depth Estimation mode
'D' - Toggle Distance Display
'+' - Increase quantization levels
'-' - Decrease quantization levels
'S' - Save current frame
'q' - Quit application
'u' - Sepia + Gradient Magnitude mode
'v' - Blur+Quantize + Gradient Magnitude mode
