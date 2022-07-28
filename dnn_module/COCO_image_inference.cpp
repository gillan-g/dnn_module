#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

using namespace std;
using namespace cv;
using namespace dnn;



int main()
{
    
    vector<string> class_names;
    vector<Scalar> colors;
    ifstream ifs(string("Resources/object_detection_classes_coco.txt").c_str());
    string line;
    int B;
    int G;
    int R;

    while (getline(ifs, line))
    {
        class_names.push_back(line);
        B = rand() % 256;
        G = rand() % 256;
        R = rand() % 256;
        colors.push_back(Scalar(B, G, R));
        cout << line << endl;
    }

    // load the neural network model
    auto model = readNet("Resources/frozen_inference_graph.pb",
                         "Resources/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt", 
                         "TensorFlow");


    // read the image from disk
    Mat image; // = imread("Resources/image_2.jpg");
    VideoCapture cap(0);

    while (true) {
        cap.read(image);


        int image_height = image.cols;
        int image_width = image.rows;

        //create blob from image
        Mat blob = blobFromImage(image, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false);

        //create blob from image
        model.setInput(blob);

        //forward pass through the model to carry out the detection
        Mat output = model.forward();
        Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

        // loop over detection
        for (int i = 0; i < detectionMat.rows; i++) {
            int class_id = detectionMat.at<float>(i, 1);
            float confidence = detectionMat.at<float>(i, 2);

            // Check if the detection is of good quality
            if (confidence > 0.4) {
                int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
                int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
                int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols - box_x);
                int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows - box_y);
                rectangle(image, Point(box_x, box_y), Point(box_x + box_width, box_y + box_height), colors[i], 2);
                putText(image, class_names[class_id - 1].c_str(), Point(box_x, box_y - 5), FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1);
            }
        }

        imshow("image", image);
        //imwrite("image_result.jpg", image);
        char key = (char)waitKey(1);   // explicit cast
        if (key == 27) break;
    }

    
    return 0;
}