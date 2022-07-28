#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <math.h>       /* exp */
#define LOG(x) std::cout<<x<<endl;

using namespace std;
using namespace cv;
using namespace dnn;

// Split line, return 1st string
string ssplit(string s, string delimiter) 
{
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res[0];
}


vector<double> my_softmax(Mat & inMat)
{
    
    float summed = sum(inMat)[0];
    float nom = 0;
    float denom = exp(summed);
    vector<double> sm;

    for (int row = 0; row < inMat.rows; row++)
    {
        for (int col = 0; col < inMat.cols; col++)

        {
            nom = exp(inMat.at<float>(row, col));
            sm.push_back((double)nom / denom);
        }        
    }   
    return sm;
}

int main()
{
    std::vector<std::string> class_names;
    ifstream ifs(string("Resources/classification_classes_ILSVRC2012.txt").c_str());
    string line, token;
   
    while (getline(ifs, line))
    {
        token = ssplit(line, ",");
        class_names.push_back(token);
        //cout << token << endl;

    }

    // load the neural network model
    auto model = readNet("Resources/DenseNet_121.prototxt",
                         "Resources/DenseNet_121.caffemodel",
                         "Caffe");
    // read image
    Mat image = imread("Resources/image_1.jpg");

    //convert to necessary format
    Mat blob = blobFromImage(image, 0.01, Size(224, 224), Scalar(104, 117, 123));
    LOG(blob.size());

    // set the input blob for the neural network
    model.setInput(blob);

    // forward pass the image blob through the model
    Mat outputs = model.forward();


    // convert results array to softmax
    Mat softmax;
    exp(outputs.reshape(1, 1), softmax);
    softmax /= sum(softmax)[0];

    // calc softmax maximum
    Point classIdPoint;
    double final_prob;
    minMaxLoc(softmax, NULL, &final_prob, NULL, &classIdPoint);
    int label_id = classIdPoint.x;

    // Print predicted class.
    string out_text = format("%s, %.3f", (class_names[label_id].c_str()), final_prob);
    
    // put the class name text on top of the image
    putText(image, out_text, Point(25, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    imshow("Image", image);
    imwrite("result_image.jpg", image);

    waitKey(0);
}

