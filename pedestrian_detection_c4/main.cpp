#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <C4Feature.hpp>
#include <C4Detector.hpp>


int main() {

    cv::Mat src=cv::imread("/Users/wengguifan/Pictures/0428_nws_ocr-l-fatal-035.jpg");

    C4Feature feature;

    //C4Detector(C4Feature _feature, double _thresh = 0.8) {
    C4Detector detector(feature);

    std::vector<cv::Rect>detectedRects;
    std::vector<cv::Rect>results;

    detector.MultiDetecte(src,2,0.8,detectedRects);


    detector.post_process_NMS(detectedRects,2, 0.8);

    // cout<<"targets_size:"<<targets.size()<<endl;

//    detector.post_process_NMS(detectedRects,0, 0.6);

    detector.post_process(detectedRects,results,src.rows,src.cols);


    for(int i = 0; i < results.size(); i++)
    {
        cv::rectangle(src, results[i],cv::Scalar(0,255,0),2 );
    }

    cv::imshow("result",src);
    cv::waitKey( 0 );

    return 0;
}

