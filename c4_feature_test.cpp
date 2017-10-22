#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <C4Feature.hpp>



int main() {
        cv::Mat src=cv::imread("/Users/wengguifan/Pictures/0428_nws_ocr-l-fatal-035.jpg");

        C4Feature feature;

        float* histogram=new float[C4Feature::Feature_size];

        feature.Compute(src, histogram);

        std::cout << "out of the function" << std::endl;

        int pixelNumExpected=24*18,pixelReal=0;

        for(int i=0; i<256; i++) {
                pixelReal+=histogram[i];
        }

        std::cout <<"Pixel num expected:"<<pixelNumExpected<< " Pixel num:"<<pixelReal<< std::endl;

        return 0;
}
