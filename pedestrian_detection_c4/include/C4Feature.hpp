//
// Created by 翁规范 on 10/7/17.
//

#ifndef PEDESTRIAN_DETECTION_C4_C4FEATURE_HPP
#define PEDESTRIAN_DETECTION_C4_C4FEATURE_HPP


#include "IntImage.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>

using std::string;
using std::ofstream;
using std::ifstream;

class C4Feature {

private:
    int BLOCK_height = 24;//HUMAN_height/HUMAN_xdiv*height;
    int BLOCK_width = 18;//HUMAN_width/HUMAN_ydiv*width
    cv::Size BLOCK_size;
    double *detector;


    double *LoadDetector(string detectorFilePath) {

        if (detector == NULL) {

            ifstream fin(detectorFilePath);

            detector = new double[C4Feature::Feature_size + 1];

            float val = 0.0f;
            while (!fin.eof()) {

//                std::cout << val << std::endl;
                fin >> val;
            }
            fin.close();
        }

        return detector;
    }


public:

    const int HUMAN_height = 108;
    const int HUMAN_width = 36;
    const int HUMAN_xdiv = 9;
    const int HUMAN_ydiv = 4;
    const int BaseLength = 256;
    static const int EXT = 1;
    static const int Feature_size = 256 * 8 * 3;//6144
// const int Feature_increment;

    double thresh;

public:

    double* GetDetector(){
        return this->detector;
    }

// compute the "ct" image from sobel
    void ComputeCT(IntImage<double> &sobel, IntImage<int> &ct) {
        ct.Create(sobel.nrow, sobel.ncol);
        for (int i = 2; i < sobel.nrow - 2; i++) {
            double *p1 = sobel.p[i - 1];
            double *p2 = sobel.p[i];
            double *p3 = sobel.p[i + 1];
            int *ctp = ct.p[i];
            for (int j = 2; j < sobel.ncol - 2; j++) {
                int index = 0;
                if (p2[j] <= p1[j - 1]) index += 0x80;
                if (p2[j] <= p1[j]) index += 0x40;
                if (p2[j] <= p1[j + 1]) index += 0x20;
                if (p2[j] <= p2[j - 1]) index += 0x10;
                if (p2[j] <= p2[j + 1]) index += 0x08;
                if (p2[j] <= p3[j - 1]) index += 0x04;
                if (p2[j] <= p3[j]) index += 0x02;
                if (p2[j] <= p3[j + 1]) index++;
                ctp[j] = index;
            }
        }
    }

    void SetBlockSize(int width=18, int height=24) {
        BLOCK_size.height = height;//108/9*2
        BLOCK_size.width = width;//36/4*2
    }

    cv::Size GetBlockSize() {
        return BLOCK_size;
    }

    void ComputeHistogram(IntImage<int> &ct, float *histogram) {

        int histogramOffset = 0, increment = 256;
        float *histogramUnit = histogram;

        for (int xoffset = 0; xoffset < ct.nrow - 2 * HUMAN_xdiv; xoffset += HUMAN_xdiv)
            for (int yoffset = 0; yoffset < ct.ncol - 2 * HUMAN_ydiv; yoffset += HUMAN_ydiv) {
                for (int i = xoffset; i < BLOCK_height; i++) {
                    int *ctp = ct.p[i];
                    for (int j = yoffset; j < BLOCK_width; j++) {
                        // histogram[ctp[j]+histogramOffset]++;
                        histogramUnit[ctp[j]]++;
                    }
                }
                histogramUnit += increment;
            }


    }

//float* Compute(IntImage<float>& result,){
    void Compute(cv::Mat src, float *histogram) {

        IntImage<double> original, sobel;
        IntImage<int> ct;
        original.Load(src);
        original.Sobel(sobel, false, false);
        ComputeCT(sobel, ct);
        ComputeHistogram(ct, histogram);
    }


    C4Feature(
            string detectorPath = "/Users/wengguifan/CLionProjects/pedestrian_detection_c4/model/DetectorOriginal.txt",
            cv::Size BlockSize = cv::Size(2, 2), double _thresh = 0.8) {
        detector = NULL;
        this->LoadDetector(detectorPath);
        this->SetBlockSize(BlockSize.width, BlockSize.height);
        this->thresh = _thresh;
    }

    C4Feature &operator=(C4Feature &source) {
        std::copy(source.detector, source.detector + Feature_size + 1, this->detector);
        this->SetBlockSize(source.GetBlockSize().width, source.GetBlockSize().height);
        this->thresh = source.thresh;
        return *this;
    }

};


#endif //PEDESTRIAN_DETECTION_C4_C4FEATURE_HPP
