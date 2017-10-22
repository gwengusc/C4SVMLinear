//
// Created by 翁规范 on 10/7/17.
//

#ifndef PEDESTRIAN_DETECTION_C4_C4DETECTOR_HPP
#define PEDESTRIAN_DETECTION_C4_C4DETECTOR_HPP


#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>
#include "IntImage.hpp"
#include <C4Feature.hpp>

using std::endl;
using std::vector;
using std::string;
using std::ofstream;
using std::ifstream;

class C4Detector {

private:
//    IntImage<double> *integrals;
    IntImage<double> image, sobel;
    IntImage<int> ct;
    Array2dC<int> hist;
    IntImage<double> scores;
    double *detector;
    double thresh;
    C4Feature feature;

// Detection functions
// initialization -- compute the Census Tranform image for CENTRIST
    void InitImage(IntImage<double> &original) {
        image = original;
        image.Sobel(sobel, false, false);
        feature.ComputeCT(sobel, ct);
    }


// combine the (xdiv-1)*(ydiv-1) integral images into a single one
    void InitIntegralImages(const int stepsize) {
        const int hd = feature.GetBlockSize().height;
        const int wd = feature.GetBlockSize().width;
        scores.Create(ct.nrow, ct.ncol);
        scores.Zero(thresh / hd / wd);
        double *linearweights = detector;

        for (int i = 0; i < feature.HUMAN_xdiv - C4Feature::EXT; i++) {

            int xoffset = feature.HUMAN_height / feature.HUMAN_xdiv * i;// 108/9=12

            for (int j = 0; j < feature.HUMAN_ydiv - C4Feature::EXT; j++) {
                int yoffset = feature.HUMAN_width / feature.HUMAN_ydiv * j;// 36/4=9

                for (int x = 2; x < ct.nrow - 2 - xoffset; x++) {

                    int *ctp = ct.p[x + xoffset] + yoffset;

                    double *tempp = scores.p[x];

                    for (int y = 2; y < ct.ncol - 2 - yoffset; y++)

                        tempp[y] += linearweights[ctp[y]];

                }

                linearweights += feature.BaseLength;

            }
        }

        scores.CalcIntegralImageInPlace();

        for (int i = 2; i < ct.nrow - 2 - feature.HUMAN_height; i += stepsize) {
            double *p1 = scores.p[i];
            double *p2 = scores.p[i + hd];

            for (int j = 2; j < ct.ncol - 2 - feature.HUMAN_width; j += stepsize)
                p1[j] += (p2[j + wd] - p2[j] - p1[j + wd]);

        }
    }

// Resize the input image and then re-compute Sobel image etc
    void ResizeImage(float ratio) {
        image.Resize(sobel, ratio);
        image.Swap(sobel);
        image.Sobel(sobel, false, false);
        feature.ComputeCT(sobel, ct);
    }

public:


    std::vector<cv::Rect> MultiDetecte(cv::Mat &src, int stepSize, float ratio,std::vector<cv::Rect> &detectedRects) {
        IntImage<double> original;
        original.Load(src);

        detectedRects.clear();

        if (original.nrow < feature.HUMAN_height + 5 || original.ncol < feature.HUMAN_width + 5) {
            std::cout<<"the size of the image is too small"<<endl;
            return detectedRects;
        }

        const int hd = feature.HUMAN_height / feature.HUMAN_xdiv;//108/9=12
        const int wd = feature.HUMAN_width / feature.HUMAN_ydiv;//36/4=9
        InitImage(original);
        // results.clear();

        // hist.Create(1, baseflength * (feature.HUMAN_xdiv - EXT) * (feature.HUMAN_ydiv - EXT));
        int oheight = original.nrow, owidth = original.ncol;
        cv::Rect rect;

        while (image.nrow >= feature.HUMAN_height && image.ncol >= feature.HUMAN_width) {

            InitIntegralImages(stepSize);
            for (int i = 2; i + feature.HUMAN_height < image.nrow - 2; i += stepSize) {
                const double *sp = scores.p[i];
                for (int j = 2; j + feature.HUMAN_width < image.ncol - 2; j += stepSize) {
                    if (sp[j] > 0) {

//                        the constructor of Rect is Rect(x(left,col),y(top,row),width,height);

                        detectedRects.push_back(cv::Rect(j * owidth / image.ncol, i * oheight / image.nrow,
                                                         feature.HUMAN_width * owidth / image.ncol,
                                                         feature.HUMAN_height * oheight / image.nrow));
                    }
                }
            }
            ResizeImage(ratio);
        }

        return detectedRects;
//        return 0;
    }


    std::vector<cv::Rect> post_process(std::vector<cv::Rect>&targets,std::vector<cv::Rect>&result_processed,int row,int col){

        int i,j,results_size=targets.size();
        cv::Rect rect0,rect_intercect;

        for(i=0; i<results_size; i++) {

            rect0=targets[i];


            for(j=i+1; j<results_size; j++) {

                rect_intercect=rect0&targets[j];

                if((rect_intercect.area()>rect0.area()*0.8)) {
                    break;
                }

            }

            if(j==results_size) {

                if(rect0.x<0) {
                    rect0.x=0;
                }

                if(rect0.y<0) {
                    rect0.y=0;
                }

                if(rect0.x+rect0.width>=col-1) {

                    rect0.width=col-1-rect0.x;

                }

                if(rect0.y+rect0.height>=row-1) {
                    rect0.height=row-1-rect0.y;
                }

                result_processed.push_back(rect0);
            }
        }

        return result_processed;
    }


// A simple post-process (NMS, non-maximal suppression)
// "result" -- rectangles before merging
//          -- after this function it contains rectangles after NMS
// "combine_min" -- threshold of how many detection are needed to survive
    void post_process_NMS(std::vector<cv::Rect>& result,const int combine_min,float overlap_proportion)
    {
        std::vector<cv::Rect> res1;
        std::vector<cv::Rect> resmax;
        std::vector<int> res2;
        bool yet;
        cv::Rect rectInter;

        for(unsigned int i=0,size_i=result.size(); i<size_i; i++)
        {
            yet = false;
            cv::Rect result_i = result[i];
            for(unsigned int j=0,size_r=res1.size(); j<size_r; j++)
            {
                cv::Rect resmax_j = resmax[j];
                rectInter=result_i&resmax[j];

                if(  rectInter.area()>overlap_proportion*result_i.area()&& rectInter.area()>overlap_proportion*resmax_j.area())
                {
                    cv::Rect res1_j = res1[j];
                    // resmax_j.Union(resmax_j,result_i);
                    resmax_j=resmax_j|result_i;

                    res1_j.x= result_i.x;
                    res1_j.y= result_i.y;
                    res1_j.height += result_i.height;
                    res1_j.width += result_i.width;


                    res2[j]++;

                    yet = true;

                    break;
                }
            }

            if(yet==false)
            {
                res1.push_back(result_i);
                resmax.push_back(result_i);
                res2.push_back(1);
            }
        }

        for(unsigned int i=0,size=res1.size(); i<size; i++)
        {
            const int count = res2[i];
            cv::Rect res1_i = res1[i];
            res1_i.x /= count;
            res1_i.y /= count;
            res1_i.height /= count;
            res1_i.width /= count;
        }

        result.clear();
        for(unsigned int i=0,size=res1.size(); i<size; i++)
            if(res2[i]>combine_min)
                result.push_back(res1[i]);
    }


    void SetSVMVersion(string _version) {

    }

    float PredictSamplesPos() {
        float correctRate;
        return correctRate;
    }

    float PredictSampleNeg() {
        float correctRate;
        return correctRate;
    }


    C4Detector(C4Feature _feature)  {
        feature = _feature;
        this->detector=feature.GetDetector();
        this->thresh=feature.thresh;
    }

};


#endif //PEDESTRIAN_DETECTION_C4_C4DETECTOR_HPP
