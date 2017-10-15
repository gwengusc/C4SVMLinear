//
// Created by 翁规范 on 10/7/17.
//

#ifndef PEDESTRIAN_DETECTION_C4_C4TRAINER_HPP
#define PEDESTRIAN_DETECTION_C4_C4TRAINER_HPP


#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <sys/time.h>
#include <dirent.h>
#include <C4Feature.hpp>

using namespace std;
using namespace cv;


#define TRAIN true

class MySVM : public CvSVM {
public:
double *get_alpha_vector() {
        return this->decision_func->alpha;
}

float get_rho() {
        return this->decision_func->rho;
}
};


class C4Trainer {

private:
C4Feature feature;
Mat sampleFeatureMat, sampleLabelMat;
int class_pos = 1, class_neg = -1;
int offset = 0;
MySVM *svmPointer;
string svmModelFile, detectorFile;
string version;

private:
void save_detector() {

        int featureSize = feature.Feature_size;
        int supportVectorNum = svmPointer->get_support_vector_count();

        Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);
        Mat supportVectorMat = Mat::zeros(supportVectorNum, featureSize, CV_32FC1);
        Mat resultMat = Mat::zeros(1, featureSize, CV_32FC1);

        for (int i = 0; i < supportVectorNum; i++) {
                const float *pSVData = svmPointer->get_support_vector(i);
                for (int j = 0; j < featureSize; j++) {
                        supportVectorMat.at<float>(i, j) = pSVData[j];
                }
        }

        double *pAlphaData = svmPointer->get_alpha_vector();
        for (int i = 0; i < supportVectorNum; i++) {
                alphaMat.at<float>(0, i) = pAlphaData[i];
        }

        resultMat = -1 * alphaMat * supportVectorMat;

        vector<float> myDetector;
        for (int i = 0; i < featureSize; i++) {
                myDetector.push_back((resultMat.at<float>(0, i)));
        }
        myDetector.push_back(svmPointer->get_rho());
        ofstream fout(detectorFile);
        for (int i = 0; i < myDetector.size(); i++) {
                fout << myDetector[i] << endl;
        }

        fout.close();
}

public:

void AddSamplesToDataMat(vector<string> samples, int type) {

        int featureSize = feature.Feature_size;
        int sampleSize = samples.size();
        Mat src;
        int i;
        for (i = 0; i < sampleSize; ++i) {

                src = imread(samples[i], CV_LOAD_IMAGE_GRAYSCALE);

                feature.Compute(src, sampleFeatureMat.ptr<float>(i + offset));

                sampleLabelMat.at<float>(i + offset, 0) = type;

                src.release();
        }

        offset += i;
}


void LoadTrainData(vector<string> sampleFilesPos, vector<string> sampleFilesNeg) {

        int lenPos = sampleFilesPos.size(), lenNeg = sampleFilesNeg.size();

        int sampleNum = lenPos + lenNeg;

        sampleFeatureMat = Mat(sampleNum, feature.Feature_size, CV_32FC1);
        sampleLabelMat = Mat(sampleNum, 1, CV_32FC1);

        AddSamplesToDataMat(sampleFilesPos, class_pos);
        AddSamplesToDataMat(sampleFilesNeg, class_neg);
}

void Train() {

        CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
        CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
        MySVM svm;
        svmPointer = &svm;
        svmPointer->train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);
        svmPointer->save(svmModelFile.c_str());

        save_detector();
}


C4Trainer(string _version, C4Feature _feature) {
        feature = _feature;
        ResetVersion(_version);
};

void ResetVersion(string _version) {

        version = _version;

        stringstream transformStream;

        transformStream << "C4_SVM_MODEL\\SVM_C4_" << _version << ".xml";

        svmModelFile.assign(transformStream.str());

        transformStream.clear();

        transformStream.str("");

        transformStream << "DETECTOR_TXT\\SVM_C4_detector_" << _version << ".txt";

        detectorFile.assign(transformStream.str());

        transformStream.clear();

        transformStream.str("");
}

string GetSvmModelVersion() {
        return version;
}

};


#endif //PEDESTRIAN_DETECTION_C4_C4TRAINER_HPP
