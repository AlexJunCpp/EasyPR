#include "easypr/core/plate_judge.h"
#include "easypr/config.h"
#include "easypr/core/core_func.h"
#include "easypr/core/params.h"

namespace easypr {
    //*leijun 命名空间声明和实现可以分离

  PlateJudge* PlateJudge::instance_ = nullptr;
  //*leijun 单例模式

  PlateJudge* PlateJudge::instance() {
    if (!instance_) {
      instance_ = new PlateJudge;
    }
    return instance_;
  }

  PlateJudge::PlateJudge() { 
    bool useLBP = false;
    if (useLBP) {
      LOAD_SVM_MODEL(svm_, kLBPSvmPath);
      //*leijun
      //CvSVM类有个方法，把训练好的结果以xml文件的形式存储，我就是把这个xml文件随EasyPR发布，并让程序在执行前先加载好这个xml。这个xml的位置就是在文件夹Model下面--svm.xml文件。
      extractFeature = getLBPFeatures;
      //*leijun 获得rois的LBP特征
    }
    else {
      LOAD_SVM_MODEL(svm_, kHistSvmPath);
      extractFeature = getHistomPlusColoFeatures;
    }
  }

  void PlateJudge::LoadModel(std::string path) {
    if (path != std::string(kDefaultSvmPath)) {
      if (!svm_->empty())
        svm_->clear();
      LOAD_SVM_MODEL(svm_, path);
    }
  }

  // set the score of plate
  // 0 is plate, -1 is not.
  int PlateJudge::plateSetScore(CPlate& plate) {
    Mat features;
    extractFeature(plate.getPlateMat(), features);
    //*leijun  获取特征·
    float score = svm_->predict(features, noArray(), cv::ml::StatModel::Flags::RAW_OUTPUT);
    //*leijun 第二个默认参数为输出的Array,默认为空，第三个参数为flag，
    //RAW_OUTPUT    
    //makes the method return the raw results (the sum), not the class label·
    //
    //
    //svm的perdict方法的输入是待预测数据的特征，也称之为features。在这里，我们输入的特征是图像全部的像素。由于svm要求输入的特征应该是一个向量，而Mat是与图像宽高对应的矩阵，因此在输入前我们需要使用reshape(1,1)方法把矩阵拉伸成向量。除了全部像素以外，也可以有其他的特征，具体看第三部分“SVM调优”。
    //
    //　　predict方法的输出是float型的值，我们需要把它转变为int型后再进行判断。如果是1代表就是车牌，否则不是。这个"1"的取值是由你在训练时输入的标签决定的。标签，又称之为label，代表某个数据的分类。如果你给SVM模型输入一个车牌，并告诉它，这个图片的标签是5。那么你这边判断时所用的值就应该是5。
    //
    //
    //

    //std::cout << "score:" << score << std::endl;
    if (0) {
      imshow("plate", plate.getPlateMat());
      waitKey(0);
      destroyWindow("plate");
    }
    // score is the distance of margin，below zero is plate, up is not
    // when score is below zero, the samll the value, the more possibliy to be a plate.
    plate.setPlateScore(score);
    if (score < 0.5) return 0;
    else return -1;      
  }

  int PlateJudge::plateJudge(const Mat& plateMat) {
    CPlate plate;
    plate.setPlateMat(plateMat);
    return plateSetScore(plate);
  }

  int PlateJudge::plateJudge(const std::vector<Mat> &inVec,
    std::vector<Mat> &resultVec) {
    int num = inVec.size();
    for (int j = 0; j < num; j++) {
      Mat inMat = inVec[j];

      int response = -1;
      response = plateJudge(inMat);

      if (response == 0) resultVec.push_back(inMat);
    }
    return 0;
  }

  int PlateJudge::plateJudge(const std::vector<CPlate> &inVec,
    std::vector<CPlate> &resultVec) {
    int num = inVec.size();
    for (int j = 0; j < num; j++) {
      CPlate inPlate = inVec[j];
      Mat inMat = inPlate.getPlateMat();
      int response = -1;
      response = plateJudge(inMat);

      if (response == 0)
        resultVec.push_back(inPlate);
      else {
        int w = inMat.cols;
        int h = inMat.rows;
        Mat tmpmat = inMat(Rect_<double>(w * 0.05, h * 0.1, w * 0.9, h * 0.8));
        //*leijun 缩小范围，然后再检查一遍
        Mat tmpDes = inMat.clone();
        resize(tmpmat, tmpDes, Size(inMat.size()));

        response = plateJudge(tmpDes);
        if (response == 0) resultVec.push_back(inPlate);
      }
    }
    return 0;
  }

  // non-maximum suppression
  void NMS(std::vector<CPlate> &inVec, std::vector<CPlate> &resultVec, double overlap) {
    std::sort(inVec.begin(), inVec.end());
    //*leijun 这里sort的是啥?  因为CPlate类重载了<, 标准就是mscore
    std::vector<CPlate>::iterator it = inVec.begin();
    for (; it != inVec.end(); ++it) {
      CPlate plateSrc = *it;
      //std::cout << "plateScore:" << plateSrc.getPlateScore() << std::endl;
      Rect rectSrc = plateSrc.getPlatePos().boundingRect();
      std::vector<CPlate>::iterator itc = it + 1;
      for (; itc != inVec.end();) {
        CPlate plateComp = *itc;
        Rect rectComp = plateComp.getPlatePos().boundingRect();
        //*leijun rotatedRect.boundingRect();
        float iou = computeIOU(rectSrc, rectComp);
        //*leijun 计算IOU
        if (iou > overlap) {
        //*leijun  当IOU大于某个值，就直接去除置信度小一点的那个
          itc = inVec.erase(itc);
        }
        else {
          ++itc;
        }
      }
    }
    resultVec = inVec;
  }

  // judge plate using nms
  int PlateJudge::plateJudgeUsingNMS(const std::vector<CPlate> &inVec, std::vector<CPlate> &resultVec, int maxPlates) {
    std::vector<CPlate> plateVec;
    int num = inVec.size();
    bool useCascadeJudge = true;

    for (int j = 0; j < num; j++) {
      CPlate plate = inVec[j];
      Mat inMat = plate.getPlateMat();
      int result = plateSetScore(plate);
      //*leijun 是否是plate的置信度, 0为plate
      if (0 == result) {
        if (0) {
          imshow("inMat", inMat);
          waitKey(0);
          destroyWindow("inMat");
        }

        if (plate.getPlateLocateType() == CMSER) {
          int w = inMat.cols;
          int h = inMat.rows;
          Mat tmpmat = inMat(Rect_<double>(w * 0.05, h * 0.1, w * 0.9, h * 0.8));
          Mat tmpDes = inMat.clone();
          resize(tmpmat, tmpDes, Size(inMat.size()));
          plate.setPlateMat(tmpDes);
          if (useCascadeJudge) {
            int resultCascade = plateSetScore(plate);
            if (plate.getPlateLocateType() != CMSER)
              plate.setPlateMat(inMat);
            if (resultCascade == 0) {
              if (0) {
                imshow("tmpDes", tmpDes);
                waitKey(0);
                destroyWindow("tmpDes");
              }
              plateVec.push_back(plate);
            }
          }
          else 
            plateVec.push_back(plate);
        }
        else 
          plateVec.push_back(plate);                  
      }
    }

    std::vector<CPlate> reDupPlateVec;
    double overlap = 0.5;
    // double overlap = CParams::instance()->getParam1f();
    // use NMS to get the result plates
    NMS(plateVec, reDupPlateVec, overlap);
    // sort the plates due to their scores
    std::sort(reDupPlateVec.begin(), reDupPlateVec.end());
    // output the plate judge plates
    std::vector<CPlate>::iterator it = reDupPlateVec.begin();
    int count = 0;
    for (; it != reDupPlateVec.end(); ++it) {
      resultVec.push_back(*it);
      if (0) {
        imshow("plateMat", it->getPlateMat());
        waitKey(0);
        destroyWindow("plateMat");
      }
      count++;
      if (count >= maxPlates)
        break;
    }
    return 0;
  }
}
