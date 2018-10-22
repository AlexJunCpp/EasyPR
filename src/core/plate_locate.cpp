#include "easypr/core/plate_locate.h"
#include "easypr/core/core_func.h"
#include "easypr/util/util.h"
#include "easypr/core/params.h"

using namespace std;

namespace easypr {

const float DEFAULT_ERROR = 0.9f;    // 0.6
const float DEFAULT_ASPECT = 3.75f;  // 3.75

CPlateLocate::CPlateLocate() {
  m_GaussianBlurSize = DEFAULT_GAUSSIANBLUR_SIZE;
  m_MorphSizeWidth = DEFAULT_MORPH_SIZE_WIDTH;
  m_MorphSizeHeight = DEFAULT_MORPH_SIZE_HEIGHT;

  m_error = DEFAULT_ERROR;
  m_aspect = DEFAULT_ASPECT;
  m_verifyMin = DEFAULT_VERIFY_MIN;
  m_verifyMax = DEFAULT_VERIFY_MAX;

  m_angle = DEFAULT_ANGLE;

  m_debug = DEFAULT_DEBUG;
}

void CPlateLocate::setLifemode(bool param) {
  if (param) {
    setGaussianBlurSize(5);
    setMorphSizeWidth(10);
    setMorphSizeHeight(3);
    setVerifyError(0.75);
    setVerifyAspect(4.0);
    setVerifyMin(1);
    setVerifyMax(200);
  } else {
    setGaussianBlurSize(DEFAULT_GAUSSIANBLUR_SIZE);
    setMorphSizeWidth(DEFAULT_MORPH_SIZE_WIDTH);
    setMorphSizeHeight(DEFAULT_MORPH_SIZE_HEIGHT);
    setVerifyError(DEFAULT_ERROR);
    setVerifyAspect(DEFAULT_ASPECT);
    setVerifyMin(DEFAULT_VERIFY_MIN);
    setVerifyMax(DEFAULT_VERIFY_MAX);
  }
}

bool CPlateLocate::verifySizes(RotatedRect mr) {
  float error = m_error;
  // Spain car plate size: 52x11 aspect 4,7272
  // China car plate size: 440mm*140mm，aspect 3.142857

  // Real car plate size: 136 * 32, aspect 4
  float aspect = m_aspect;

  // Set a min and max area. All other patchs are discarded
  // int min= 1*aspect*1; // minimum area
  // int max= 2000*aspect*2000; // maximum area
  int min = 34 * 8 * m_verifyMin;  // minimum area
  int max = 34 * 8 * m_verifyMax;  // maximum area

  // Get only patchs that match to a respect ratio.
  float rmin = aspect - aspect * error;
  float rmax = aspect + aspect * error;

  float area = mr.size.height * mr.size.width;
  float r = (float) mr.size.width / (float) mr.size.height;
  if (r < 1) r = (float) mr.size.height / (float) mr.size.width;

  // cout << "area:" << area << endl;
  // cout << "r:" << r << endl;

  if ((area < min || area > max) || (r < rmin || r > rmax))
    return false;
  else
    return true;
}

//! mser search method
int CPlateLocate::mserSearch(const Mat &src,  vector<Mat> &out,
  vector<vector<CPlate>>& out_plateVec, bool usePlateMser, vector<vector<RotatedRect>>& out_plateRRect,
  int img_index, bool showDebug) {
  vector<Mat> match_grey;

  vector<CPlate> plateVec_blue;
  plateVec_blue.reserve(16);
  vector<RotatedRect> plateRRect_blue;
  plateRRect_blue.reserve(16);

  vector<CPlate> plateVec_yellow;
  plateVec_yellow.reserve(16);

  vector<RotatedRect> plateRRect_yellow;
  plateRRect_yellow.reserve(16);

  mserCharMatch(src, match_grey, plateVec_blue, plateVec_yellow, usePlateMser, plateRRect_blue, plateRRect_yellow, img_index, showDebug);

  out_plateVec.push_back(plateVec_blue);
  out_plateVec.push_back(plateVec_yellow);

  out_plateRRect.push_back(plateRRect_blue);
  out_plateRRect.push_back(plateRRect_yellow);

  out = match_grey;

  return 0;
}


int CPlateLocate::colorSearch(const Mat &src, const Color r, Mat &out,
                              vector<RotatedRect> &outRects) {
  Mat match_grey;

  // width is important to the final results;
  const int color_morph_width = 10;
  const int color_morph_height = 2;

  colorMatch(src, match_grey, r, false);
  SHOW_IMAGE(match_grey, 0);

  Mat src_threshold;
  threshold(match_grey, src_threshold, 0, 255,
            CV_THRESH_OTSU + CV_THRESH_BINARY);

  Mat element = getStructuringElement(
      MORPH_RECT, Size(color_morph_width, color_morph_height));
  morphologyEx(src_threshold, src_threshold, MORPH_CLOSE, element);

  //if (m_debug) {
  //  utils::imwrite("resources/image/tmp/color.jpg", src_threshold);
  //}

  src_threshold.copyTo(out);


  vector<vector<Point>> contours;

  findContours(src_threshold,
               contours,               // a vector of contours
               CV_RETR_EXTERNAL,
               CV_CHAIN_APPROX_NONE);  // all pixels of each contours

  vector<vector<Point>>::iterator itc = contours.begin();
  while (itc != contours.end()) {
    RotatedRect mr = minAreaRect(Mat(*itc));

    if (!verifySizes(mr))
      itc = contours.erase(itc);
    else {
      ++itc;
      outRects.push_back(mr);
    }
  }

  return 0;
}


int CPlateLocate::sobelFrtSearch(const Mat &src,
                                 vector<Rect_<float>> &outRects) {
    //*leijun sobelFrtSearch()函数中通过 sobelOper()
    //进行sobel定位，主要步骤如下：
    //
    //1、对图像进行高斯滤波，为Sobel算子计算去除干扰噪声；
    //
    //2、图像灰度化，提高运算速度；
    //
    //3、对图像进行Sobel运算，得到图像的一阶水平方向导数；
    //
    //4、通过otsu进行阈值分割；
    //
    //5、通过形态学闭操作，连接车牌区域。
    //
    //此处通过Sobel算子进行车牌定位，仅仅做水平方向求导，而不做垂直方向求导。这样做的意义是，如果做了垂直方向求导，会检测出很多水平边缘。水平边缘多也许有利于生成更精确的轮廓，但是由于有些车子前端太多的水平边缘了，例如车头排气孔，标志等等，很多的水平边缘会误导我们的连接结果，导致我们得不到一个恰好的车牌位置。 
  Mat src_threshold;

  sobelOper(src, src_threshold, m_GaussianBlurSize, m_MorphSizeWidth,
            m_MorphSizeHeight);

  vector<vector<Point>> contours;
  findContours(src_threshold,
               contours,               // a vector of contours
               CV_RETR_EXTERNAL,
               CV_CHAIN_APPROX_NONE);  // all pixels of each contours
  //*leijun
  //对图像轮廓进行搜索，轮廓搜索将全图的轮廓都搜索出来了，需要进行筛选，对轮廓求最小外接矩形，并在
  //verifySizes() 中进行验证，不满足条件的删除。

  vector<vector<Point>>::iterator itc = contours.begin();

  vector<RotatedRect> first_rects;

  while (itc != contours.end()) {
    RotatedRect mr = minAreaRect(Mat(*itc));
    //*leijun  最小外接矩形


    if (verifySizes(mr)) {
        //* 验证size大小
      first_rects.push_back(mr);

      float area = mr.size.height * mr.size.width;
      float r = (float) mr.size.width / (float) mr.size.height;
      if (r < 1) r = (float) mr.size.height / (float) mr.size.width;
    }

    ++itc;
  }

  for (size_t i = 0; i < first_rects.size(); i++) {
    RotatedRect roi_rect = first_rects[i];

    Rect_<float> safeBoundRect;
    if (!calcSafeRect(roi_rect, src, safeBoundRect)) continue;

    outRects.push_back(safeBoundRect);
  }
  return 0;
}


int CPlateLocate::sobelSecSearchPart(Mat &bound, Point2f refpoint,
                                     vector<RotatedRect> &outRects) {
  Mat bound_threshold;

  sobelOperT(bound, bound_threshold, 3, 6, 2);

  Mat tempBoundThread = bound_threshold.clone();

  clearLiuDingOnly(tempBoundThread);

  int posLeft = 0, posRight = 0;
  if (bFindLeftRightBound(tempBoundThread, posLeft, posRight)) {

    // find left and right bounds to repair

    if (posRight != 0 && posLeft != 0 && posLeft < posRight) {
      int posY = int(bound_threshold.rows * 0.5);
      for (int i = posLeft + (int) (bound_threshold.rows * 0.1);
           i < posRight - 4; i++) {
        bound_threshold.data[posY * bound_threshold.cols + i] = 255;
      }
    }

    utils::imwrite("resources/image/tmp/repaireimg1.jpg", bound_threshold);

    // remove the left and right boundaries

    for (int i = 0; i < bound_threshold.rows; i++) {
      bound_threshold.data[i * bound_threshold.cols + posLeft] = 0;
      bound_threshold.data[i * bound_threshold.cols + posRight] = 0;
    }
    utils::imwrite("resources/image/tmp/repaireimg2.jpg", bound_threshold);
  }

  vector<vector<Point>> contours;
  findContours(bound_threshold,
               contours,               // a vector of contours
               CV_RETR_EXTERNAL,
               CV_CHAIN_APPROX_NONE);  // all pixels of each contours

  vector<vector<Point>>::iterator itc = contours.begin();

  vector<RotatedRect> second_rects;
  while (itc != contours.end()) {
    RotatedRect mr = minAreaRect(Mat(*itc));
    second_rects.push_back(mr);
    ++itc;
  }

  for (size_t i = 0; i < second_rects.size(); i++) {
    RotatedRect roi = second_rects[i];
    if (verifySizes(roi)) {
      Point2f refcenter = roi.center + refpoint;
      Size2f size = roi.size;
      float angle = roi.angle;

      RotatedRect refroi(refcenter, size, angle);
      outRects.push_back(refroi);
    }
  }

  return 0;
}


int CPlateLocate::sobelSecSearch(Mat &bound, Point2f refpoint,
                                 vector<RotatedRect> &outRects) {
  Mat bound_threshold;


  sobelOper(bound, bound_threshold, 3, 10, 3);

  utils::imwrite("resources/image/tmp/sobelSecSearch.jpg", bound_threshold);

  vector<vector<Point>> contours;
  findContours(bound_threshold,
               contours,               // a vector of contours
               CV_RETR_EXTERNAL,
               CV_CHAIN_APPROX_NONE);  // all pixels of each contours

  vector<vector<Point>>::iterator itc = contours.begin();

  vector<RotatedRect> second_rects;
  while (itc != contours.end()) {
    RotatedRect mr = minAreaRect(Mat(*itc));
    second_rects.push_back(mr);
    ++itc;
  }

  for (size_t i = 0; i < second_rects.size(); i++) {
    RotatedRect roi = second_rects[i];
    if (verifySizes(roi)) {
      Point2f refcenter = roi.center + refpoint;
      Size2f size = roi.size;
      float angle = roi.angle;

      RotatedRect refroi(refcenter, size, angle);
      outRects.push_back(refroi);
    }
  }

  return 0;
}


int CPlateLocate::sobelOper(const Mat &in, Mat &out, int blurSize, int morphW,
                            int morphH) {
  Mat mat_blur;
  mat_blur = in.clone();
  GaussianBlur(in, mat_blur, Size(blurSize, blurSize), 0, 0, BORDER_DEFAULT);
  //*leijun  高斯模糊处理噪声，为后面灰度化后进行sobel算子提供有利条件

  Mat mat_gray;
  if (mat_blur.channels() == 3)
    cvtColor(mat_blur, mat_gray, CV_RGB2GRAY);
  //*leijun 灰度化，提高运算速度
  else
    mat_gray = mat_blur;

  int scale = SOBEL_SCALE;
  int delta = SOBEL_DELTA;
  int ddepth = SOBEL_DDEPTH;

  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;


  Sobel(mat_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
  //*leijun  sobel算法
  convertScaleAbs(grad_x, abs_grad_x);

  Mat grad;
  addWeighted(abs_grad_x, SOBEL_X_WEIGHT, 0, 0, 0, grad);

  Mat mat_threshold;
  double otsu_thresh_val =
      threshold(grad, mat_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);


  Mat element = getStructuringElement(MORPH_RECT, Size(morphW, morphH));
  morphologyEx(mat_threshold, mat_threshold, MORPH_CLOSE, element);

  out = mat_threshold;

  return 0;
}

void deleteNotArea(Mat &inmat, Color color = UNKNOWN) {
  Mat input_grey;
  cvtColor(inmat, input_grey, CV_BGR2GRAY);

  int w = inmat.cols;
  int h = inmat.rows;

  Mat tmpMat = inmat(Rect_<double>(w * 0.15, h * 0.1, w * 0.7, h * 0.7));

  Color plateType;
  if (UNKNOWN == color) {
    plateType = getPlateType(tmpMat, true);
  }
  else {
    plateType = color;
  }

  Mat img_threshold;

  if (BLUE == plateType) {
    img_threshold = input_grey.clone();
    Mat tmp = input_grey(Rect_<double>(w * 0.15, h * 0.15, w * 0.7, h * 0.7));
    int threadHoldV = ThresholdOtsu(tmp);

    threshold(input_grey, img_threshold, threadHoldV, 255, CV_THRESH_BINARY);
    // threshold(input_grey, img_threshold, 5, 255, CV_THRESH_OTSU +
    // CV_THRESH_BINARY);

    utils::imwrite("resources/image/tmp/inputgray2.jpg", img_threshold);

  } else if (YELLOW == plateType) {
    img_threshold = input_grey.clone();
    Mat tmp = input_grey(Rect_<double>(w * 0.1, h * 0.1, w * 0.8, h * 0.8));
    int threadHoldV = ThresholdOtsu(tmp);

    threshold(input_grey, img_threshold, threadHoldV, 255,
              CV_THRESH_BINARY_INV);

    utils::imwrite("resources/image/tmp/inputgray2.jpg", img_threshold);

    // threshold(input_grey, img_threshold, 10, 255, CV_THRESH_OTSU +
    // CV_THRESH_BINARY_INV);
  } else
    threshold(input_grey, img_threshold, 10, 255,
              CV_THRESH_OTSU + CV_THRESH_BINARY);

  //img_threshold = input_grey.clone();
  //spatial_ostu(img_threshold, 8, 2, plateType);

  int posLeft = 0;
  int posRight = 0;

  int top = 0;
  int bottom = img_threshold.rows - 1;
  clearLiuDing(img_threshold, top, bottom);

  if (0) {
    imshow("inmat", inmat);
    waitKey(0);
    destroyWindow("inmat");
  }

  if (bFindLeftRightBound1(img_threshold, posLeft, posRight)) {
    inmat = inmat(Rect(posLeft, top, w - posLeft, bottom - top));
    if (0) {
      imshow("inmat", inmat);
      waitKey(0);
      destroyWindow("inmat");
    }
  }
}


int CPlateLocate::deskew(const Mat &src, const Mat &src_b,
                         vector<RotatedRect> &inRects,
                         vector<CPlate> &outPlates, bool useDeteleArea, Color color) {
    //*leijun  https://www.cnblogs.com/freedomker/p/7250997.html
    //
  Mat mat_debug;
  src.copyTo(mat_debug);

  for (size_t i = 0; i < inRects.size(); i++) {
    RotatedRect roi_rect = inRects[i];

    float r = (float) roi_rect.size.width / (float) roi_rect.size.height;
    float roi_angle = roi_rect.angle;

    Size roi_rect_size = roi_rect.size;
    if (r < 1) {
      roi_angle = 90 + roi_angle;
      swap(roi_rect_size.width, roi_rect_size.height);
    }

    if (m_debug) {
      Point2f rect_points[4];
      roi_rect.points(rect_points);
      for (int j = 0; j < 4; j++)
        line(mat_debug, rect_points[j], rect_points[(j + 1) % 4],
             Scalar(0, 255, 255), 1, 8);
    }

    // changed
    // rotation = 90 - abs(roi_angle);
    // rotation < m_angel;

    // m_angle=60
    if (roi_angle - m_angle < 0 && roi_angle + m_angle > 0) {
      Rect_<float> safeBoundRect;
      bool isFormRect = calcSafeRect(roi_rect, src, safeBoundRect);
      if (!isFormRect) continue;

      Mat bound_mat = src(safeBoundRect);
      Mat bound_mat_b = src_b(safeBoundRect);

      if (0) {
        imshow("bound_mat_b", bound_mat_b);
        waitKey(0);
        destroyWindow("bound_mat_b");
      }

      Point2f roi_ref_center = roi_rect.center - safeBoundRect.tl();

      Mat deskew_mat;
      if ((roi_angle - 5 < 0 && roi_angle + 5 > 0) || 90.0 == roi_angle ||
          -90.0 == roi_angle) {
        deskew_mat = bound_mat;
        //*leijun 直接输出
      } else {
        Mat rotated_mat;
        Mat rotated_mat_b;

        if (!rotation(bound_mat, rotated_mat, roi_rect_size, roi_ref_center, roi_angle))
          continue;

        if (!rotation(bound_mat_b, rotated_mat_b, roi_rect_size, roi_ref_center, roi_angle))
            //*leijun  rotation()函数主要用于对倾斜的图片进行旋转
          continue;

        // we need affine for rotatioed image
        double roi_slope = 0;
        // imshow("1roated_mat",rotated_mat);
        // imshow("rotated_mat_b",rotated_mat_b);
        if (isdeflection(rotated_mat_b, roi_angle, roi_slope)) {
            //*leijun 函数 isdeflection()
            //的主要功能是判断车牌偏斜的程度，并且计算偏斜的值
          affine(rotated_mat, deskew_mat, roi_slope);
          //*leijun 偏斜校正
        } else
          deskew_mat = rotated_mat;
      }

      Mat plate_mat;
      plate_mat.create(HEIGHT, WIDTH, TYPE);

      // haitungaga add，affect 25% to full recognition.
      if (useDeteleArea)
        deleteNotArea(deskew_mat, color);

      if (deskew_mat.cols * 1.0 / deskew_mat.rows > 2.3 && deskew_mat.cols * 1.0 / deskew_mat.rows < 6) {
        if (deskew_mat.cols >= WIDTH || deskew_mat.rows >= HEIGHT)
          resize(deskew_mat, plate_mat, plate_mat.size(), 0, 0, INTER_AREA);
        //*leijun 最后使用 resize 函数将车牌区域统一化为 EasyPR
        //的车牌大小，大小为136*36。
        else
          resize(deskew_mat, plate_mat, plate_mat.size(), 0, 0, INTER_CUBIC);

        CPlate plate;
        plate.setPlatePos(roi_rect);
        plate.setPlateMat(plate_mat);
        if (color != UNKNOWN) plate.setPlateColor(color);
        outPlates.push_back(plate);
      }
    }
  }
  return 0;
}


bool CPlateLocate::rotation(Mat &in, Mat &out, const Size rect_size,
                            const Point2f center, const double angle) {
  if (0) {
    imshow("in", in);
    waitKey(0);
    destroyWindow("in");
  }

  Mat in_large;
  in_large.create(int(in.rows * 1.5), int(in.cols * 1.5), in.type());

  float x = in_large.cols / 2 - center.x > 0 ? in_large.cols / 2 - center.x : 0;
  float y = in_large.rows / 2 - center.y > 0 ? in_large.rows / 2 - center.y : 0;

  float width = x + in.cols < in_large.cols ? in.cols : in_large.cols - x;
  float height = y + in.rows < in_large.rows ? in.rows : in_large.rows - y;

  /*assert(width == in.cols);
  assert(height == in.rows);*/

  if (width != in.cols || height != in.rows) return false;

  Mat imageRoi = in_large(Rect_<float>(x, y, width, height));
  addWeighted(imageRoi, 0, in, 1, 0, imageRoi);

  Point2f center_diff(in.cols / 2.f, in.rows / 2.f);
  Point2f new_center(in_large.cols / 2.f, in_large.rows / 2.f);

  Mat rot_mat = getRotationMatrix2D(new_center, angle, 1);
  //*leijun
  //在旋转的过程当中，遇到一个问题，就是旋转后的图像被截断了.getRotationMatrix2D()
  //函数主要根据旋转中心和角度进行旋转，当旋转角度还小时，一切都还好，但当角度变大时，明显我们看到的外接矩形的大小也在扩增。在这里，外接矩形被称为视框，也就是我需要旋转的正方形所需要的最小区域。随着旋转角度的变大，视框明显增大。 

  /*imshow("in_copy", in_large);
  waitKey(0);*/

  Mat mat_rotated;
  warpAffine(in_large, mat_rotated, rot_mat, Size(in_large.cols, in_large.rows),
             CV_INTER_CUBIC);

  /*imshow("mat_rotated", mat_rotated);
  waitKey(0);*/

  Mat img_crop;
  getRectSubPix(mat_rotated, Size(rect_size.width, rect_size.height),
                new_center, img_crop);

  out = img_crop;

  if (0) {
    imshow("out", out);
    waitKey(0);
    destroyWindow("out");
  }

  /*imshow("img_crop", img_crop);
  waitKey(0);*/

  return true;
}

bool CPlateLocate::isdeflection(const Mat &in, const double angle,
                                double &slope) { /*imshow("in",in);
                                                waitKey(0);*/
  if (0) {
    imshow("in", in);
    waitKey(0);
    destroyWindow("in");
  }
  
  int nRows = in.rows;
  int nCols = in.cols;

  assert(in.channels() == 1);

  int comp_index[3];
  int len[3];

  comp_index[0] = nRows / 4;
  comp_index[1] = nRows / 4 * 2;
  comp_index[2] = nRows / 4 * 3;

  const uchar* p;

  for (int i = 0; i < 3; i++) {
    int index = comp_index[i];
    p = in.ptr<uchar>(index);

    int j = 0;
    int value = 0;
    while (0 == value && j < nCols) value = int(p[j++]);

    len[i] = j;
  }

  // cout << "len[0]:" << len[0] << endl;
  // cout << "len[1]:" << len[1] << endl;
  // cout << "len[2]:" << len[2] << endl;

  // len[0]/len[1]/len[2] are used to calc the slope

  double maxlen = max(len[2], len[0]);
  double minlen = min(len[2], len[0]);
  double difflen = abs(len[2] - len[0]);

  double PI = 3.14159265;

  double g = tan(angle * PI / 180.0);

  if (maxlen - len[1] > nCols / 32 || len[1] - minlen > nCols / 32) {

    double slope_can_1 =
        double(len[2] - len[0]) / double(comp_index[1]);
    double slope_can_2 = double(len[1] - len[0]) / double(comp_index[0]);
    double slope_can_3 = double(len[2] - len[1]) / double(comp_index[0]);
    // cout<<"angle:"<<angle<<endl;
    // cout<<"g:"<<g<<endl;
    // cout << "slope_can_1:" << slope_can_1 << endl;
    // cout << "slope_can_2:" << slope_can_2 << endl;
    // cout << "slope_can_3:" << slope_can_3 << endl;
    // if(g>=0)
    slope = abs(slope_can_1 - g) <= abs(slope_can_2 - g) ? slope_can_1
                                                         : slope_can_2;
    // cout << "slope:" << slope << endl;
    return true;
  } else {
    slope = 0;
  }

  return false;
}


void CPlateLocate::affine(const Mat &in, Mat &out, const double slope) {
  // imshow("in", in);
  // waitKey(0);

  Point2f dstTri[3];
  Point2f plTri[3];

  float height = (float) in.rows;
  float width = (float) in.cols;
  float xiff = (float) abs(slope) * height;

  if (slope > 0) {

    // right, new position is xiff/2

    plTri[0] = Point2f(0, 0);
    plTri[1] = Point2f(width - xiff - 1, 0);
    plTri[2] = Point2f(0 + xiff, height - 1);

    dstTri[0] = Point2f(xiff / 2, 0);
    dstTri[1] = Point2f(width - 1 - xiff / 2, 0);
    dstTri[2] = Point2f(xiff / 2, height - 1);
  } else {

    // left, new position is -xiff/2

    plTri[0] = Point2f(0 + xiff, 0);
    plTri[1] = Point2f(width - 1, 0);
    plTri[2] = Point2f(0, height - 1);

    dstTri[0] = Point2f(xiff / 2, 0);
    dstTri[1] = Point2f(width - 1 - xiff + xiff / 2, 0);
    dstTri[2] = Point2f(xiff / 2, height - 1);
  }

  Mat warp_mat = getAffineTransform(plTri, dstTri);

  Mat affine_mat;
  affine_mat.create((int) height, (int) width, TYPE);

  if (in.rows > HEIGHT || in.cols > WIDTH)

    warpAffine(in, affine_mat, warp_mat, affine_mat.size(),
               CV_INTER_AREA);
  else
    warpAffine(in, affine_mat, warp_mat, affine_mat.size(), CV_INTER_CUBIC);

  out = affine_mat;
}

int CPlateLocate::plateColorLocate(Mat src, vector<CPlate> &candPlates,
                                   int index) {
    //*leijun H 分量是 HSV 模型中唯一跟颜色本质相关的分量。 只要固定了 H 的值，
    //并且保持 S 和 V 分量不太小，那么表现的颜色就会基本固定
  vector<RotatedRect> rects_color_blue;
  rects_color_blue.reserve(64);
  vector<RotatedRect> rects_color_yellow;
  rects_color_yellow.reserve(64);

  vector<CPlate> plates_blue;
  plates_blue.reserve(64);
  vector<CPlate> plates_yellow;
  plates_yellow.reserve(64);

  Mat src_clone = src.clone();

  Mat src_b_blue;
  Mat src_b_yellow;
#pragma omp parallel sections
  {
#pragma omp section
    {
      colorSearch(src, BLUE, src_b_blue, rects_color_blue);
      //*leijun
      //colorSearch()主要是根据上文介绍的HSV模型，进行相关颜色定位，然后依据常见的图像处理方法进行处理，例如阈值分割，形态学处理和轮廓查找等
      //colorMatch()函数比较复杂，读者可以简单理解为用inRange函数对图像hsv空间进行处理，得到颜色过滤后的图像。(其实colotMatch函数中对hsv模型中的s和v根据h的值进行自适应变化)，进行阈值分割后，采用了形态学图像处理，内核为一个
      //10X2矩形，需要注意的是，内核的大小对最终的结果有很大的影响。对寻找到的轮廓，先进性尺寸验证，不符合尺寸的轮廓直接去除。尺寸验证调用函数
      //verifySizes()
      //。尺寸验证函数主要是对轮廓的长度和宽度，还有长宽比做了限制，以过滤掉大部分的明显非车牌的轮廓区域。
      deskew(src, src_b_blue, rects_color_blue, plates_blue, true, BLUE);
      //*leijun  偏斜扭转
    }
#pragma omp section
    {
      colorSearch(src_clone, YELLOW, src_b_yellow, rects_color_yellow);
      deskew(src_clone, src_b_yellow, rects_color_yellow, plates_yellow, true, YELLOW);
    }
  }

  candPlates.insert(candPlates.end(), plates_blue.begin(), plates_blue.end());
  candPlates.insert(candPlates.end(), plates_yellow.begin(), plates_yellow.end());

  return 0;
}


//! MSER plate locate
int CPlateLocate::plateMserLocate(Mat src, vector<CPlate> &candPlates, int img_index) {
  std::vector<Mat> channelImages;
  std::vector<Color> flags;
  flags.push_back(BLUE);
  flags.push_back(YELLOW);

  bool usePlateMser = false;
  int scale_size = 1000;
  //int scale_size = CParams::instance()->getParam1i();
  double scale_ratio = 1;

  // only conside blue plate
  if (1) {
    Mat grayImage;
    cvtColor(src, grayImage, COLOR_BGR2GRAY);
    channelImages.push_back(grayImage);
  }

  for (size_t i = 0; i < channelImages.size(); ++i) {
    vector<vector<RotatedRect>> plateRRectsVec;
    vector<vector<CPlate>> platesVec;
    vector<Mat> src_b_vec;

    Mat channelImage = channelImages.at(i);   
    Mat image = scaleImage(channelImage, Size(scale_size, scale_size), scale_ratio);

    // vector<RotatedRect> rects;
    mserSearch(image, src_b_vec, platesVec, usePlateMser, plateRRectsVec, img_index, false);

    for (size_t j = 0; j < flags.size(); j++) {
      vector<CPlate>& plates = platesVec.at(j);
      Mat& src_b = src_b_vec.at(j);
      Color color = flags.at(j);

      vector<RotatedRect> rects_mser;
      rects_mser.reserve(64);
      std::vector<CPlate> deskewPlate;
      deskewPlate.reserve(64);
      std::vector<CPlate> mserPlate;
      mserPlate.reserve(64);

      // deskew for rotation and slope image
      for (auto plate : plates) {
        RotatedRect rrect = plate.getPlatePos();
        RotatedRect scaleRect = scaleBackRRect(rrect, (float)scale_ratio);
        plate.setPlatePos(scaleRect);
        plate.setPlateColor(color);

        rects_mser.push_back(scaleRect);
        mserPlate.push_back(plate);
      }

      Mat resize_src_b;
      resize(src_b, resize_src_b, Size(channelImage.cols, channelImage.rows));

      deskew(src, resize_src_b, rects_mser, deskewPlate, false, color);

      for (auto dplate : deskewPlate) {
        RotatedRect drect = dplate.getPlatePos();
        Mat dmat = dplate.getPlateMat();

        for (auto splate : mserPlate) {
          RotatedRect srect = splate.getPlatePos();
          float iou = 0.f;
          bool isSimilar = computeIOU(drect, srect, src.cols, src.rows, 0.95f, iou);
          if (isSimilar) {
            splate.setPlateMat(dmat);
            candPlates.push_back(splate);
            break;
          }
        }
      }
    }
  }

  if (0) {
    imshow("src", src);
    waitKey(0);
    destroyWindow("src");
  }

  return 0;
}

int CPlateLocate::sobelOperT(const Mat &in, Mat &out, int blurSize, int morphW,
                             int morphH) {
  Mat mat_blur;
  mat_blur = in.clone();
  GaussianBlur(in, mat_blur, Size(blurSize, blurSize), 0, 0, BORDER_DEFAULT);

  Mat mat_gray;
  if (mat_blur.channels() == 3)
    cvtColor(mat_blur, mat_gray, CV_BGR2GRAY);
  else
    mat_gray = mat_blur;

  utils::imwrite("resources/image/tmp/grayblure.jpg", mat_gray);

  // equalizeHist(mat_gray, mat_gray);

  int scale = SOBEL_SCALE;
  int delta = SOBEL_DELTA;
  int ddepth = SOBEL_DDEPTH;

  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  Sobel(mat_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(grad_x, abs_grad_x);

  Mat grad;
  addWeighted(abs_grad_x, 1, 0, 0, 0, grad);

  utils::imwrite("resources/image/tmp/graygrad.jpg", grad);

  Mat mat_threshold;
  double otsu_thresh_val =
      threshold(grad, mat_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
  //*leijun 阈值分割

  utils::imwrite("resources/image/tmp/grayBINARY.jpg", mat_threshold);

  Mat element = getStructuringElement(MORPH_RECT, Size(morphW, morphH));
  morphologyEx(mat_threshold, mat_threshold, MORPH_CLOSE, element);
  //*leijun 形态学闭操作

  utils::imwrite("resources/image/tmp/phologyEx.jpg", mat_threshold);

  out = mat_threshold;

  return 0;
}

int CPlateLocate::plateSobelLocate(Mat src, vector<CPlate> &candPlates,
                                   int index) {
  vector<RotatedRect> rects_sobel_all;
  rects_sobel_all.reserve(256);

  vector<CPlate> plates;
  plates.reserve(32);

  vector<Rect_<float>> bound_rects;
  bound_rects.reserve(256);

  sobelFrtSearch(src, bound_rects);
  //*leijun sobelFrtSearch()函数中通过 sobelOper() 进行sobel定位，主要步骤如下：
  //
  //1、对图像进行高斯滤波，为Sobel算子计算去除干扰噪声；
  //
  //2、图像灰度化，提高运算速度；
  //
  //3、对图像进行Sobel运算，得到图像的一阶水平方向导数；
  //
  //4、通过otsu进行阈值分割；
  //
  //5、通过形态学闭操作，连接车牌区域。
  //
  //此处通过Sobel算子进行车牌定位，仅仅做水平方向求导，而不做垂直方向求导。这样做的意义是，如果做了垂直方向求导，会检测出很多水平边缘。水平边缘多也许有利于生成更精确的轮廓，但是由于有些车子前端太多的水平边缘了，例如车头排气孔，标志等等，很多的水平边缘会误导我们的连接结果，导致我们得不到一个恰好的车牌位置。 

  vector<Rect_<float>> bound_rects_part;
  bound_rects_part.reserve(256);

  // enlarge area 
  for (size_t i = 0; i < bound_rects.size(); i++) {
    float fRatio = bound_rects[i].width * 1.0f / bound_rects[i].height;
    if (fRatio < 3.0 && fRatio > 1.0 && bound_rects[i].height < 120) {
      Rect_<float> itemRect = bound_rects[i];

      itemRect.x = itemRect.x - itemRect.height * (4 - fRatio);
      if (itemRect.x < 0) {
        itemRect.x = 0;
      }
      itemRect.width = itemRect.width + itemRect.height * 2 * (4 - fRatio);
      if (itemRect.width + itemRect.x >= src.cols) {
        itemRect.width = src.cols - itemRect.x;
      }

      itemRect.y = itemRect.y - itemRect.height * 0.08f;
      itemRect.height = itemRect.height * 1.16f;

      bound_rects_part.push_back(itemRect);
    }
  }

  // second processing to split one
#pragma omp parallel for
  for (int i = 0; i < (int)bound_rects_part.size(); i++) {
    Rect_<float> bound_rect = bound_rects_part[i];
    Point2f refpoint(bound_rect.x, bound_rect.y);

    float x = bound_rect.x > 0 ? bound_rect.x : 0;
    float y = bound_rect.y > 0 ? bound_rect.y : 0;

    float width =
        x + bound_rect.width < src.cols ? bound_rect.width : src.cols - x;
    float height =
        y + bound_rect.height < src.rows ? bound_rect.height : src.rows - y;

    Rect_<float> safe_bound_rect(x, y, width, height);
    Mat bound_mat = src(safe_bound_rect);

    vector<RotatedRect> rects_sobel;
    rects_sobel.reserve(128);
    sobelSecSearchPart(bound_mat, refpoint, rects_sobel);
    //*leijun
    //为了进一步提高搜索的准确性，EasyPR里面对第一次搜索出的矩形扩大一定范围后，进行了二次搜素，具体函数为
    //sobelSecSearchPart() 。sobelSecSearchPart() 函数和 sobelFrtSearch()
    //大致过程是类似的，此处不再详细叙述，唯一的不同是sobelSecSearchPart()
    //对车牌上铆钉的去除进行了对应的处理。之后对定位区域进行偏斜扭转
    //deskew()处理之后，即可得到车牌定位的结果。

#pragma omp critical
    {
      rects_sobel_all.insert(rects_sobel_all.end(), rects_sobel.begin(), rects_sobel.end());
    }
  }

#pragma omp parallel for
  for (int i = 0; i < (int)bound_rects.size(); i++) {
    Rect_<float> bound_rect = bound_rects[i];
    Point2f refpoint(bound_rect.x, bound_rect.y);

    float x = bound_rect.x > 0 ? bound_rect.x : 0;
    float y = bound_rect.y > 0 ? bound_rect.y : 0;

    float width =
        x + bound_rect.width < src.cols ? bound_rect.width : src.cols - x;
    float height =
        y + bound_rect.height < src.rows ? bound_rect.height : src.rows - y;

    Rect_<float> safe_bound_rect(x, y, width, height);
    Mat bound_mat = src(safe_bound_rect);

    vector<RotatedRect> rects_sobel;
    rects_sobel.reserve(128);
    sobelSecSearch(bound_mat, refpoint, rects_sobel);

#pragma omp critical
    {
      rects_sobel_all.insert(rects_sobel_all.end(), rects_sobel.begin(), rects_sobel.end());
    }
  }

  Mat src_b;
  sobelOper(src, src_b, 3, 10, 3);

  deskew(src, src_b, rects_sobel_all, plates);
  //*leijun 偏斜扭转 `
  //通过颜色定位和Sobel算子定位可以计算出一个个的矩形区域，这些区域都是潜在车牌区域，但是在进行SVM判别是否是车牌之前，还需要进行一定的处理。主要是考虑到以下几个问题：
  //
  //1、定位区域存在一定程度的倾斜，需要旋转到正常视角；
  //
  //2、定位区域存在偏斜，除了进行旋转之后，还需要进行仿射变换；
  //
  //3、定位出区域的大小不一致，需要对车牌的尺寸进行统一。

  //for (size_t i = 0; i < plates.size(); i++) 
  //  candPlates.push_back(plates[i]);

  candPlates.insert(candPlates.end(), plates.begin(), plates.end());

  return 0;
}


int CPlateLocate::plateLocate(Mat src, vector<Mat> &resultVec, int index) {
  vector<CPlate> all_result_Plates;

  plateColorLocate(src, all_result_Plates, index);
  plateSobelLocate(src, all_result_Plates, index);
  plateMserLocate(src, all_result_Plates, index);

  for (size_t i = 0; i < all_result_Plates.size(); i++) {
    CPlate plate = all_result_Plates[i];
    resultVec.push_back(plate.getPlateMat());
  }

  return 0;
}

int CPlateLocate::plateLocate(Mat src, vector<CPlate> &resultVec, int index) {
  vector<CPlate> all_result_Plates;

  plateColorLocate(src, all_result_Plates, index);
  plateSobelLocate(src, all_result_Plates, index);
  plateMserLocate(src, all_result_Plates, index);

  for (size_t i = 0; i < all_result_Plates.size(); i++) {
    resultVec.push_back(all_result_Plates[i]);
  }

  return 0;
}

}
