#include "scrnn.h"

#include <time.h>

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dnn;
// g++ -O2 scrnn.cpp `pkg-config opencv4 --cflags --libs`

// copied example from the OpenCV
// https://www.ccoderun.ca/programming/doxygen/opencv_3.2.0/tutorial_gpu_basics_similarity.html
double getPSNR(const Mat& I1, const Mat& I2) {
  Mat s1;
  absdiff(I1, I2, s1);       // |I1 - I2|
  s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
  s1 = s1.mul(s1);           // |I1 - I2|^2
  Scalar s = sum(s1);        // sum elements per channel
  double sse = s.val[0] + s.val[1] + s.val[2];  // sum channels
  double rmse = sqrt(sse / (double)(I1.channels() * I1.total() * 255));

  if (rmse == 0)  // for small values    zero
    return 100;
  else {
    double psnr = 20.0 * log10(1.0 / rmse);
    return psnr;
  }
}
// copied example from the OpenCV
// https://www.ccoderun.ca/programming/doxygen/opencv_3.2.0/tutorial_gpu_basics_similarity.html
Scalar getMSSIM(const Mat& i1, const Mat& i2) {
  const double C1 = 6.5025, C2 = 58.5225;
  /***************************** INITS **********************************/
  int d = CV_32F;
  Mat I1, I2;
  i1.convertTo(I1, d);  // cannot calculate on one byte large values
  i2.convertTo(I2, d);
  Mat I2_2 = I2.mul(I2);   // I2^2
  Mat I1_2 = I1.mul(I1);   // I1^2
  Mat I1_I2 = I1.mul(I2);  // I1 * I2
  /*************************** END INITS **********************************/
  Mat mu1, mu2;  // PRELIMINARY COMPUTING
  GaussianBlur(I1, mu1, Size(11, 11), 1.5);
  GaussianBlur(I2, mu2, Size(11, 11), 1.5);
  Mat mu1_2 = mu1.mul(mu1);
  Mat mu2_2 = mu2.mul(mu2);
  Mat mu1_mu2 = mu1.mul(mu2);
  Mat sigma1_2, sigma2_2, sigma12;
  GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
  sigma1_2 -= mu1_2;
  GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
  sigma2_2 -= mu2_2;
  GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
  sigma12 -= mu1_mu2;
  Mat t1, t2, t3;
  t1 = 2 * mu1_mu2 + C1;
  t2 = 2 * sigma12 + C2;
  t3 = t1.mul(t2);  // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
  t1 = mu1_2 + mu2_2 + C1;
  t2 = sigma1_2 + sigma2_2 + C2;
  t1 = t1.mul(t2);  // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
  Mat ssim_map;
  divide(t3, t1, ssim_map);       // ssim_map =  t3./t1;
  Scalar mssim = mean(ssim_map);  // mssim = average of ssim map
  return mssim;
}

int main() {
  clock_t tStart = clock();
  std::vector<std::string> data_paths{"./Set5/original/*.png",
                                      "./Set14/original/*png"};
  std::vector<std::string> data_names{"Set5", "Set14"};
  Net model = readNetFromONNX("model.onnx");
  int it = 0;
  for (auto& data_path : data_paths) {
    vector<cv::String> fn;
    glob(data_path, fn, false);
    vector<Mat> images;
    size_t count = fn.size();  // number of png files in images folder
    double pnsr = 0;
    double ssim = 0;
    for (size_t i = 0; i < count; i++) images.push_back(imread(fn[i]));
    for (auto& image : images) {
      image.convertTo(image, CV_32FC3);
      int image_height = image.cols;
      int image_width = image.rows;

      // create blob from image
      Mat blob = blobFromImage(image, 1, Size(image_height, image_width),
                               cv::Scalar(), false);

      // create blob from image
      model.setInput(blob);
      // forward pass through the model to carry out the detection
      Mat pred = model.forward();
      std::vector<Mat> output;
      imagesFromBlob(pred, output);

      pnsr += getPSNR(image, output[0]);
      //  imwrite("result.png", output[0]);
    }

    cout << data_names[it] << " pnsr: " << pnsr / data_paths.size() << "\n";
    it++;
  }
  return 0;
}