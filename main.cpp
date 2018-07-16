#include <iostream>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>

#define NX  9
#define NY  6

using namespace std;
using namespace cv;


static bool findPoints(vector<Point3f> objp, vector<vector<Point3f>>& objpoints, vector<vector<Point2f>>& imgpoints) {
  string path = "./camera_cal";

  for (int i = 0; i < 25; i++) {
    ostringstream os;
    os << "./camera_cal/calibrate" << i << ".jpg";
    string p(os.str());

    Mat img = imread(p.c_str(), IMREAD_COLOR);
    if (img.empty()) {
      continue;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Size pattern(NX, NY);
    vector<Point2f> corners;

    bool ret = findChessboardCorners(gray, pattern, corners,
                                     CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
    if (ret) {
      cout << "Found chessboard corners on: " << p << " size: " << img.size() << endl;
      drawChessboardCorners(img, pattern, corners, ret);
      imgpoints.push_back(corners);
      objpoints.push_back(objp);
      namedWindow("Window", WINDOW_AUTOSIZE);
      imshow("Window", img);
      waitKey(0);
    }

  }

  return true;
}

static void calibrate_and_save(vector<vector<Point3f>>& objpoints, vector<vector<Point2f>>& imgpoints) {
  Mat intrinsic = Mat(3, 3, CV_32FC1);
  Mat distCoeffs;
  vector<Mat> rvecs, tvecs;

  intrinsic.ptr<float>(0)[0] = 1;
  intrinsic.ptr<float>(1)[1] = 1;

  calibrateCamera(objpoints, imgpoints, Size(640, 480), intrinsic,
          distCoeffs, rvecs, tvecs);

  FileStorage fs("cal_params.yaml", FileStorage::WRITE);
  fs << "mtx" << intrinsic;
  fs << "dist" << distCoeffs;

  for (int i = 0; i < 25; i++) {
    ostringstream os;
    os << "./camera_cal/calibrate" << i << ".jpg";
    string p(os.str());

    Mat img = imread(p.c_str(), IMREAD_COLOR);
    if (img.empty()) {
      continue;
    }

    Mat undist;
    undistort(img, undist, intrinsic, distCoeffs);

    Mat result(Size(1280, 480), CV_8UC3);
    img.copyTo(result(Rect(0, 0, 640, 480)));
    undist.copyTo(result(Rect(640, 0, 640, 480)));
    imshow("Result", result);
    waitKey(0);
  }
}

int main() {
  vector<vector<Point3f>> object_points;
  vector<vector<Point2f>> image_points;
  vector<Point3f> objp;

  for (int i = 0; i < (NX * NY); i++) {
    objp.emplace_back(i / NX, i % NX, 0.0f);
  }

  findPoints(objp, object_points, image_points);

  calibrate_and_save(object_points, image_points);

  return 0;
}
