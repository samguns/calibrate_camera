#include <iostream>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>

static int NX;
static int NY;
static int w;
static int h;
static bool verbose = false;

using namespace std;
using namespace cv;


static bool findPoints(vector<Point3f> objp, vector<vector<Point3f>>& objpoints, vector<vector<Point2f>>& imgpoints) {
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
      drawChessboardCorners(img, pattern, corners, ret);
      imgpoints.push_back(corners);
      objpoints.push_back(objp);
      if (verbose) {
        cout << "Found chessboard corners on: " << p << endl;
        namedWindow("Window", WINDOW_AUTOSIZE);
        imshow("Window", img);
        waitKey(0);
      }
    }

  }

  return true;
}

static void calibrate_and_save(vector<vector<Point3f>>& objpoints,
                               vector<vector<Point2f>>& imgpoints,
                               Size img_size) {
  Mat intrinsic = Mat(3, 3, CV_32FC1);
  Mat distCoeffs;
  vector<Mat> rvecs, tvecs;

  intrinsic.ptr<float>(0)[0] = 1;
  intrinsic.ptr<float>(1)[1] = 1;

  calibrateCamera(objpoints, imgpoints, img_size, intrinsic,
          distCoeffs, rvecs, tvecs);

  FileStorage fs("cal_params.yml", FileStorage::WRITE);
  fs << "mtx" << intrinsic;
  fs << "dist" << distCoeffs;
  fs.release();

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

    if (verbose) {
      Mat result(Size(w*2, h), CV_8UC3);
      img.copyTo(result(Rect(0, 0, w, h)));
      undist.copyTo(result(Rect(w, 0, w, h)));
      imshow("Result", result);
      waitKey(0);
    }
  }
}

static void show_usage(string name) {
  cerr << "Usage: " << name << " <options>"
       << "Options:\n"
       << "\t-w,    Image width\n"
       << "\t-h,    Image height\n"
       << "\t-nx,   Black grids from left to right\n"
       << "\t-ny,   Black grids from top to bottom\n"
       << "\t-d,    Show debug images\n"
       << endl;
}

static void parse_arguments(int argc, char *argv[]) {
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    int value = atoi(argv[i+1]);
    if (arg == "-w") {
      w = value;
      i++;
    } else if (arg == "-h") {
      h = value;
      i++;
    } else if (arg == "-nx") {
      NX = value;
      i++;
    } else if (arg == "-ny") {
      NY = value;
      i++;
    } else if (arg == "-d") {
      if (value != 0) {
        verbose = true;
        i++;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  vector<vector<Point3f>> object_points;
  vector<vector<Point2f>> image_points;
  vector<Point3f> objp;

  if (argc < 9) {
    show_usage(argv[0]);
    return -1;
  }

  parse_arguments(argc, argv);

  Size img_size(w, h);
  cout << "width " << img_size.width << " height " << img_size.height << endl;

  for (int i = 0; i < (NX * NY); i++) {
    objp.emplace_back(i / NX, i % NX, 0.0f);
  }

  findPoints(objp, object_points, image_points);

  calibrate_and_save(object_points, image_points, img_size);

  return 0;
}
