#include <stdio.h>
#include <string>
#include <vector>
#include <map>
#include <list>
#include <iostream>
// #include <time.h>
#include <fstream>
#include <cstring>
#include <cmath>
//#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <mxnet/c_predict_api.h>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>
using namespace std;
using namespace cv;

//#include "LFQueue.hpp"
#include "MTCNN.hpp"
#include "Feature.hpp"
//#include "CameraSettings.h"
#include "clustering.h"

//#define DISPLAY_ON


string CanvasName = "Camera";

char ConfigPath[30] = "Config.txt";

bool isFileExist(char* fname){
	struct stat bff;
	return ((stat(fname, &bff))==0);
}

void SetConfigParameter(std::map <std::string, std::string> &Configs)
{
  Configs["Model_Dir"] = "../model/";
  Configs["Model_Name"] =  "EFM_RES";
  Configs["Extract_IMG_Size"] =  "128";
  Configs["Feature_Vector_Size"] = "342";	//dim = 342 for face images, 682 for iris imgs
  Configs["Feature_Layer"] = "drop1";
  Configs["Total_IMG_Num"] = "2";
}

static float CosineDistance(vector<float>& f1, vector<float>&f2)
{
  if (f1.size() != f2.size() || f1.size() == 0 || f2.size() == 0)
    return 0.0f;

  double n1, n2, c;
  n1 = n2 = c = 0.0;
  for (int i = 0; i < f1.size(); i ++) {
    double v1 = f1[i];
    double v2 = f2[i];
    n1 += v1*v1;
    n2 += v2*v2;
    c += v1*v2;
  }
  if (fabs(n1) < 1e-8 || fabs(n2) < 1e-8) {
    return 0.0f;
  }

  return float(c/sqrt(n1*n2));
}



vector<string> read_data(string filename){
  vector<string> tmp;
  string path;
  ifstream Img_Path_File(filename);
  while (getline (Img_Path_File, path)) {
    tmp.push_back(path);
  }
  Img_Path_File.close();
  return tmp;
}


static bool file_not_open = true;

void store_feature_vector(ofstream& Feature_File, vector<float>& feature, string& id)
{
  if (Feature_File.is_open()){
    //Feature_File << "[" ;
    for(int i = 0; i < feature.size(); ++i){
      Feature_File << feature[i];
      if(i != feature.size() - 1) 
        Feature_File << ",";  
    }
    //Feature_File << "], " << id << endl;  
    Feature_File << endl;
  }
}


int main(int argc, char* argv[])
{
  ofstream fpslog("FPS.txt");
  PredictorHandle pred_hnd;
  std::map <std::string, std::string> Configs = Get_Parameter(ConfigPath);
  SetConfigParameter(Configs);

  string model_dir = Configs["Model_Dir"].c_str();
  int cap_window_width = atoi(Configs["Width"].c_str());
  int cap_window_height = atoi(Configs["Height"].c_str());
  int mini_FaceSize = atoi(Configs["mini_FaceSize"].c_str());

  struct stat dbfile;

 
  list<vector<float>> regFacialFeature1;
  list<vector<Point2d>> regFLandmarks1;
  list<vector<float>> regFacialFeature2;
  list<int> clusterID1, clusterID2;
  list<int> selectedImage1, selectedImage2;
  list<Mat> regFaceImage1, regFaceImage2, regOrgImage;

  char name[256];
  name[0] = '_';
  name[1] = 0;
  int idx = 0;
  vector<Mat> regImages;
  vector<vector<float>> regFeatures;
  vector<vector<Point2d>> regFLandmarks;
  vector<int> regValidImage;
  Mat SelFace;
  vector<float> SelFeature;

  std::string PNet_json_file = model_dir + "/det1-symbol.json";
  std::string PNet_param_file = model_dir + "/det1-0001.params";
  std::string RNet_json_file = model_dir + "/det2-symbol.json";
  std::string RNet_param_file = model_dir + "/det2-0001.params";
  std::string ONet_json_file = model_dir + "/det3-symbol.json";
  std::string ONet_param_file = model_dir + "/det3-0001.params";
  PreLoadPNetPool toFasterMTCNN(PNet_json_file, PNet_param_file);
  BufferFile RNet_json_data(RNet_json_file);
  BufferFile RNet_param_data(RNet_param_file);
  BufferFile ONet_json_data(ONet_json_file);
  BufferFile ONet_param_data(ONet_param_file);

  if(!Load_Identify_model(Configs,pred_hnd)){
    fprintf(stderr, "Fail to load model\n");
    exit(-1);
  }

	//Load Model
  //2D
  dlib::shape_predictor sp;
  dlib::deserialize(model_dir + "/shape_predictor_68_face_landmarks.dat") >> sp;

  Mat IMG, IMG_, IMG_GRAY, Canvas;

  //Detection
  int People_num = 0, Person_num = 0;
  int Face_num;
  vector<cv::Rect> Bounding_Box;
  vector<double*> LMK;
  //ST
  double lmk[68*2];
  cv::Mat Align_IMG, OUTPUT_LMK;

  //Identify
  vector<float> feature(atoi(Configs["Feature_Vector_Size"].c_str()));
  float sim_th = atof(Configs["sim_th"].c_str());

  TimeGoesBy();
  int m_frames = 0;
  double t0, t;

  //Joy starts
  //Store all img names into a vector<string>
  vector<string> img_id = read_data("Img_Ids_iris_train.txt");

  FILE *imgPath = fopen("Img_Paths_iris_train.txt", "rt");
  char imgfname[2048];
  int img_iter = 0;
  ofstream Feature_File;
  Feature_File.open("feature_vector.txt", std::ios::app);

  while (!feof(imgPath)) {
    if (fgets(imgfname, sizeof(imgfname), imgPath) == NULL)
      break;
    
    if (imgfname[strlen(imgfname)-1] == '\n')
      imgfname[strlen(imgfname)-1] = 0;
    
    IMG = imread(imgfname);
#ifdef DISPLAY_ON
    printf("%s: %d x %d\n", imgfname, IMG.rows, IMG.cols);
#endif

    if(!IMG.data){ 
      cout<<"Image not found\n"; 
      exit(-1); 
    }
    
    /*
    IMG_ = IMG.clone();
    toFasterMTCNN.reload(IMG_.rows, IMG_.cols, mini_FaceSize);
    
    //detection
    try{
      Face_num = MTCNN_Dlib_Detection(IMG_, model_dir, sp, Bounding_Box, LMK, toFasterMTCNN, RNet_json_data, RNet_param_data, ONet_json_data, ONet_param_data, 2, 0);
    } catch (Exception e) {
      cout<<"Detectoin Error:\n"<<e.what()<<endl;
      cin.get();
      continue;
    }

#ifdef DISPLAY_ON
    printf("# detected face(s): %d\n", Face_num);
#endif
    
    //Joy here
    double maxBBoxArea = -DBL_MAX;
    int idx = -1;
    for(int i = 0; i < Face_num; i++){
#ifdef DISPLAY_ON
      rectangle(IMG, Bounding_Box[i], Scalar(255,0,0), 3);
#endif
      double area = Bounding_Box[i].area();
      if (area > maxBBoxArea) {
        maxBBoxArea = area;
        idx = i;
      }
    }
    
    // 68 facial landmarks, * 2 points
    if((LMK.size() != 0) && idx != -1){
#ifdef DISPLAY_ON
      rectangle(IMG, Bounding_Box[idx], Scalar(0,0,255), 3);
#endif
      for(int j = 0;j < 68 * 2; j++)
        lmk[j] = LMK[idx][j];
      Align_2D(IMG_, lmk, Align_IMG, OUTPUT_LMK, 128);    //output 128 * 128
#ifdef DISPLAY_ON
      imshow("Aligned", Align_IMG);
#endif
      if (Align_IMG.channels() == 3)
        cv::cvtColor(Align_IMG, IMG_GRAY, CV_BGR2GRAY);
      else
        IMG_GRAY = Align_IMG;

      */
      Feature_Extract_exe(IMG_GRAY, &feature[0], pred_hnd);
      // Store the feature vector to a csv file
      store_feature_vector(Feature_File, feature, img_id[img_iter]);

      cout << "Img ID: " << img_id[img_iter] << endl;
    //}
#ifdef DISPLAY_ON    
    imshow("IMG", IMG);
    waitKey(2000);
#endif
    
    img_iter ++;
  }
  fclose(imgPath);
  Feature_File.close();
 
  MXPredFree(pred_hnd);
  fpslog.close();
	return 0;
}

