#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "highgui.h"
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>
#include <pthread.h>

#include "includes/helperfunctions.h"

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h> // defines Surf64Vocabulary
#include "DLoopDetector.h" // defines Surf64LoopDetector
#include <DUtilsCV/DUtilsCV.h> // defines macros CVXX

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
// Demo
#include "includes/demoDetector.h"


using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;
using namespace cv;

#define PI 3.14159265

// Directory of Images
string IMG_DIR = "/media/Heavy_dataset/CAIR_final_dataset/exp3_images/";
//string IMG_DIR = "/home/tushar/dataset/CAIR_review_4/dataset_cair_final/exp_12_10_2_sync_imgs/";

static const string image_dir_firstloop="/media/Heavy_dataset/CAIR_final_dataset/exp4_images/";
//static const string image_dir_firstloop="/home/tushar/dataset/CAIR_review_4/dataset_cair_final/exp12101-new/exp_12_10_1_sync_images/";
// DLoop resources
static const string VOC_FILE = "./resources/iiit2_voc.voc.gz";
//static const string IMAGE_DIR = IMG_DIR1 + "left/";
static const int IMAGE_W = 640; // image size
static const int IMAGE_H = 480;
gtsam::Values initial;
gtsam::Values initialunopt;
gtsam::Pose3 refPose; 
gtsam::Pose3 currentPose; 
int latestkey=0;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
void loadLoop(gtsam::NonlinearFactorGraph &dataset, gtsam::ISAM2 &isam2);
void my_dloop(std::vector<Matrix> &m,std::vector<int> &index1, std::vector<int> &index2,string file, string img_dir, int a, int b, gtsam::ISAM2 &isam2,gtsam::NonlinearFactorGraph &nfg, string image_dir_firstloop);
/// This functor extracts SURF64 descriptors in the required format
class SurfExtractor: public FeatureExtractor<FSurf64::TDescriptor>
{
public:
  /** 
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<vector<float> > &descriptors) const;
};

// Struct for passing variables to viso thread
struct for_libviso_thread
{
  int numimages;
  string imgdir;
  gtsam::ISAM2 *isam2;
  gtsam::NonlinearFactorGraph *nfg;
};

// Struct for passing variables to Dloop thread
struct for_dloop_thread
{
  std::vector<Matrix> mat;
  std::vector<int> i1;
  std::vector<int> i2;
  gtsam::ISAM2 *isam2;
  gtsam::NonlinearFactorGraph *nfg;
};

// Viso Thread
 void *libviso_thread(void *t)
 {
    cout << "entering libviso2 thread!" << endl;
    for_libviso_thread* obj = (for_libviso_thread *) t;

    // Calling libviso function
    my_libviso2(obj->imgdir,obj->numimages,*(obj->isam2),*(obj->nfg));
    
    cout << "exiting libviso thread!!" << endl;
    return (void *)obj;
 }

// DLoop Thread
 void *dloop_thread(void *t)
{
    cout << "entered Dloop thread! "<< endl;
    for_dloop_thread* obj = (for_dloop_thread *) t;

    // Calling Dloop function
    my_dloop(obj->mat,obj->i1,obj->i2,VOC_FILE,IMG_DIR,IMAGE_W,IMAGE_H,*(obj->isam2),*(obj->nfg),image_dir_firstloop);

    cout << "exiting Dloop thread!!" << endl;
    return (void *)obj;
 }

// For single loop
int main()
{
    gtsam::ISAM2Params params;
    params.optimizationParams = gtsam::ISAM2DoglegParams();
    params.relinearizeSkip = 1;
    params.enablePartialRelinearizationCheck = true;
    gtsam::ISAM2 isam2(params);
    gtsam::NonlinearFactorGraph nfg;
    string g2ofilename = "3doptimisedloopexp4.g2o";
    std::pair<gtsam::NonlinearFactorGraph::shared_ptr, gtsam::Values::shared_ptr> data = gtsam::load3D(g2ofilename);
    //initial=*(data.second);
    //initial.print();
    nfg = *(data.first);
    std::cout<<"dataset size "<<nfg.size()<<std::endl;
    loadLoop(nfg,isam2);
std::cout<<"load loop over"<<std::endl;
std::cout<<isam2.size()<<std::endl;
    //gtsam::Values estimate(isam2.calculateEstimate());
    initial=isam2.calculateBestEstimate();
    initial.print();
    initialunopt.insert(initial);
    //std::cout<<"error "<<nfg.error(*(data.second))<<std::endl;
    //std::cout<<"error "<<nfg.error(estimate)<<std::endl;


    
    
  pthread_t thread[2];
  pthread_attr_t attr;

  /* Initialize and set thread detached attribute */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  // Number of images in given directory
  int numImages1 =0;
  std::string path1 = IMG_DIR + "/left/";
  char * dst1 = new char[path1.length() + 1];
  std::strcpy(dst1,path1.c_str());

  numImages1 = listdir(dst1);

  for_libviso_thread lib;
  for_dloop_thread dthread;

  lib.imgdir = IMG_DIR;
  lib.numimages = numImages1;
  lib.isam2 = &isam2;
  lib.nfg = &nfg;

  dthread.isam2 = &isam2;
  dthread.nfg = &nfg;
  // Create Dloop and Viso Thread 
  pthread_create(&thread[0], &attr, dloop_thread, (void *)&dthread);
  pthread_create(&thread[1], &attr, libviso_thread, (void *)&lib);  
  
  // Wait for Completion
  pthread_join(thread[0],NULL);
  pthread_join(thread[1],NULL);

  const string outputfile="isam.g2o";
  const string outputfile2="isamnfg.g2o";
  const string outputfile3="nfg.g2o";
  const string outputfile4="unopt-batch.g2o";
    gtsam::Values estimate2(isam2.calculateEstimate());
    gtsam::NonlinearFactorGraph datasetempty;
    gtsam::writeG2o(datasetempty,estimate2,outputfile);
    gtsam::writeG2o(nfg,gtsam::Values(),outputfile3);
    gtsam::writeG2o(nfg,estimate2,outputfile2);
    gtsam::writeG2o(nfg,initialunopt,outputfile4);
  cout << "done" << endl;

  return 0;
}

// Load into isam2 object function
// ----------------------------------------------------------------------------
void loadLoop(gtsam::NonlinearFactorGraph &dataset, gtsam::ISAM2 &isam2){
      // First time step (time from which to process the data file (set this to zero))
  int firstStep = 0;
  // Last time step (time up to which to process the data file (set this to -1))
  int lastStep = -1;

  // Looking for the first observation
  size_t nextObs = 0;
  bool havePreviousPose = false;
  gtsam::Key firstPose;
  // As long as there is an observation to be added to the ISAM problem
  while(nextObs < dataset.size()){

    if(gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr observation = 
      boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3> >(dataset[nextObs])){
      
      // Get the keys from the observation
      gtsam::Key key1 = observation->key1();
      gtsam::Key key2 = observation->key2();
      
      if((key1 >= firstStep && key1 < key2) || (key2 >= firstStep && key2 < key1)) {
        // This is an odometry starting at firstStep
        firstPose = std::min(key1, key2);
        break;
      }

      if((key2 >= firstStep && key1 < key2) || (key1 >= firstStep && key2 < key1)) {
        // This is an odometry connecting firstStep with a previous pose
        havePreviousPose = true;
        firstPose = std::max(key1, key2);
        break;
      }

    }

    ++ nextObs;

  }

  if (nextObs == dataset.size()){
    std::cout << "The supplied first step is past the end of the dataset." << std::endl;
    exit(1);
  }

  // If we didn't find an odometry linking to a previous pose, create a first pose and a prior
  if(!havePreviousPose){

    gtsam::NonlinearFactorGraph newFactors;
    gtsam::Values newVariables;
    //gtsam::Vector v(3);
    //v<<0.01, 0.01, 0.01;
    //gtsam::SharedDiagonal priorNoise = gtsam::noiseModel::Diagonal::Sigmas(v);
    newFactors.push_back(boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(firstPose, gtsam::Pose3(), gtsam::noiseModel::Unit::Create(gtsam::Pose3::dimension)));
    newVariables.insert(firstPose, gtsam::Pose3());
    // Update the ISAM2 problem
    gtsam::ISAM2Result res =  isam2.update(newFactors, newVariables);
    std::cout<<"first factor update "<<std::endl;  
  }
 
  int dsSize=dataset.size();
  for(int d=nextObs;d<dsSize;d++){
         gtsam::NonlinearFactor::shared_ptr obsFactor = dataset[d];
      if(gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr observation = 
        boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(obsFactor)){
          std::cout<<"obs1 obs2    "<<observation->key1()<<"  "<<observation->key2()<<std::endl;
                gtsam::Values newVariables;
                gtsam::NonlinearFactorGraph newFactors;
                newFactors.push_back(observation);
                
                if(observation->key1() > observation->key2()){
                  if(!isam2.valueExists(observation->key1()) ){                     
                    if(!isam2.valueExists(observation->key2())){
                        std::cout<<"neither exist"<<std::endl;
                    }
                    if(d==nextObs){
                      newVariables.insert(observation->key1(), observation->measured().inverse());
                    }
                    else{
                      gtsam::Pose3 previousPose = isam2.calculateEstimate<gtsam::Pose3>(observation->key2());
                      newVariables.insert(observation->key1(), previousPose * observation->measured().inverse());
                    }
                  }
                }
                else {
                  if(!isam2.valueExists(observation->key2()) ){
                    if(!isam2.valueExists(observation->key1())){
                        std::cout<<"neither exist"<<std::endl;
                    }
                    if (d==nextObs){
                      newVariables.insert(observation->key2(), observation->measured());
                    }
                    else{
                      gtsam::Pose3 previousPose = isam2.calculateEstimate<gtsam::Pose3>(observation->key1());
                      newVariables.insert(observation->key2(), previousPose * observation->measured());
                    }
                  }
                }
                if(d==dsSize-1)
                {
                    gtsam::ISAM2Result res1 = isam2.update(newFactors, newVariables,gtsam::FactorIndices(),boost::none,boost::none,boost::none, true);
                    std::cout<<"first kind of update"<<std::endl;
                    std::cout<<"relin-ed = "<< res1.getVariablesRelinearized() <<std::endl;
                }
                else{
                    gtsam::ISAM2Result res1 =  isam2.update(newFactors, newVariables);
                    std::cout<<"second kind of update"<<std::endl;
                    std::cout<<"relin-ed = "<< res1.getVariablesRelinearized() <<std::endl;
                }
      }
      else{
                   std::cout<<"couldnt cast"<<std::endl;
      }
  }
      //gtsam::Values estimate(isam2.calculateEstimate());
}






// DLoop helper functions
// ----------------------------------------------------------------------------

void SurfExtractor::operator() (const cv::Mat &im, 
  vector<cv::KeyPoint> &keys, vector<vector<float> > &descriptors) const
{
  // extract surfs with opencv
   static cv::Ptr<cv::xfeatures2d::SURF> surf_detector = 
    cv::xfeatures2d::SURF::create(400);
  
  surf_detector->setExtended(false);
  
  keys.clear(); // opencv 2.4 does not clear the vector
  vector<float> plain;
  surf_detector->detectAndCompute(im, cv::Mat(), keys, plain);
  
  // change descriptor format
  const int L = surf_detector->descriptorSize();
  descriptors.resize(plain.size() / L);

  unsigned int j = 0;
  for(unsigned int i = 0; i < plain.size(); i += L, ++j)
  {
    descriptors[j].resize(L);
    std::copy(plain.begin() + i, plain.begin() + i + L, descriptors[j].begin());
  }
}

// ----------------------------------------------------------------------------

void my_dloop(std::vector<Matrix> &Mat,std::vector<int> &index1, std::vector<int> &index2,string VOC_FILE1,string IMAGE_DIR, int IMAGE_W1, int IMAGE_H1, gtsam::ISAM2 &isam2,gtsam::NonlinearFactorGraph &nfg,string image_dir_firstloop)
{
   demoDetector<Surf64Vocabulary, Surf64LoopDetector, FSurf64::TDescriptor>
    demo(VOC_FILE1, IMAGE_DIR, IMAGE_W1, IMAGE_H1,image_dir_firstloop);
  try 
  {  
    // run the demo with the given functor to extract features
    SurfExtractor extractor;
    int firstloopsize=0;
    Surf64LoopDetector detector = demo.runPreSession("SURF64",extractor,firstloopsize);
    std::cout<<"filenames size is "<<firstloopsize<<std::endl;  //need to check this
    demo.runSession("SURF64",extractor,isam2,nfg,detector,firstloopsize);
  }
  catch(const std::string &ex)
  {
    cout << "Error: " << ex << endl;
  }
}

