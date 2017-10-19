// To find number of images in a directory

#include <sys/types.h>
#include <dirent.h>
#include "helperfunctions.h"
#include "filewriter.h"
#include "threadcomm.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdint.h>
#include <sstream> 

#include <viso_stereo.h>
//#include <png++/png.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>

// g2o optimization
#include <g2o/core/sparse_optimizer.h>
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <unistd.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/dataset.h>
#include "g2o/core/factory.h"
#include "g2o/stuff/command_args.h"

using namespace std;
using namespace cv;
using namespace g2o;

// we use the 2D and 3D SLAM types here
G2O_USE_TYPE_GROUP(slam2d)
G2O_USE_TYPE_GROUP(slam3d)

#define PI 3.14159265

writer mywriter;
comm mythread_comm;
std::vector<Matrix> Tr_global;

pthread_mutex_t myMutex;

double valx;
double valz;

int listdir(char *dir) {
  struct dirent *dp;
  DIR *fd;
  int counter =0;
  //cout << dir<<endl;

  if ((fd = opendir(dir)) == NULL) {
    fprintf(stderr, "listdir: can't open %s\n", dir);
    return -1;
  }
  while ((dp = readdir(fd)) != NULL) {
  if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, ".."))
    continue;    /* skip self and parent */
  // printf("%s/%s\n", dir, dp->d_name);
  	counter++;
  //printf("%s\n", dp->d_name);
  }
  //cout<< counter<<endl;
  closedir(fd);
  return counter;
}

struct for_g2o_thread
{
  string s1;
  //string outfile;
  int maxIterations;
};

 void *g2o_thread(void *t)
 {
  cout << "entering g2o thread!" << endl;
  for_g2o_thread* obj = (for_g2o_thread *) t;
  stringstream mystring;
  
  //cout << "string" << obj->s1 << endl;
  mystring << obj->s1;
  //cout << "mystring " << mystring.str() << endl;

   // create the linear solver
  BlockSolverX::LinearSolverType * linearSolver = new LinearSolverCSparse<BlockSolverX::PoseMatrixType>();

  // create the block solver on top of the linear solver
  BlockSolverX* blockSolver = new BlockSolverX(linearSolver);

  // create the algorithm to carry out the optimization
  //OptimizationAlgorithmGaussNewton* optimizationAlgorithm = new OptimizationAlgorithmGaussNewton(blockSolver);
  OptimizationAlgorithmLevenberg* optimizationAlgorithm = new OptimizationAlgorithmLevenberg(blockSolver);

  // NOTE: We skip to fix a variable here, either this is stored in the file
  // itself or Levenberg will handle it.

  // create the optimizer to load the data and carry out the optimization
  SparseOptimizer optimizer;
  optimizer.setVerbose(true);
  optimizer.setAlgorithm(optimizationAlgorithm);

  //cout << mystring.str() << endl;

  optimizer.load1(mystring);  

  optimizer.initializeOptimization();
  optimizer.optimize(obj->maxIterations);

  //cout << "saving g2o file" << obj->outfile << " ... ";

  string out = optimizer.save1(obj->s1,0);
  obj->s1 = "";
  mywriter.g2o_string = "";
  mywriter.g2o_string = out;

  cout << "exiting g2o thread!!" << endl;

  return(void*)obj;
 }

void insertIntoIsam(gtsam::ISAM2 &isam2,gtsam::NonlinearFactorGraph &nfg, int id1){
 
    int obs=id1-1;
    gtsam::NonlinearFactorGraph loopFactors;
    gtsam::NonlinearFactor::shared_ptr factor = nfg[obs];
    gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr observation = 
        boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3> >(factor);
    while(observation->key1()!=id1){
                obs++;
                factor = nfg[obs];
                observation = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3> >(factor);
    }
    
    //should have found where id1 is the first key
    //assuming dloop is always lagging behind, this is first odometric measurement for id1
    if(abs( observation->key2() - observation->key1() ) !=1 ){
        throw runtime_error("out of while loop without finding node for id1's first odometry");
    }
    //at obs we have observation.key1 = id1
    int forwardkey=obs;
    int backwardkey=obs-1;
    int nfgsize=nfg.size();
    
    
    
    for(;backwardkey>=mythread_comm.Nfirstloop+1;backwardkey--){
        gtsam::NonlinearFactor::shared_ptr factor = nfg[backwardkey];
        gtsam::Values newVariables;
        gtsam::NonlinearFactorGraph newFactors;
        if(gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr observation = 
            boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3> >(factor))
        {
            std::cout<<" casted"<<std::endl;
            std::cout<<observation->key1()<<" "<<observation->key2()<<std::endl;
            newFactors.push_back(observation);
            
            if(observation->key1()+1== observation->key2()){ 
              //normal odometric measurement up till first cross loop closure(1 case ignored)
              cout<<" observation->key1() < observation->key2()"<<endl;
              if(!isam2.valueExists(observation->key1()) ){                     
                if(!isam2.valueExists(observation->key2())){
                    throw runtime_error("Problem! None exist");
                }

                cout<<"inserting key1"<<endl;
                gtsam::Pose3 previousPose = isam2.calculateEstimate<gtsam::Pose3>(observation->key2());
                newVariables.insert(observation->key1(), previousPose * observation->measured().inverse());
		std::cout<<"BOTH PRINTS FOR COMPARE"<<std::endl;
		std::cout<<"updating for key1 "<<observation->key1()<<std::endl;
		std::cout<<"observation key2 "<<observation->key2()<<std::endl;
		std::cout<<"observation->measured().inverse()*currentPose"<<std::endl;
		(observation->measured().inverse()*currentPose).print();
		std::cout<<"currentPose*observation->measured().inverse()"<<std::endl;
		(currentPose*observation->measured().inverse()).print();
		std::cout<<"latest key was "<<latestkey<<std::endl;
	        if(latestkey!=observation->key2()){throw runtime_error("Problem exists with latest key");}
		currentPose=currentPose*observation->measured().inverse();
		initialunopt.insert(observation->key1(),currentPose);
		latestkey=observation->key1();
		std::cout<<"latest key is now "<<latestkey<<std::endl;
              }
            }
            else {  //loop closure case ( both +5 and real loop closure(when key1>key2) )
                loopFactors.push_back(factor);
                continue;
            }
                      
        }
        cout<<"next line is isamupdate in insertintoisam!!!"<<endl;
        cout<<newVariables.size()<<endl;
        cout<<newFactors.size()<<endl;
        gtsam::ISAM2Result res1 =  isam2.update(newFactors, newVariables,gtsam::FactorIndices(),boost::none,boost::none,boost::none, true);
        std::cout<< "(backward)variables relinearised= "<<res1.variablesRelinearized<<std::endl;
        
    }
    currentPose=refPose;
   
    latestkey=id1;
    std::cout<<"latest key is now "<<latestkey<<std::endl;
    std::cout<<"STARTING FORWARD!!"<<std::endl;
    for(;forwardkey<nfgsize;forwardkey++){
        gtsam::NonlinearFactor::shared_ptr factor = nfg[forwardkey];
        gtsam::Values newVariables;
        gtsam::NonlinearFactorGraph newFactors;
        if(gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr observation = 
            boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3> >(factor))
        {
                std::cout<<"casted"<<std::endl;
		std::cout<<observation->key1()<<" "<<observation->key2()<<std::endl;
                newFactors.push_back(observation);


                if(observation->key1()+1 ==  observation->key2()){
                  //normal odometric measurement  or +5 frame closure
                  cout<<" observation->key1() < observation->key2()"<<endl;
                  if(!isam2.valueExists(observation->key2()) ){                     
                    if(!isam2.valueExists(observation->key1())){
                        throw runtime_error("Problem! None exist");
                    }

			if(observation->key1() +1== observation->key2()){
                            cout<<" inserting key2"<<endl;
                            gtsam::Pose3 previousPose = isam2.calculateEstimate<gtsam::Pose3>(observation->key1());
                            newVariables.insert(observation->key2(), previousPose * observation->measured());

				std::cout<<"BOTH PRINTS FOR COMPARE"<<std::endl;
				std::cout<<"updating for key2 "<<observation->key2()<<std::endl;
				std::cout<<"observation key1 "<<observation->key1()<<std::endl;
				std::cout<<"observation->measured()*currentPose"<<std::endl;
				(observation->measured()*currentPose).print();
				std::cout<<"currentPose*observation->measured()"<<std::endl;
				(currentPose*observation->measured()).print();
				currentPose=currentPose*observation->measured();
	       			std::cout<<"latest key was "<<latestkey<<std::endl;
				if(latestkey!=observation->key1()){throw runtime_error("Problem exists with latest key");}
				initialunopt.insert(observation->key2(),currentPose);
				latestkey=observation->key2();
				std::cout<<"latest key is now "<<latestkey<<std::endl;
			}
                  }
                }
                else {  //loop closure case
                    /**
                  cout<<" observation->key1() > observation->key2()"<<endl;
                  if(!isam2.valueExists(observation->key1()) ){
                    if(!isam2.valueExists(observation->key2())){
                        throw runtime_error("Problem! None exist");
                    }
                    else{
                      cout<<" inserting key1"<<endl;
                      gtsam::Pose3 previousPose = isam2.calculateEstimate<gtsam::Pose3>(observation->key2());
                      newVariables.insert(observation->key1(), previousPose * observation->measured().inverse());
                    }
                  }**/
                                    loopFactors.push_back(factor);
                                                    continue;
                }
                      
        }
        cout<<"next line is isamupdate in insertintoisam!!!"<<endl;
        cout<<newVariables.size()<<endl;
        cout<<newFactors.size()<<endl;
        gtsam::ISAM2Result res1 =  isam2.update(newFactors, newVariables,gtsam::FactorIndices(),boost::none,boost::none,boost::none, true);
        std::cout<< "(forward)variables relinearised= "<<res1.variablesRelinearized<<std::endl;
        
    }
    
    
    //now go over remaining factors
    int vectorsize=loopFactors.size();
    for(int y=0;y<vectorsize;y++){
        gtsam::NonlinearFactorGraph newFactors;
        gtsam::Values newVariables;
        gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr observation = 
                boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3> >(loopFactors[y]);
        newFactors.push_back(loopFactors[y]);
        if(!isam2.valueExists(observation->key1()) ){
                if(!isam2.valueExists(observation->key2()) ){
                    throw runtime_error("Problem! key2 also doesn't exist, so none exist");
                }
                cout<<" inserting key1"<<endl;
                gtsam::Pose3 previousPose = isam2.calculateEstimate<gtsam::Pose3>(observation->key2());
                newVariables.insert(observation->key1(), previousPose * observation->measured().inverse());
        }
        else if(!isam2.valueExists(observation->key2()) ){
                cout<<" inserting key2"<<endl;
                gtsam::Pose3 previousPose = isam2.calculateEstimate<gtsam::Pose3>(observation->key1());
                newVariables.insert(observation->key2(), previousPose * observation->measured());           
        }
        else{
            std::cout<<"(loopward) BOTH EXIST !!!! "<<std::endl;
        }
        
        cout<<"next line is isamupdate in insertintoisam!!!"<<endl;
        cout<<newVariables.size()<<endl;
        cout<<newFactors.size()<<endl;
        gtsam::ISAM2Result res1 =  isam2.update(newFactors, newVariables,gtsam::FactorIndices(),boost::none,boost::none,boost::none, true);
        std::cout<< "(loopward) variables relinearised= "<<res1.variablesRelinearized<<std::endl;
    }

    
    
}

void my_libviso2(string dir, int numImg,gtsam::ISAM2 &isam2,gtsam::NonlinearFactorGraph &nfg) {

    while(!mythread_comm.can_start_viso){
        sleep(5);
    }
    
    std::cout<<"S T A R T I N G     V I S O"<<std::endl;
  // set most important visual odometry parameters
  // for a full parameter list, look at: viso_stereo.h
  VisualOdometryStereo::parameters param;
/**
    param.calib.f  = 722.658813477; // focal length in pixels
      param.calib.cu = 619.283325195; // principal point (u-coordinate) in pixels
        param.calib.cv = 360.0; // principal point (v-coordinate) in pixels
          param.base     = 0.12; // baseline in meters
  **/

/*      param.calib.f  =  699.11900; // focal length in pixels
          param.calib.cu = 669.65700; // principal point (u-coordinate) in pixels
                param.calib.cv = 360.60200; // principal point (v-coordinate) in pixels
                     param.base     = 0.12; // baseline in meters
  **/

        param.calib.f = 567.65218; // focal length in pixels
                param.calib.cu = 679.34969; // principal point (u-coordinate) in pixels
                          param.calib.cv = 370.93921; // principal point (v-coordinate) in pixels
                                   param.base     = 0.12; // baseline in meters

  /**  
  // calibration parameters for sequence 2010_03_09_drive_0019 
  param.calib.f  = 711.9212646484375; // focal length in pixels
  param.calib.cu = 647.0408325195312; // principal point (u-coordinate) in pixels
  param.calib.cv = 360.7899169921875; // principal point (v-coordinate) in pixels
  param.base     = 0.12; // baseline in meters
**/
  pthread_mutex_init(&myMutex,0);
  pthread_t g2othread;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  for_g2o_thread gthread;
  //gthread.outfile = outputFilename;
  gthread.maxIterations = 10;

  mywriter.my_write_file.open("data/offline.g2o");
  string ss;
  
  mythread_comm.loop_wait = false;
  
  // init visual odometry
  VisualOdometryStereo viso(param);
  
  Matrix pose = Matrix::eye(4);
  
  Matrix extra_relations;
  bool is_good = false;
  gtsam::Values estimate2(isam2.calculateEstimate());
  int numNodes=estimate2.size();
  std::cout<<"numnodes = "<<numNodes<<std::endl;   
  
  for (int32_t i=1; i<=numImg; i++) 
  {  
//  if(i==1){
//      try{
//        //init isam2 with prior
//        gtsam::NonlinearFactorGraph newFactors;
//        gtsam::Values newVariables;
//        gtsam::Vector v(3);
//        v<<0.01, 0.01, 0.01;
//        gtsam::SharedDiagonal priorNoise = gtsam::noiseModel::Diagonal::Sigmas(v);
//        newFactors.push_back(boost::make_shared<gtsam::PriorFactor<gtsam::Pose3> >(numNodes+1, gtsam::Pose3(-14.7,-16.7,-1.3), gtsam::noiseModel::Unit::Create(gtsam::Pose3::dimension)));
//        newVariables.insert(numNodes+1, gtsam::Pose3(-14.7,-16.7,-1.3));
//        isam2.update(newFactors, newVariables);
//        cout<<"SUCCESSFULLY UPDATED ISAM2 AFTER ADDING NODE WITH PRIOR"<<endl;
//      }
//      catch(Exception e){
//        std::cout << e.err << '\n'; 
//        std::cout << e.line << '\n'; 
//        std::cout << e.what() << '\n';
//        cerr << "(libviso)ERROR: Couldn't add first pose prior!" << endl;
//        //break;
//        } 
//      }
 
  pthread_mutex_lock(&myMutex);
  mythread_comm.viso_wait_flag = true;
  pthread_mutex_unlock(&myMutex);

  // input file names
  char base_name[256]; sprintf(base_name,"%04d.png",i);
  string left_img_file_name  = dir + "left/" + base_name;
  string right_img_file_name = dir + "right/" + base_name;
  cout << "////////will start Processing: Frame: " << i << endl;
  cout<<"//////////"<<left_img_file_name<<endl;
  // catch image read/write errors here
  try 
  {

    // load left and right input image
    Mat left_img_src, right_img_src;
    std::cout<<"//before imread"<<std::endl;
    left_img_src = imread(left_img_file_name,CV_LOAD_IMAGE_COLOR);
    right_img_src = imread(right_img_file_name,CV_LOAD_IMAGE_COLOR);
    std::cout<<"//after imread"<<std::endl;

    Mat left_img,right_img;
    cvtColor(left_img_src,left_img,CV_BGR2GRAY);
    cvtColor(right_img_src,right_img,CV_BGR2GRAY);
    
    int32_t width  = left_img.cols;
    int32_t height = left_img.rows;

    // convert input images to uint8_t buffer
    uint8_t* left_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
    uint8_t* right_img_data = (uint8_t*)malloc(width*height*sizeof(uint8_t));
    int32_t k=0;
    for (int32_t row=0; row < left_img.rows; row++) {
      for (int32_t col=0; col < left_img.cols; col++) {
        left_img_data[k]  = left_img.at<uchar>(row,col);
        right_img_data[k] = right_img.at<uchar>(row,col);
        k++;
      }
    }

    // status
    cout << "(my libviso)Processing: Frame: " << i << endl;
    
    // compute visual odometry
    int32_t dims[] = {width,height,width};
    
      viso.process(left_img_data,right_img_data,dims);
      // on success, update current pose
      //cout<<"(libviso) viso.process done"<<endl;
      //Tr_local.push_back(Matrix::inv(viso.getMotion()));
      
      
      if(i%10==0) //try without this also
      {
        while(mythread_comm.in_dloop_g2oedge){std::cout<<"waiting on dloop"<<std::endl;}
        pthread_mutex_lock(&myMutex);
        cout<<"(libviso) going into struct g2o, relin yes"<<endl;
        std::string pref="////relin-ing (struct g2o) regular pose ////";
        mythread_comm.in_viso_g2oedge=true;
        ss = my_for_g2o_edge(numNodes+i,numNodes+i+1,Matrix::inv(viso.getMotion()),isam2,true,nfg,pref);
        mythread_comm.in_viso_g2oedge=false;
        pthread_mutex_unlock(&myMutex);
        //cout<<"(libviso)updated isam through struct g2o edge- relin"<<endl;
      }
      else{
        cout<<"(libviso) going into struct g2o, relin no"<<endl;
        std::string pref="////non relin (struct g2o) regular pose /// ";
        while(mythread_comm.in_dloop_g2oedge){std::cout<<"waiting on dloop"<<std::endl;}
        pthread_mutex_lock(&myMutex);
        mythread_comm.in_viso_g2oedge=true;
        ss = my_for_g2o_edge(numNodes+i,numNodes+i+1,Matrix::inv(viso.getMotion()),isam2,false,nfg,pref);
        mythread_comm.in_viso_g2oedge=false;
        pthread_mutex_unlock(&myMutex);
        //cout<<"(libviso)updated isam through struct g2o edge-non relin"<<endl;
      }
      mywriter.my_write_file << ss;
      mywriter.g2o_string = mywriter.g2o_string + ss;

      if((i%20==0) && (i+5< numImg))
      { 
          cout<<"(libviso)starting for extra relations"<<endl;

          std::string dir2;
        is_good = my_libviso2_relative(extra_relations,i,i+5,dir,dir2);
        cout<<"completed call of libviso relative extra relations function"<<std::endl;
          if(is_good)
          {
            std::string pref="//// relin-ing (struct g2o) frame+5 case ///";
            while(mythread_comm.in_dloop_g2oedge){std::cout<<"waiting on dloop"<<std::endl;}
            pthread_mutex_lock(&myMutex);
            mythread_comm.in_viso_g2oedge=true;
            ss = my_for_g2o_edge(numNodes+i,numNodes+i+5,extra_relations,isam2,true,nfg,pref);
            mythread_comm.in_viso_g2oedge=false;
            pthread_mutex_unlock(&myMutex);

            mywriter.my_write_file << ss;
            //g2o_string << ss;
            mywriter.g2o_string = mywriter.g2o_string + ss;
          }

          cout << "extra done!! " << i << " " << i+5 << endl;
      }

      if(i>1)
      {
        pose = pose * Matrix::inv(viso.getMotion());
        Tr_global.push_back(pose);  
      }
      

      // // output some statistics
      // double num_matches = viso.getNumberOfMatches();
      // double num_inliers = viso.getNumberOfInliers();
      // cout << ", Matches: " << num_matches;
      // cout << ", Inliers: " << 100.0*num_inliers/num_matches << " %" << ", Current pose: " << endl;
      // cout << pose << endl << endl;

    // release uint8_t buffers
    free(left_img_data);
    free(right_img_data);

  // catch image read errors here
  } catch (Exception e) {
        std::cout << e.err << '\n'; 
        std::cout << e.line << '\n'; 
        std::cout << e.what() << '\n';
    cerr << "(libviso)ERROR: Couldn't read input files!" << endl;
    break;
  } 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  while(mythread_comm.loop_wait==true)
  {

  }

  //needed?
  if(((i%500==0) || i==numImg) && mythread_comm.g2o_loop_flag==true) 
  {
       
    //cout << "in function " << gthread.s1 << endl;
    gthread.s1 =  mywriter.g2o_string + "\nFIX 1\n";;
    pthread_create(&g2othread, &attr, g2o_thread, (void *)&gthread);
    pthread_join(g2othread,NULL);
    
    pthread_mutex_lock(&myMutex);
    mythread_comm.g2o_loop_flag = false;
    pthread_mutex_unlock(&myMutex);
    //g2o_string.str("");
     //gthread.s1 = "";
  }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  pthread_mutex_lock(&myMutex);
  mythread_comm.viso_wait_flag = false;
  pthread_mutex_unlock(&myMutex);
  // cout << "going after while" << endl;
 }

pthread_mutex_destroy(&myMutex);

 while(!mythread_comm.loop_write_done)
  {}


ofstream g2o_file;
g2o_file.open("data/online.g2o");

g2o_file << mywriter.g2o_string << "\nFIX 1\n";
g2o_file.close();

mywriter.my_write_file << "\nFIX 1 \n";
mywriter.my_write_file.close();


}


bool my_libviso2_relative(Matrix &Tr_final, int index1, int index2, std::string dir, std::string dir2)
{
   // set most important visual odometry parameters
  // for a full parameter list, look at: viso_stereo.h
  VisualOdometryStereo::parameters param;
  std::string dir_second;
  if(dir2.empty()){
      dir_second=dir;
      std::cout<<"dir2 is null!"<<std::endl;
  }
  else{
      dir_second=dir2;
  }
  /**
    param.calib.f  = 722.658813477; // focal length in pixels
    param.calib.cu = 619.283325195; // principal point (u-coordinate) in pixels
    param.calib.cv = 360.0; // principal point (v-coordinate) in pixels
    param.base     = 0.12; // baseline in meters
**/
  /**
      param.calib.f  =  699.11900; // focal length in pixels
          param.calib.cu = 669.65700; // principal point (u-coordinate) in pixels
                param.calib.cv = 360.60200; // principal point (v-coordinate) in pixels
                     param.base     = 0.12; // baseline in meters
  **/

          param.calib.f = 567.65218; // focal length in pixels
                  param.calib.cu = 679.34969; // principal point (u-coordinate) in pixels
                            param.calib.cv = 370.93921; // principal point (v-coordinate) in pixels
                                     param.base     = 0.12; // baseline in meters

  /**
   * // calibration parameters for sequence 2010_03_09_drive_0019 
  param.calib.f  = 711.9212646484375; // focal length in pixels
  param.calib.cu = 647.0408325195312; // principal point (u-coordinate) in pixels
  param.calib.cv = 360.7899169921875; // principal point (v-coordinate) in pixels
  param.base     = 0.12; // baseline in meters
**/
  
  // init visual odometry
  VisualOdometryStereo viso(param);
  VisualOdometryStereo viso2(param);
  
  // current pose (this matrix transforms a point from the current

    Matrix pose = Matrix::eye(4);
   // Tr_relative r;
    string left_img_file_name;
    string right_img_file_name;
    uint8_t *left_img_data1;
    uint8_t *right_img_data1;

    for (int j = 1; j < 3; j++)
    {
      if (j==1)
      {
          //if(dir2.empty()){
            //  char base_name[256]; sprintf(base_name,"%04d.jpg",index1);
        //  }
         // else{
            char base_name[256]; sprintf(base_name,"%04d.png",index1);
          //}
        
        left_img_file_name  = dir + "left/" + base_name;
        right_img_file_name = dir + "right/" + base_name;
       // r.frame1 = index1;
      }
      else
      {
          char base_name[256];
          if(true || dir==dir_second){
             sprintf(base_name,"%04d.png",index2);
         }
          else{
             sprintf(base_name,"%04d.jpg",index2);
          }
        
        left_img_file_name  = dir_second + "left/" + base_name;
        right_img_file_name = dir_second + "right/" + base_name; 
       // r.frame2 = index2;
      }
          cout << "///////////will start Processing: Frames: " << index1 <<"  "<< index2<< endl;
          cout<<"///////"<<left_img_file_name<<endl;
      try {

      // load left and right input image
      Mat left_img_src, right_img_src;
      left_img_src = imread(left_img_file_name,CV_LOAD_IMAGE_COLOR);
      right_img_src = imread(right_img_file_name,CV_LOAD_IMAGE_COLOR);

      Mat left_img,right_img;
      cvtColor(left_img_src,left_img,CV_BGR2GRAY);
      cvtColor(right_img_src,right_img,CV_BGR2GRAY);
      // png::image< png::gray_pixel > left_img(left_img_file_name);
      // png::image< png::gray_pixel > right_img(right_img_file_name);

      // image dimensions
      // int32_t width  = left_img.get_width();
      // int32_t height = left_img.get_height();

      int32_t width  = left_img.cols;
      int32_t height = left_img.rows;

      // convert input images to uint8_t buffer
      uint8_t* left_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
      uint8_t* right_img_data = (uint8_t*)malloc(width*height*sizeof(uint8_t));
          
      int32_t k=0;
        for (int32_t row=0; row < left_img.rows; row++) {
          for (int32_t col=0; col < left_img.cols; col++) {
          // left_img_data[k]  = left_img.get_pixel(u,v);
          // right_img_data[k] = right_img.get_pixel(u,v);
            left_img_data[k]  = left_img.at<uchar>(row,col);
            right_img_data[k] = right_img.at<uchar>(row,col);
            k++;
          }
        }
      if(j==1){
        left_img_data1  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
        right_img_data1 = (uint8_t*)malloc(width*height*sizeof(uint8_t));
        k=0;
        for (int32_t row=0; row < left_img.rows; row++) {
          for (int32_t col=0; col < left_img.cols; col++) {
          // left_img_data[k]  = left_img.get_pixel(u,v);
          // right_img_data[k] = right_img.get_pixel(u,v);
            left_img_data1[k]  = left_img.at<uchar>(row,col);
            right_img_data1[k] = right_img.at<uchar>(row,col);
            k++;
          }
        }
      }

      // status
        cout << "(libviso relative)Processing: Frame between: " << index1 << '\t' << index2 << endl;
      
        // compute visual odometry
        int32_t dims[] = {width,height,width};
      
      
        if(viso.process(left_img_data,right_img_data,dims)){
            std::cout<<" VISO PROCESS PASSED, J = "<<j<<std::endl;
        }
        else{
            std::cout<<" VISO PROCESS FAILED, J = "<<j<<std::endl;
            if(j==2){
                  free(left_img_data);
                  free(right_img_data);
                  free(left_img_data1);
                  free(right_img_data1);
              return false;
            }
        }
        if(j==2){
         viso2.process(left_img_data,right_img_data,dims);
        }
        // on success, update current pose

        // output some statistics
        //std::cout<<"J = "<<j<<std::endl;
        if(j==2){
            double num_matches = viso.getNumberOfMatches();
            double num_inliers = viso.getNumberOfInliers();
            double inliers_percent = 100.0*num_inliers/num_matches;


            if(viso2.process(left_img_data1,right_img_data1,dims)){
                    std::cout<<" SECOND VISO2 PROCESS PASSED, J = "<<j<<std::endl;
            }        
            else{
                    std::cout<<" SECOND VISO2 PROCESS FAILED, J = "<<j<<std::endl;
                    free(left_img_data);
                  free(right_img_data);
                  free(left_img_data1);
                  free(right_img_data1);
              return false;
            }


            double num_matches2 = viso2.getNumberOfMatches();
            double num_inliers2 = viso2.getNumberOfInliers();
            double inliers_percent2 = 100.0*num_inliers2/num_matches2;

            cout << ", Matches: " << num_matches;
            cout << ", Inliers: " << inliers_percent << " %"<< endl;

            cout << ", Matches2: " << num_matches2;
            cout << ", Inliers2: " << inliers_percent2 << " %"<< endl;   
            double threshold=35;
            if(dir2.empty()){
              threshold=10;
            }
            if(min(inliers_percent,inliers_percent2)>=9)
            {
              pose = pose * Matrix::inv(viso.getMotion());
              std::cout<<"pose "<<pose<<std::endl;
              Tr_final = pose;  

              free(left_img_data);
              free(right_img_data);
              free(left_img_data1);
              free(right_img_data1);
              return true;
              //cout << "i: " << i << '\t' << "j: " << j <<endl << endl;
            }
        }



      // release uint8_t buffers
      free(left_img_data);
      free(right_img_data);

      // catch image read errors here
    } 
      catch (Exception e) 
      {
        std::cout << e.err << '\n'; 
        std::cout << e.line << '\n'; 
        std::cout << e.what() << '\n';
        cerr << "(libviso relative)ERROR: Couldn't read input files!" << endl;
        break;
      }
   }

   return false;
}


// Old function using Vectors
void my_libviso2_relative(std::vector<Tr_relative> &Tr_final, std::vector<int> index1, std::vector<int> index2, std::string dir)
{
   // set most important visual odometry parameters
  // for a full parameter list, look at: viso_stereo.h
  VisualOdometryStereo::parameters param;
  
  
  // calibration parameters for sequence 2010_03_09_drive_0019 
  param.calib.f  = 711.9212646484375; // focal length in pixels
  param.calib.cu = 647.0408325195312; // principal point (u-coordinate) in pixels
  param.calib.cv = 360.7899169921875; // principal point (v-coordinate) in pixels
  param.base     = 0.12; // baseline in meters

  // init visual odometry
  VisualOdometryStereo viso(param);
  
  for (uint i=0; i<index1.size(); i++) 
  {

    Matrix pose = Matrix::eye(4);
    Tr_relative r;
    string left_img_file_name;
    string right_img_file_name;

    for (int j = 1; j < 3; j++)
    {
      if (j==1)
      {
        char base_name[256]; sprintf(base_name,"%04d.jpg",index1[i]);
        left_img_file_name  = dir + "left/" + base_name;
        right_img_file_name = dir + "right/" + base_name;
        r.frame1 = index1[i];
      }
      else
      {
        char base_name[256]; sprintf(base_name,"%04d.jpg",index2[i]);
        left_img_file_name  = dir + "left/" + base_name;
        right_img_file_name = dir + "right/" + base_name; 
        r.frame2 = index2[i];
      }
        
      // cout << left_img_file_name << endl;
      // cout << right_img_file_name << endl;
      // catch image read/write errors here
      try {

      // load left and right input image
      Mat left_img_src, right_img_src;
      left_img_src = imread(left_img_file_name,CV_LOAD_IMAGE_COLOR);
      right_img_src = imread(right_img_file_name,CV_LOAD_IMAGE_COLOR);

      Mat left_img,right_img;
      cvtColor(left_img_src,left_img,CV_BGR2GRAY);
      cvtColor(right_img_src,right_img,CV_BGR2GRAY);
      // png::image< png::gray_pixel > left_img(left_img_file_name);
      // png::image< png::gray_pixel > right_img(right_img_file_name);

      // image dimensions
      // int32_t width  = left_img.get_width();
      // int32_t height = left_img.get_height();

      int32_t width  = left_img.cols;
      int32_t height = left_img.rows;

      // convert input images to uint8_t buffer
      uint8_t* left_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
      uint8_t* right_img_data = (uint8_t*)malloc(width*height*sizeof(uint8_t));
      int32_t k=0;
        for (int32_t row=0; row < left_img.rows; row++) {
          for (int32_t col=0; col < left_img.cols; col++) {
          // left_img_data[k]  = left_img.get_pixel(u,v);
          // right_img_data[k] = right_img.get_pixel(u,v);
            left_img_data[k]  = left_img.at<uchar>(row,col);
            right_img_data[k] = right_img.at<uchar>(row,col);
            k++;
          }
        }

      // status
        cout << "Processing: Frame between: " << r.frame1 << '\t' << r.frame2 << endl;
      
        // compute visual odometry
       int32_t dims[] = {width,height,width};
     
        viso.process(left_img_data,right_img_data,dims);
        // on success, update current pose

        // output some statistics

        double num_matches = viso.getNumberOfMatches();
        double num_inliers = viso.getNumberOfInliers();
        double inliers_percent = 100.0*num_inliers/num_matches;
        cout << ", Matches: " << num_matches;
        cout << ", Inliers: " << inliers_percent << " %" << ", Current pose: " << endl;
        cout << pose << endl << endl;

         if(j>1 && inliers_percent>30)
        {
          pose = pose * Matrix::inv(viso.getMotion());
          r.transform = pose;
          Tr_final.push_back(r);  

          cout << "i: " << i << '\t' << "j: " << j <<endl << endl;
        }

      // release uint8_t buffers
      free(left_img_data);
      free(right_img_data);

      // catch image read errors here
    } 
      catch (Exception e) 
      {
        std::cout << e.err << '\n'; 
        std::cout << e.line << '\n'; 
        std::cout << e.what() << '\n';  
        cerr << "ERROR: Couldn't read input files!" << endl;
        break;
      }
   }
  }    
}



string my_for_g2o_edge(int id1, int id2, Matrix transform, gtsam::ISAM2 &isam2,bool relin,gtsam::NonlinearFactorGraph &nfg,std::string prefix)
{

  std::cout<<prefix<<" struct g2o function entered"<<std::endl;
  stringstream out;

  Matrix new_tr;

  new_tr = transform;
  std::cout<<"transform is "<<std::endl;
  std::cout<<transform<<std::endl;
  double dx = new_tr.getData(0,3);
  double dy = new_tr.getData(1,3);
  double dz = new_tr.getData(2,3);

  double thetay = atan2(new_tr.getData(0,2), sqrt(new_tr.getData(0,0)*new_tr.getData(0,0) + new_tr.getData(0,1)*new_tr.getData(0,1)));

  double thetax = atan2(-new_tr.getData(1,2), new_tr.getData(2,2));

  double thetaz = atan2(-new_tr.getData(0,1), new_tr.getData(0,0));

  double thetay0 = atan2(-new_tr.getData(2,0), sqrt(new_tr.getData(2,1)*new_tr.getData(2,1) + new_tr.getData(2,2)*new_tr.getData(2,2)));

  double thetax0 = atan2(new_tr.getData(2,1), new_tr.getData(2,2));

  double thetaz0 = atan2(new_tr.getData(1,0), new_tr.getData(0,0));


  if(id2>mythread_comm.Nfirstloop+1){
  out << "EDGE3 " << id1 <<' '<< id2 <<' '<< dx << ' '<< dy <<' '<< dz <<' '<< thetax << ' ' << thetay <<' '<< thetaz<<" "<< "1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;
  }
  std::cout<<"dx dy dz thetax thetay thetaz : "<<dx<<"  "<<dy<<" "<<dz<<"  "<<thetax<<"  "<<thetay<<" "<<thetaz<<std::endl;

  std::cout<<"old thetax thetay thetaz (ZYX) : "<<thetax0<<" "<<thetay0<<" "<<thetaz0<<std::endl;
  std::cout<<"new thetax thetay thetaz (XYZ): "<<thetax<<" "<<thetay<<" "<<thetaz<<std::endl;

  gtsam::Matrix M(6, 6);
  M<<0.00001,0,0,0,0,0,0,0.00001,0,0,0,0,0,0,0.00001,0,0,0,0,0,0,0.00001,0,0,0,0,0,0,0.00001,0,0,0,0,0,0,0.00001;
  gtsam::SharedNoiseModel model;
  model = gtsam::noiseModel::Gaussian::Information(M, true);
  gtsam::Rot3 rot1(new_tr.getData(0,0),new_tr.getData(0,1),new_tr.getData(0,2),new_tr.getData(1,0),new_tr.getData(1,1),new_tr.getData(1,2),new_tr.getData(2,0),new_tr.getData(2,1),new_tr.getData(2,2));
  gtsam::Rot3 rot2 = gtsam::Rot3::RzRyRx(thetax,thetay,thetaz);
  gtsam::Rot3 rot3 = gtsam::Rot3::RzRyRx(thetax0,thetay0,thetaz0);
  std::cout<<"RZRYRX WITH NEW THETA"<<std::endl;
  rot2.print();
  std::cout<<"RZRYRX WITH OLD THETA"<<std::endl;
  rot3.print();
  std::cout<<"rotation matrix from rotation matrix (being used)"<<std::endl;
  rot1.print();
  
  gtsam::Point3 point(dx,dy,dz);
  gtsam::Pose3 l1Xl2(rot1,point);

      gtsam::NonlinearFactor::shared_ptr factor(
          new gtsam::BetweenFactor<gtsam::Pose3>(id1, id2, l1Xl2, model));
      factor->print(prefix);
      nfg.push_back(factor);


	if(mythread_comm.first_loop_just_found)
	{
            //call insert into isam 
            if(id1>mythread_comm.Nfirstloop+1 && id2<=mythread_comm.Nfirstloop+1){
                mythread_comm.inserting_into_isam=true;
                
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"<<<<<<<<<<   ADDING PRIOR     <<<<<<<<<<<<<"<<std::endl;
                l1Xl2.print();
                std::cout<<std::endl;
                std::cout<<"<<<<<<<<<<<   ADDED PRIOR     <<<<<<<<<<<<"<<std::endl;
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;

        //adding prior
        gtsam::NonlinearFactorGraph newFactors;
        gtsam::Values newVariables;
        //gtsam::Vector v(3);
        //v<<0.01, 0.01, 0.01;
        //gtsam::SharedDiagonal priorNoise = gtsam::noiseModel::Diagonal::Sigmas(v);
        std::cout<<"calculated index into tr_global is "<<id1-mythread_comm.Nfirstloop-4<<std::endl;
        //Matrix estimatesecondLoop = Tr_global[id1-mythread_comm.Nfirstloop-4]; //?

        //double dx2 = estimatesecondLoop.getData(0,3);
        //double dz2 = estimatesecondLoop.getData(2,3);
        //double thetay2 = atan2(-estimatesecondLoop.getData(2,0), sqrt(estimatesecondLoop.getData(2,1)*estimatesecondLoop.getData(2,1) + estimatesecondLoop.getData(2,2)*estimatesecondLoop.getData(2,2))); //remove minus here?
        
        //std::cout<<"tr_global dz dy thety : "<<dz2<<"  "<<dx2<<"  "<<thetay2<<std::endl;
        
        //gtsam::Pose3 secondLoopLocation(dz2,dx2,thetay2);//minus?
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                //std::cout<<"secondLoopLocation BEING ADDED IS: "<<std::endl;
                //secondLoopLocation.print();
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
        gtsam::Pose3 firstLoopLocation = initial.at<gtsam::Pose3>(id2);
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"firstLoopLocation BEING ADDED IS: "<<std::endl;
                firstLoopLocation.print();
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
        //gtsam::Pose3 betweenPose((firstLoopLocation.x()-l1Xl2.x()),(firstLoopLocation.y()-l1Xl2.y()),(firstLoopLocation.theta()-l1Xl2.theta()));
	gtsam::Pose3 betweenPose( l1Xl2.inverse()*firstLoopLocation );
        refPose=betweenPose;
	currentPose=refPose;
	std::cout<<"latest key was "<<latestkey<<std::endl;
        latestkey=id1;
	std::cout<<"latest key is now "<<latestkey<<std::endl;
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"betweenPose BEING ADDED IS: "<<std::endl;
		betweenPose.print();
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
                std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
        newFactors.push_back(boost::make_shared<gtsam::PriorFactor<gtsam::Pose3> >(id1, betweenPose, gtsam::noiseModel::Unit::Create(gtsam::Pose3::dimension)));
        newVariables.insert(id1,betweenPose);
        initialunopt.insert(id1,refPose);
        
        gtsam::ISAM2Result res1 =  isam2.update(newFactors, newVariables,gtsam::FactorIndices(),boost::none,boost::none,boost::none, true);
        std::cout<< "variables relinearised = "<<res1.variablesRelinearized<<std::endl;
        
                insertIntoIsam(isam2,nfg,id1);
                mythread_comm.first_loop_just_found=false;  
                mythread_comm.inserting_into_isam=true;
                mythread_comm.normal_postloop_flow=true;
            }
            else{
                throw runtime_error("first loop closure conditions not met");
            }
            return out.str();
	}

      if(mythread_comm.normal_postloop_flow){
          
        gtsam::Values newVariables;
        gtsam::NonlinearFactorGraph newFactors;
        if(gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr observation = 
            boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3> >(factor))
        {
                std::cout<<" casted"<<std::endl;
                newFactors.push_back(observation);
                if(observation->key1() > observation->key2()){
                  cout<<" observation->key1() > observation->key2()"<<endl;
                  if(!isam2.valueExists(observation->key1()) ){                     
                    if(!isam2.valueExists(observation->key2())){
                        throw runtime_error("Problem! None exist");
                    }
                    if(id1 == 1){
                      cout<<" inserting at id1=1"<<endl;
                      newVariables.insert(observation->key1(), observation->measured().inverse());
                      cout<<" inserted at id1=1"<<endl;
                    }
                    else{
                      cout<<" inserting key1"<<endl;
                      gtsam::Pose3 previousPose = isam2.calculateEstimate<gtsam::Pose3>(observation->key2());
                      newVariables.insert(observation->key1(), previousPose * observation->measured().inverse());
                    }
                  }
                }
                else {  //+5 frame constraint case
                  cout<<" observation->key1() < observation->key2()"<<endl;
                  if(!isam2.valueExists(observation->key2()) ){
                    if(!isam2.valueExists(observation->key1())){
                        throw runtime_error("Problem! None exist");
                    }
                    if (id1 == 1){
                      cout<<" inserted at id1=1"<<endl;
                      newVariables.insert(observation->key2(), observation->measured());
                      cout<<" inserted at id1=1"<<endl;
                    }
                    else{
                      cout<<" inserting key2"<<endl;
                      gtsam::Pose3 previousPose = isam2.calculateEstimate<gtsam::Pose3>(observation->key1());
                      newVariables.insert(observation->key2(), previousPose * observation->measured());
                    }
                  }
                  if(!initialunopt.exists(observation->key2())){
                      	if(observation->key1() +1== observation->key2()){
				std::cout<<"BOTH PRINTS FOR COMPARE"<<std::endl;

				std::cout<<"updating for key2 "<<observation->key2()<<std::endl;
				std::cout<<"observation key1 "<<observation->key1()<<std::endl;
				std::cout<<"observation->measured()*currentPose"<<std::endl;
				(observation->measured()*currentPose).print();
				std::cout<<"currentPose*observation->measured()"<<std::endl;
				(currentPose*observation->measured()).print();
				std::cout<<"latest key was "<<latestkey<<std::endl;
				if(latestkey!=observation->key1()){throw runtime_error("Problem exists with latest key");}
				currentPose=currentPose*observation->measured();
				initialunopt.insert(observation->key2(),currentPose);
				latestkey=observation->key2();
				std::cout<<"latest key is now "<<latestkey<<std::endl;
			}
                  }
                }
                      
        }
      //gtsam::Values estimate(isam2.calculateEstimate());
      //estimate.print("");
        cout<<"going to call isamupdate in struct g2o function"<<endl;
	if(relin){
            std::cout<<"ASKED TO RELIN"<<std::endl;
        gtsam::ISAM2Result res1 =  isam2.update(newFactors, newVariables,gtsam::FactorIndices(),boost::none,boost::none,boost::none, true);
        std::cout<< "variables relinearised (non zero)= "<<res1.variablesRelinearized<<std::endl;
//            if(res1.variablesRelinearized>200)
//            {
//                 //write a file to visualise later
//            }
	}
	else{
            cout<<"next line is isamupdate"<<endl;
            cout<<newVariables.size()<<endl;
            cout<<newFactors.size()<<endl;
	gtsam::ISAM2Result res1 = isam2.update(newFactors,newVariables);
        std::cout<< "variables relinearised (should be zero)= "<<res1.variablesRelinearized<<std::endl;
	}

       
    
        std::cout<<"end of struct g2o edge function"<<std::endl;
      
      }
//      else{
//        throw runtime_error("in the unexpected else case of struct g2o edge");
//      }
   return out.str();

}
