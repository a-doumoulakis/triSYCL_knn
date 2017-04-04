#include <iostream>
#include <iterator>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <array>
#include <cmath>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <CL/cl.hpp>

constexpr size_t training_set_size = 5000;
constexpr size_t data_size = 784;

using Vector = std::array<int, data_size>;

struct Img {
  int label;
  Vector pixels;
};

std::vector<Img> training_set;
std::vector<Img> validation_set;

std::vector<int> get_vector(const std::vector<Img>& imgs) {
  std::vector<int> res;
  for(Img elem : imgs) {
    res.insert(res.end(), std::begin(elem.pixels), std::end(elem.pixels));
  }
  return res;
}


std::vector<Img> slurp_file(const std::string& name) {
  std::ifstream infile(name, std::ifstream::in);
  std::cout << "Reading " << name << std::endl;
  std::string line, token;
  std::vector<Img> res;
  bool fst_1 = true;

  while (std::getline(infile, line)) {
    if(fst_1) {
      fst_1 = false;
      continue;
    }
    Img img;
    std::istringstream iss(line);
    bool fst = true;
    int index = 0;
    while(std::getline(iss, token, ',')) {
      if(fst) {
        img.label = std::stoi(token);
        fst = false;
      }
      else {
        img.pixels[index] = std::stoi(token);
        index++;
      }
    }
    res.push_back(img);
  }
  std::cout << "Done" << std::endl;
  return res;
}

int compute(cl::Buffer& training, cl::Buffer& data, cl::Buffer& res,
	    cl::CommandQueue& q,  cl::Kernel& kern, int label) {

  kern.setArg(0, training);
  kern.setArg(1, data);
  kern.setArg(2, res);
  kern.setArg(3, 5000);
  kern.setArg(4, 784);

  q.enqueueNDRangeKernel(kern, cl::NullRange, cl::NDRange(5000), cl::NullRange);
  q.finish();
  
  int result[5000];

  q.enqueueReadBuffer(res, CL_TRUE, 0, sizeof(int) * 5000, result);
  
  int index = 0;
  double square = std::sqrt(result[0]);
  for(unsigned i = 1; i < training_set_size; i++) {
    double tmp = std::sqrt(result[i]);
    if(tmp < square) {
      index = i;
      square = tmp;
    }
  }
  if(training_set[index].label == label) return 1;
  return 0;
}


int main(int argc, char* argv[]) {
  
  training_set = slurp_file("../data/trainingsample.csv");
  validation_set =  slurp_file("../data/validationsample.csv");
  
  std::vector<cl::Platform> platform_list;
  cl::Platform::get(&platform_list);
  if(platform_list.size() == 0) {
    std::cout << "No platform found" << std::endl;
    return 1;
  }
  cl::Platform default_platform = platform_list[0];

  //std::cout << "Using " << default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
  //std::cout << "Size " << platform_list.size() << std::endl;
  
  std::vector<cl::Device> device_list;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &device_list);
  if(device_list.size() == 0) {
    std::cout << "No device found" << std::endl;
    return 1;
  }
  cl::Device default_device = device_list[0];

  //std::cout << "Using " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;
  //std::cout << "Size " << device_list.size() << std::endl;
  
  cl::Context ctx({ default_device });

  cl::Program::Sources src;
  std::string kernel_src = "                                            \
    __kernel void kernel_compute(__global const int* trainingSet,	\
                                 __global const int* data,		\
                                 __global int* res,			\
                                 int setSize, int dataSize) {		\
    int diff, toAdd, computeId;						\
    computeId = get_global_id(0);					\
    if(computeId < setSize){						\
        diff = 0;							\
        for(int i = 0; i < dataSize; i++){				\
            toAdd = data[i] - trainingSet[computeId*dataSize + i];	\
            diff += toAdd * toAdd;					\
        }								\
    res[computeId] = diff;						\
    }} ";
  src.push_back({kernel_src.c_str(), kernel_src.length()});

  
  cl::Program program(ctx, src);
  if(program.build({default_device}) != CL_SUCCESS) {
    std::cout << "Error building the program" << std::endl;
    return 1;
  }

  cl::Kernel kernel = cl::Kernel(program, "kernel_compute");

  cl::CommandQueue q(ctx, default_device);

  std::vector<int> train_vect = get_vector(training_set);

  
  cl::Buffer training(ctx, CL_MEM_READ_ONLY,
		      (sizeof(int) * (training_set_size * data_size)));
  cl::Buffer data(ctx, CL_MEM_READ_ONLY, (sizeof(int) * data_size));
  cl::Buffer res(ctx, CL_MEM_WRITE_ONLY, (sizeof(int) * training_set_size));
  
  q.enqueueWriteBuffer(training, CL_TRUE, 0,
		       sizeof(int) * train_vect.size(), train_vect.data());
  int correct = 0;

  //boost::posix_time::ptime mst1 = boost::posix_time::microsec_clock::local_time(); 

  for(Img img : validation_set) {
  
    q.enqueueWriteBuffer(data, CL_TRUE, 0,
			 sizeof(int) * img.pixels.size(),
			 img.pixels.data());
    
    correct += compute(training, data, res, q, kernel, img.label);
  }
  
  //boost::posix_time::ptime mst2 = boost::posix_time::microsec_clock::local_time();
  //boost::posix_time::time_duration msdiff = mst2 - mst1;
  //std::cout << (msdiff.total_milliseconds() / 500.0) << std::endl;
  std::cout << "\nResult : " << ((correct / 500.0) * 100.0) << "%"
            << " (" << correct << ")"
	    << std::endl;
  
  return 0;
}

