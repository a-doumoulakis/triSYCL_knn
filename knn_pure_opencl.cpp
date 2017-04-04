#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <array>
#include <sstream>

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
int result[training_set_size];

std::vector<int> get_vector(const std::vector<Img>& imgs) {
  std::vector<int> res;
  for(Img elem : imgs) {
    res.insert(res.end(), std::begin(elem.pixels), std::end(elem.pixels));
  }
  return res;
}


std::vector<Img> slurp_file(const std::string& name) {
  std::ifstream infile { name, std::ifstream::in };
  //std::cout << "Reading " << name << std::endl;
  std::string line, token;
  std::vector<Img> res;
  bool fst_1 = true;

  while (std::getline(infile, line)) {
    if(fst_1) {
      fst_1 = false;
      continue;
    }
    Img img;
    std::istringstream iss {line };
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
  //std::cout << "Done" << std::endl;
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

  q.enqueueReadBuffer(res, CL_TRUE, 0, sizeof(int) * 5000, result);
  
  // Find the image with the minimum distance
  auto min_image = std::min_element(std::begin(result), std::end(result));
  
  // Test if we found the good digit
  return
    training_set[std::distance(std::begin(result), min_image)].label == label;
}


int main(int argc, char* argv[]) {
  
  training_set = slurp_file("data/trainingsample.csv");
  validation_set =  slurp_file("data/validationsample.csv");
  
  std::vector<cl::Platform> platform_list;
  cl::Platform::get(&platform_list);
  if(platform_list.size() == 0) {
    std::cout << "No platform found" << std::endl;
    return 1;
  }
  cl::Platform default_platform = platform_list[2];

  //std::cout << "Using " << default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
  //std::cout << "Size " << platform_list.size() << std::endl;
  
  std::vector<cl::Device> device_list;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &device_list);
  if(device_list.size() == 0) {
    std::cout << "No device found" << std::endl;
    return 1;
  }
  cl::Device default_device = device_list[0];

  std::cout << "\nUsing " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;
  //std::cout << "Size " << device_list.size() << std::endl;
  
  std::cout << std::endl;

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

  auto start_time = std::chrono::high_resolution_clock::now();
  
  for(Img img : validation_set) {
  
    q.enqueueWriteBuffer(data, CL_TRUE, 0,
			 sizeof(int) * img.pixels.size(),
			 img.pixels.data());
    
    correct += compute(training, data, res, q, kernel, img.label);
  }

  std::chrono::duration<double, std::milli> duration_ms =
    std::chrono::high_resolution_clock::now() - start_time;

  std::cout << (duration_ms.count()/validation_set.size())
            << " ms/kernel" << std::endl;


  std::cout << "\nResult : " << ((correct / 500.0) * 100.0) << "%"
            << " (" << correct << ")"
	    << std::endl;
  
  return 0;
}

