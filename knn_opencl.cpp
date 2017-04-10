/* Digit recognition in images using nearest neighbour matching */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <boost/compute.hpp>

#include <CL/sycl.hpp>

using namespace cl::sycl;

constexpr size_t training_set_size = 5000;
constexpr size_t pixel_number = 784;
range<1> global_size {5000};

using Vector = std::array<int, pixel_number>;

struct Img {
  // The digit value [0-9] represented on the image
  int label;
  // The 1D-linearized image pixels
  Vector pixels;
};

std::vector<Img> training_set;
std::vector<Img> validation_set;
int result[training_set_size];

// Construct a SYCL buffer from a vector of images
buffer<int> get_buffer(const std::vector<Img>& imgs) {
  std::vector<int> res;
  for (auto const& elem : imgs) {
    res.insert(res.end(), std::begin(elem.pixels), std::end(elem.pixels));
  }
  return { std::begin(res), std::end(res) };
}

// Read a CSV-file containing image pixels
std::vector<Img> slurp_file(const std::string& name) {
  std::ifstream infile { name, std::ifstream::in };
  //std::cout << "Reading " << name << std::endl;
  std::string line, token;
  std::vector<Img> res;
  bool fst_1 = true;

  while (std::getline(infile, line)) {
    if (fst_1) {
      fst_1 = false;
      continue;
    }
    Img img;
    std::istringstream iss { line };
    bool fst = true;
    int index = 0;
    while (std::getline(iss, token, ',')) {
      if (fst) {
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

int search_image(buffer<int>& training, buffer<int>& res,
		 const Img& img, queue& q, const kernel& k) {

  {
    buffer<int> A { std::begin(img.pixels), std::end(img.pixels) };
    // Compute the L2 distance between an image and each one from the
    // training set
    q.submit([&] (handler &cgh) {
        // Set the kernel arguments. The accessors lazily trigger data
        // transfers between host and device only if necessary. For
        // example "training" and "res" are only transfered to the device
	// the first time the kernel is executed and "res" is transfered back
	// after every execution
        cgh.set_args(training.get_access<access::mode::read>(cgh),
                     A.get_access<access::mode::read>(cgh),
                     res.get_access<access::mode::write>(cgh),
                     int { training_set_size }, int { pixel_number });
        // Launch the kernel with training_set_size work-items
        cgh.parallel_for(global_size, k);
      });
  }

  // Wait for kernel to finish so "result" contains the right data
  q.wait();
  
  // Find the image with the minimum distance
  auto min_image = std::min_element(std::begin(result), std::end(result));

  // Test if we found the good digit
  return
    training_set[std::distance(std::begin(result), min_image)].label == img.label;
}

int main(int argc, char* argv[]) {
  //int correct = 0;
  training_set = slurp_file("data/trainingsample.csv");
  validation_set =  slurp_file("data/validationsample.csv");
  buffer<int> training_buffer = get_buffer(training_set);
  buffer<int> result_buffer { result, training_set_size };

  // Device selection
  std::vector<boost::compute::device> devices = boost::compute::system::devices();
  boost::compute::device device = devices[2];
  
  std::cout << "\nUsing " << device.name() << std::endl;
  //for(auto& device : devices)
  //  std::cout << " -" << device.name() << std::endl;

  std::cout << std::endl;

  // Boost context and queue to allow us to choose
  // whichever device we want
  boost::compute::context context { device };
  boost::compute::command_queue b_queue { context, device };


  // A SYCL queue to send the heterogeneous work-load to
  queue q { b_queue };

  auto program = boost::compute::program::create_with_source(R"(
    __kernel void kernel_compute(__global const int* trainingSet,
                                 __global const int* data,
                                 __global int* res, int setSize, int dataSize) {
      int diff, toAdd, computeId;
      computeId = get_global_id(0);
      if (computeId < setSize) {
        diff = 0;
        for (int i = 0; i < dataSize; i++) {
            toAdd = data[i] - trainingSet[computeId*dataSize + i];
            diff += toAdd * toAdd;
        }
        res[computeId] = diff;
      }
    }
    )", context);

  program.build();

  // Construct a SYCL kernel from OpenCL kernel to be used in
  // interoperability mode
  kernel k { boost::compute::kernel { program, "kernel_compute"} };
  
  double sum = 0.0;
  
  for(int h = 1; h <= 200; h++){
    auto start_time = std::chrono::high_resolution_clock::now();
  
  // Match each image from the validation set against the images from
  // the training set
  //for (auto const & img : validation_set)
    for(int i = 0; i < 500; i++)
      search_image(training_buffer, result_buffer, validation_set[i], q, k);

    std::chrono::duration<double, std::milli> duration_ms =
      std::chrono::high_resolution_clock::now() - start_time;

    sum += (duration_ms.count()/500); 
  //std::cout << (duration_ms.count()/validation_set.size())
  //          << " ms/kernel" << std::endl;
    //if(sum/h > last){
    std::cout << h/2.0 << "%" << " avg : " << (sum/h) << std::endl;
      //}
  }
  std::cout << "AVERAGE : " << (sum/1000) << std::endl;
  //std::cout << "\nResult : " << (100.0*correct/validation_set.size()) << '%'
  //          << " (" << correct << ")"
  //          << std::endl;
  return 0;
}
