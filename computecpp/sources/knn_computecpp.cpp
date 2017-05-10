/* Digit recognition in images using nearest neighbour matching */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <sstream>

#include <CL/sycl.hpp>

using namespace cl::sycl;

constexpr size_t training_set_size = 5000;
constexpr size_t pixel_number = 784;

using Vector = std::array<int, pixel_number>;

class KnnKernel;

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
  std::cout << "Reading " << name << std::endl;
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
  std::cout << "Done" << std::endl;
  return res;
}

void search_image(buffer<int>& training, const Img& img, queue& q) {
  {
    buffer<cl::sycl::cl_int, 1> res_buffer(result, 5000);
    buffer<int> A { std::begin(img.pixels), std::end(img.pixels) };
    // Compute the L2 distance between an image and each one from the
    // training set

    q.submit([&] (handler &cgh) {
        // These accessors lazily trigger data transfers between host
        // and device only if necessary. For example "training" is
        // only transfered the first time the kernel is executed.
        auto train = training.get_access<access::mode::read>(cgh);
        auto ka = A.get_access<access::mode::read>(cgh);
        auto kb = res_buffer.get_access<access::mode::discard_write>(cgh);
        // Launch a kernel with training_set_size work-items
        cgh.parallel_for<class KnnKernel>(range<1> { training_set_size }, [=] (id<1> index) {
            decltype(ka)::value_type diff = 0;
            // For each pixel
            for (auto i = 0; i != pixel_number; i++) {
              auto toAdd = ka[i] - train[index[0]*pixel_number + i];
              diff += toAdd*toAdd;
            }
            kb[index] = diff;
          });
      });
  }
}

int main(int argc, char* argv[]) {
  training_set = slurp_file("/home/anastasi/Documents/triSYCL_knn/data/trainingsample.csv");
  validation_set =  slurp_file("/home/anastasi/Documents/triSYCL_knn/data/validationsample.csv");
  buffer<int> training_buffer = get_buffer(training_set);

  // A SYCL queue to send the heterogeneous work-load to
  queue q;

  double sum = 0.0;
  
  for(int h = 1; h <= 1000; h++){
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int correct = 0;

    // Match each image from the validation set against the images from
    // the training set
    for (auto const & img : validation_set){
      search_image(training_buffer, img, q);
      int index = 0;
      for(int i = 0; i < 5000; i++) if(result[i] < result[index]) index=i;
      correct += training_set[index].label == img.label;
    }

    std::chrono::duration<double, std::milli> duration_ms =
      std::chrono::high_resolution_clock::now() - start_time;

    double exec_for_image = (duration_ms.count()/validation_set.size());

    sum += exec_for_image; 

    std::cout << h/10.0 << "% | " << "Duration : " << exec_for_image << " ms/kernel\n";
    
    std::cout << "     | Average : " << (sum/h) << "\n"
              << "     | Result " << (100.0*correct/validation_set.size()) << "%" << std::endl;
    
    std::cout << std::endl;
    correct = 0;
  }
  std::cout << "FINAL AVERAGE : " << (sum/1000) << std::endl;
  return 0;
}
