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
  return res;
}

int search_image(buffer<int>& training, buffer<int>& res_buffer,
                 const Img& img, queue& q) {

  {
    buffer<int> A { std::begin(img.pixels), std::end(img.pixels) };
    // Compute the L2 distance between an image and each one from the
    // training set
    q.submit([&] (handler &cgh) {
        // These accessors lazily trigger data transfers between host
        // and device only if necessary. For example "training" is
        // only transfered the first time the kernel is executed.
        auto train = training.get_access<access::mode::read>(cgh);
        auto ka = A.get_access<access::mode::read>(cgh);
        auto kb = res_buffer.get_access<access::mode::write>(cgh);
        // Launch a kernel with training_set_size work-items
        cgh.parallel_for<class KnnKernel>(range<1> { training_set_size },
                                          [=] (id<1> index) {
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

  auto r = res_buffer.get_access<access::mode::read>();

  // Find the image with the minimum distance
  int index = 0;
  for(int i = 0; i < 5000; i++) if(result[i] < result[index]) index=i;

  // Test if we found the good digit
  return training_set[index].label == img.label;
}

int main(int argc, char* argv[]) {
  training_set = slurp_file("data/trainingsample.csv");
  validation_set =  slurp_file("data/validationsample.csv");
  buffer<int> training_buffer = get_buffer(training_set);
  buffer<int> result_buffer { result, training_set_size };

  // A SYCL queue to send the heterogeneous work-load to
  queue q;

  double sum = 0.0;
  int correct = 0;

  for(int h = 1; h <= 1000; h++){

    auto start_time = std::chrono::high_resolution_clock::now();

    // Match each image from the validation set against the images from
    // the training set
    for (auto const & img : validation_set)
      correct += search_image(training_buffer, result_buffer, img, q);

    std::chrono::duration<double, std::milli> duration_ms =
      std::chrono::high_resolution_clock::now() - start_time;

    double exec_for_image = (duration_ms.count()/validation_set.size());

    sum += exec_for_image;

    std::cout << h/10.0 << "% \t| " << "Duration : " << exec_for_image
              << " ms/kernel\n";

    std::cout << "\t| Average : " << (sum/h) << "\n"
              << "\t| Result " << (100.0*correct/validation_set.size()) << "%"
              << std::endl;

    std::cout << std::endl;
    correct = 0;
  }
  std::cout << "FINAL AVERAGE : " << (sum/1000) << std::endl;
  return 0;
}
