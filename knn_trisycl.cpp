#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>
#include <limits>

#include <boost/compute.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <CL/sycl.hpp>

using namespace cl::sycl;

constexpr size_t training_set_size = 5000;
constexpr size_t data_size = 784;

using Vector = std::array<int, data_size>;

struct Img {
  int label;
  Vector pixels;
};

std::vector<Img> training_set;
std::vector<Img> validation_set;

buffer<int> get_buffer(const std::vector<Img>& imgs) {
  std::vector<int> res;
  for (Img elem : imgs) {
    res.insert(res.end(), std::begin(elem.pixels), std::end(elem.pixels));
  }
  return buffer<int> { std::begin(res), std::end(res) };
}

std::vector<Img> slurp_file(const std::string& name) {
  std::ifstream infile { name, std::ifstream::in };
  std::cout << "Loading " << name << std::endl;
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

int compute(buffer<int>& training, const Img& img, queue& q) {
  int res[training_set_size];

  {
    buffer<int> A { std::begin(img.pixels), std::end(img.pixels) };
    buffer<int> B { res, training_set_size };
    q.submit([&](handler &cgh) {
        auto train = training.get_access<access::mode::read>(cgh);
        auto ka = A.get_access<access::mode::read>(cgh);
        auto kb = B.get_access<access::mode::write>(cgh);
        cgh.parallel_for(range<1> { 5000 }, [=] (id<1> index){
            int diff, toAdd;
            if (index < 5000) {
              diff = 0;
              for (int i = 0; i < 784; i++) {
                toAdd = ka[i] - train[index[0] * 784 + i];
                diff += toAdd * toAdd;
              }
              kb[index] = diff;
            }
          });
      });
  }

  int index = 0;
  double square = std::sqrt(res[0]);
  for (unsigned i = 1; i < training_set_size; i++) {
    double tmp = std::sqrt(res[i]);
    if (tmp < square) {
      index = i;
      square = tmp;
    }
  }
  if (training_set[index].label == img.label) return 1;
  return 0;
}

int main(int argc, char* argv[]) {
  int correct = 0;
  training_set = slurp_file("data/trainingsample.csv");
  validation_set =  slurp_file("data/validationsample.csv");
  buffer<int> training_buffer = get_buffer(training_set);

  queue q;

  boost::posix_time::ptime mst1 = boost::posix_time::microsec_clock::local_time(); 

  for (Img img : validation_set)
    correct += compute(training_buffer, img, q);

  boost::posix_time::ptime mst2 = boost::posix_time::microsec_clock::local_time();
  boost::posix_time::time_duration msdiff = mst2 - mst1;
  std::cout << (msdiff.total_milliseconds() / 500.0) << std::endl;

  std::cout << "\nResult : " << ((correct / 500.0) * 100.0) << "%"
            << " (" << correct << ")"
            << std::endl;
  return 0;
}
