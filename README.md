
## K-NN with triSYCL and OpenCL interop

----------

### The Algorithm

To demonstrate the OpenCL interoperability mode for triSYCL, we programmed a simple **k-NN** algorithm ([k-nearest neighbor](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm "Wiki page"); with k=1) which we want to use to recognize a set of hand written digits.  
To be able to perform the digit recognition we have two sets of data, the first dataset is called the **training set** and consists of 5000 labeled images of hand written digits. The second dataset is the **validation set**, it is comprised of 500 labeled images of hand written digits. The goal of the program is to figure out which digit is written in each picture present in the validation set, this is done by comparing this picture to all 5000 pictures of the training set. We return the digit present on the image of the training set that has the closest match to the image from the validation set. We know the digit of every image from the training set and the validation set but will only use the digit of the validation set to check if the guessed digit is correct.
Each digit example is a set of 28X28 grayscale pixels (we manipulate a vector of 784 integer values from 0 to 255) and to determine the closest match of an item in the validation set when compared to every example in the training set we simply pick the vector with the shortest [distance](https://en.wikipedia.org/wiki/Euclidean_distance) from that item.
Even with this very simple algorithm we can get 94.4% accuracy rate, meaning can recognize 492 out of the 500 digits we were given.
> This example was inspired by an [article](https://philtomson.github.io/blog/2014/05/29/comparing-a-machine-learning-algorithm-implemented-in-f-number-and-ocaml/) written by Phil Tomson, comparing performance between F# and OCaml when running this algorithm in both languages. The dataset was taken from an F# [coding dojo](https://github.com/c4fsharp/Dojo-Digits-Recognizer "GitHub").

### Program Description

#### Data extraction

The code can be split into two parts, the first part, which is the least interesting for the demonstration of the OpenCL interoperability mode for triSYCL, is the parsing of the files to extract the image data. 
First of all we define a data structure to hold the information, this is done by the structure `Img` which contains a field `label` which is the digit written in the image and `pixels` which is an array of 784 integers with values from 0 to 255. 
 The two sets of images are contained in two .csv files, in each file every line contains 785 integers separated by commas, this first integer is the label and the last 784 integers are the pixel values. The parsing of the files is done by the function `slurp_file` which takes a file name as an argument and returns a vector of `Img`'s.

Once the data is properly loaded we have two vectors of `Img`'s, one containing 5000 images which is the training set and the other containing 500 images which is the validation set. To be able to use the data with an OpenCL device we are required to use SYCL buffers, meaning we have to transfer the data in `cl::sycl::buffer<int>` objects, for the training set we use the function `get_buffer` to go from a `vector<Img>` to a  `buffer<int>`. 

#### Preparation steps

The data is now properly processed to be used in the computation. The first thing we need is a **queue**, since triSYCL is using Boost Compute as a backend for OpenCL computation we declare a new queue  by giving a `boost::compute::systen::default_queue()` to the constructor. 
In the same vein, we use `boost::compute::program::create_with_source` to get a boost compute program object, we give a string literal describing the OpenCL kernel code and the default boost compute **context** to the function. We then need to compile the **kernel**, this is done by calling `build` which is a member function of the `boost::compute::program` class.
Once the kernel is compiled we create a `cl::sycl::kernel` which is a wrapper around a `boost::compute::kernel` which takes a compiled kernel and the name of the function of the kernel that will be called as arguments for the constructor. 

#### Kernel Description

In this example the OpenCL kernel is quite simple, it takes the training set and an image as integer arrays along with the the respective sizes of the arrays. The kernel is designed to run in parallel with one thread for each image present in the training set. Each thread will compute the euclidean distance between a single vector in the training set and the image for which we want to guess the digit. The distance is then written in an array the size of the number of images present in the training set, the element at the index i of the array will contain the distance between the i-th picture of the training set and the data.
> In our case we will run 5000 parallel threads for every one of the 500 pictures from the validation set.

#### Computation

We now have all the elements need to perform the computation. The function `compute` takes the training set, an `Img`, a `cl::sycl::queue` and a `cl::sycl::kernel` as arguments.  It allocates an integer array the size of the training set that will hold the result of the computation, then it creates two `cl::sycl::buffer`s, one for the result and one for the data containing the digit we want to recognize. This is easily done by declaring A and B as such :
``` c++
buffer<int> A { std::begin(img.pixels), std::end(img.pixels) };
buffer<int> B { res, training_set_size };
```

We now only need to perform the computation, to do that we submit to the queue a c++ lamda expression that takes a reference to a `cl::sycl::handler` as an argument. The only thing remaining to do is to call the `set_args` method from the handler and give it the buffers and integer values corresponding to the kernel argument while indicating the access modes for the buffers. In our case we want the training set and the data to be read and the result buffer to be written to, the call to set the argument look like this :
``` c++
cgh.set_args(training.get_access<access::mode::read>(cgh),
             A.get_access<access::mode::read>(cgh),
             B.get_access<access::mode::write>(cgh),
             5000, 784);
```
To launch the kernel we simply call ` cgh.parallel_for(5000, k);`, the first argument is the number of parallel threads that will be running and the second is a reference to `cl::sycl::kernel`.
> We can also give a `cl::sycl::range<N>` as a first argument to `parallel_for` to indicate a custom N-D range and work group size for the kernel execution.

To be sure we read the result when the computation has ended we call `q.wait()` which will return only once all the previous commands present in the queue have been executed, making it a synchronization point.

The last step of the computation is iterating over the array containing the results and finding which index of the training set contains the image which vector has the closest distance to the input data. After that we compare the label present at the index we just found to the index from the input image. If the two labels match it means we have guessed to write digit and we return a one if we didn't we return 0.

We can them see how many right guesses we had and get an accuracy rate, which should always be 94.4% with the given dataset 

#### Optimizations made to triSYCL

To have an optimal performance me must minimize the number of transfers from the host to the device. In our example the training set is a large buffer containing 3920000 integers, for every one of the 500 images in the validation set we use the same training set, this means that we can save a lot of time by transferring the training set to the device once for the first image and then reuse it for every subsequent computation, leaving only the 784 integers of the image to be transferred.  In an earlier version of triSYCL the training set was transferred every time leading to poor performance, we had to modify triSYCL to prevent this from happening.

To prevent unneeded transfers from happening, a caching mechanism was implemented, every buffer keeps track of all the contexts it was used in. When we request a `cl_buffer` from a buffer we check if there is an existing `cl_buffer` associated with the given context and only request an allocation and transfer to the device if there isn't one or if the buffer has been written to prior to the call. To accomplish that, further implementation of the `cl::sycl::context` class had to be made.


### Results

In this part we show the results obtained with different triSYCL modes and OpenCL runtimes.

#### Numbers
| triSYCL Mode                    | Execution Time (ms)| Avg Per Image (ms) | Gain w/ Opt  |
| :------------------------------ |:------------------:| :-----------------:| :-----------:|
| OpenCL triSYCL Optimized (CPU)  |                    | 0.6                |              |
| ComputeCPP (CPU                 |                    | 0.8                |              | 
| OpenMP triSYCL                  |                    | 40                 |              |
| OpenCL (CPU)                    |                    | 0.4                |              |
                                                                                            

> TODO : put a nice graph here

-----------------

#### Explanations

The different triSYCL modes of the first column are  :

* OpenCL triSYCL CPU : Running the Intel OpenCL runtime with the **i7-6700HQ**
* OpenCL triSYCL iGPU : Running the beignet OpenCL implementation with the **Intel® HD Graphics 530** integrated graphics of the skylake processor
* OpenCL triSYCL GPU : Running with the Nvidia OpenCL runtime with the **GTX 960M**
* OpenMP triSYCL : Running triSYCL without the OpenCL interoperability mode but with OpenMP
* OpenCL (CPU/iGPU/CPU) : Running "pure" OpenCL code on the hardware without triSYCL or Boost Compute 

The next three data columns correspond to :

* Execution Time : Real time in milliseconds used by the process as measured by `time %e`, this includes the time taken to process the files
*  Avg per Image : Average time in milliseconds taken to process one image, this include the transfers to and from the device and the computing time, this is mesured with `boost::posix_time::ptime` and `boost::posix_time::time_duration`
* Gain w/ opt : The improvement in term of speed, observed when going from the unoptimized to the optimized version of triSYCL, exept for the pure OpenCL implementation in which the value is the improvement over the optimized triSYCL version


#### Conclusion

> TODO : Conclude once the OpenCL number are available
