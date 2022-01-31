#include <iostream>
#include <vector>
#include <array>
#include <cmath>


template<typename T = float>
T activateFun(T x) {
  return 1.0 / (1.0 + std::exp(-x));
}


template<typename T = float>
T interpolate(T x, T min1, T max1, T min2, T max2) {
  return (x - min1) / (max1 - min1) * (max2 - min2) + min2;
}

/*
  it needs assessment
*/
template<typename T = float>
class NeuralNetwork {
public:
  std::vector<T> layer1;
  std::vector<T> layer2;
  
  void setLayersSize(size_t layer1_size, size_t layer2_size) {
    layer1 = std::vector<T>(layer1_size);
    layer2 = std::vector<T>(layer2_size);
  }

  auto calcResult(std::vector<T>& input, size_t out_size) {
    auto result = std::vector<T>(out_size);

    for(auto& destination : layer1) {
      T result = 0.0;
      for(const auto& source : input) {
        result += source;
      }
      destination = activateFun(result) - destination;
    }

    for(auto& destination : layer2) {
      T result = 0.0;
      for(const auto& source : layer1) {
        result += source;
      }
      destination = activateFun(result) - destination;
    }

    for(auto& destination : result) {
      T result = 0.0;
      for(const auto& source : layer2) {
        result += source;
      }
      destination = activateFun(result) - destination;
    }
    return std::move(result);
  }
};



int main(int argc, char** argv) {
  NeuralNetwork n;
  n.setLayersSize(1024, 1024);
  std::vector<float> input(16*16);
  for(int i = 0; ; i++) {
    auto output = n.calcResult(input, 10);
    for(int j = 0; auto& element : output) {
      std::printf("%i: %f\n", j, element);
      j++;
    }
  }

  return 0;
}
