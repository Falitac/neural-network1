#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <random>
#include "Timer.hpp"

std::mt19937 mt{std::random_device{}()};

std::uniform_real_distribution zeroOne(0.f, 1.f);



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
  struct Layer {
    std::vector<T> biases;
    std::vector<T> weights;
  };
public:
  std::vector<Layer> layers;
  
  void addLayer(std::size_t size) {
    if(layers.size() > 0) {
      auto weightsCount = size * layers.back().biases.size();
      auto& weights = layers.back().weights;
      weights = std::vector<T>(weightsCount);
      std::generate(weights.begin(), weights.end(), [] {
        return zeroOne(mt);
      });
      std::printf("weightsCount = %i\n", weightsCount);
    }

    Layer layer;
    layer.biases = std::vector<T>(size, 0.0f);
    layers.push_back(std::move(layer));
  }

  void printInfo() {
    std::printf("Layer Biases Weights\n");
    for(int i = 0; auto layer : layers) {
      std::printf(
        "%5llu %6llu %7llu\n",
        i,
        layer.biases.size(),
        layer.weights.size()
      );
      i++;
    }
  }

  void printBiasState() {
    std::puts("Bias state:");
    for(const auto& layer : layers) {
      for(const auto& bias : layer.biases) {
        std::printf("%.2f ", bias);
      }
      std::puts("");
    }
  }
  void printWeightState() {
    std::puts("Weight state:");
    for(const auto& layer : layers) {
      for(std::size_t i = 0; const auto& weight : layer.weights) {
        std::printf("%.2f ", weight);
        i++;
        if(i >= layer.biases.size()) {
          i = 0;
          std::puts("");
        }
      }
      std::puts("----------");
    }
  }

  // @TODO output target
  auto propagate(std::vector<T> inputBiases, std::vector<T> targetOutput) {
    if(layers.size() < 2) {
      throw std::runtime_error("Too few layers");
    }
    if(inputBiases.size() != layers.front().biases.size()) {
      throw std::runtime_error("Given input layer does not equal with defined one");
    }
    if(targetOutput.size() != layers.back().biases.size()) {
      throw std::runtime_error("Given target output layer does not equal with defined one");
    }

    Timer timer;
    layers.front().biases = inputBiases;

    for(auto it = layers.begin(); it != layers.end() - 1; it++) {
      calcInterLayer(*it, *std::next(it, 1));
    }

    T score = 0.0;
    for(std::size_t i = 0; i < targetOutput.size(); i++) {
      auto diff = targetOutput[i] - layers.back().biases[i];
      score += diff * diff;
    }
    
    std::printf("Time taken: %lfs\n", timer.count());
    return score;
  }

  void calcInterLayer(const Layer& layer1, Layer& layer2) {
    auto sums = std::vector<T>(layer2.biases.size(), 0.f);

    for(std::size_t i = 0; const auto& bias1 : layer1.biases) {
      for(auto& sum : sums) {
        sum += bias1 * layer1.weights[i];
        i++;
      }
    }
    for(std::size_t i = 0; auto& bias2 : layer2.biases) {
      bias2 = activateFun(sums[i] + bias2);
      i++;
    }
  }

  void dumpToFile() {
  }
  
  void loadFromFile() {
  }
};

void printIterable(const auto& vector) {
  for(const auto& element : vector) {
    std::printf("%lf ", element);
  }
  std::puts("");
}


int main(int argc, char** argv) {
  NeuralNetwork n;
  n.addLayer(28*28);
  n.addLayer(128);
  n.addLayer(128);
  n.addLayer(10);
  n.printInfo();
  //n.printWeightState();
  //n.printBiasState();
  auto desiredVector = std::vector<float>(10, 0.0f);
  desiredVector[1] = 1.0f;
  auto score = n.propagate(
    std::vector<float>(28*28, 0.f),
    desiredVector
  );
  printIterable(n.layers.back().biases);
  std::printf("Score: %f\n", score);
  return 0;
}
