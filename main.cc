#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <random>
#include "Timer.hpp"

std::mt19937 mt{std::random_device{}()};
std::uniform_real_distribution zeroOne(0.f, 1.f);
std::uniform_int_distribution digits(0, 9);

void printIterable(const auto& vector) {
  for(const auto& element : vector) {
    std::printf("%f ", element);
  }
  std::puts("");
}

template<typename T = float>
T activateFun(T x) {
  return 1.0 / (1.0 + std::exp(-x));
}

template<typename T = float>
T activateFunDerivative(T x) {
  return activateFun(x) * (1 - activateFun(x));
}


template<typename T = float>
T interpolate(T x, T min1, T max1, T min2, T max2) {
  return (x - min1) / (max1 - min1) * (max2 - min2) + min2;
}

template<typename T = float>
class NeuralNetwork {
  using Neuron = std::pair<T, T>;
  struct Layer {
    std::vector<Neuron> neurons;
    std::vector<T> weights;
  };
public:
  std::vector<Layer> layers;
  
  void addLayer(std::size_t size) {
    if(layers.size() > 0) {
      auto weightsCount = size * layers.back().neurons.size();
      auto& weights = layers.back().weights;
      weights = std::vector<T>(weightsCount);

      std::generate(weights.begin(), weights.end(), [] {
        return zeroOne(mt);
      });
      std::printf("WeightsCount = %i\n", weightsCount);
    }

    Layer layer;
    layer.neurons = std::vector<Neuron>(size, {0.f, 0.f});
    layers.push_back(std::move(layer));
  }


  void printInfo() {
    std::printf("Layer Neurons Weights\n");
    for(int i = 0; auto layer : layers) {
      std::printf(
        "%5llu %6llu %7llu\n",
        i,
        layer.neurons.size(),
        layer.weights.size()
      );
      i++;
    }
  }
  void printBiasState() {
    std::puts("Bias state:");
    for(const auto& layer : layers) {
      for(const auto& neuron : layer.neurons) {
        std::printf("%.2f ", neuron.second);
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
        if(i >= layer.neurons.size()) {
          i = 0;
          std::puts("");
        }
      }
      std::puts("----------");
    }
  }

  auto execute(const std::vector<T>& input, const std::vector<T>& targetOutput) {
    if(layers.size() < 2) {
      throw std::runtime_error("Too few layers");
    }
    if(input.size() != layers.front().neurons.size()) {
      throw std::runtime_error("Given input layer does not equal with defined one");
    }
    if(targetOutput.size() != layers.back().neurons.size()) {
      throw std::runtime_error("Given target output layer does not equal with defined one");
    }


    Timer timer;
    auto& firstLayerInput = layers.front().neurons;
    for(auto it = input.begin(); auto& [value, bias] : firstLayerInput) {
      value = *it;
      std::advance(it, 1);
    }

    for(auto it = layers.begin(); it != layers.end() - 1; it++) {
      propagate(*it, *std::next(it, 1));
    }

    auto& lastLayerOutput = layers.back().neurons;
    std::vector<T> scores(lastLayerOutput.size(), 0.f);

    for(std::size_t i = 0; i < targetOutput.size(); i++) {
      auto diff = targetOutput[i] - lastLayerOutput[i].first;
      scores[i] = diff * diff;
    }
    auto score = std::accumulate(scores.begin(), scores.end(), 0.f);

    for(auto it = layers.rbegin(); it != layers.rend() - 1; it++) {
      backPropagate(*it, *std::next(it, 1), scores);
    }
    
    if(score < 3.0) {
      puts("Score < 3.0");
      printIterable(input);
      for(auto element : lastLayerOutput) {
        std::printf("%f ", element.first);
      }
      std::puts("");
    }
    //std::printf("Time taken: %lfs\n", timer.count());

    return score;
  }

  void propagate(const Layer& layer1, Layer& layer2) {
    auto sums = countLayerSums(layer1, layer2.neurons.size());
    for(std::size_t i = 0; auto& neuron : layer2.neurons) {
      neuron.first = activateFun(sums[i] + neuron.second);
      i++;
    }
  }

  auto countLayerSums(const Layer& layer, auto nextLayerSize) {
    auto sums = std::vector<T>(nextLayerSize, 0.f);

    for(std::size_t i = 0; const auto& neuron : layer.neurons) {
      for(auto& sum : sums) {
        sum += neuron.first * layer.weights[i];
        i++;
      }
    }
    return std::move(sums);
  }

  void backPropagate(Layer& layer2, Layer& layer1, auto& target) {
    auto sums = countLayerSums(layer1, layer2.neurons.size());

    auto i = 0;
    auto sumIt = sums.begin();
    auto it = target.begin();
    for(auto& neuron2 : layer2.neurons) {
      auto dc_da = 2.f * (neuron2.first - *it);
      auto da_dz = activateFunDerivative(*sumIt);

      auto dc_dz = dc_da * da_dz;

      for(auto& neuron1 : layer1.neurons) {
        auto dz_dw = neuron1.first;
        auto dz_da = layer1.weights[i];

        // 3 dc
        auto dc_dw = dc_dz * dz_dw;
        auto dc_da = dc_dz * dz_da;
        auto dc_db = dc_dz;

        layer1.weights[i] -= dc_dw;
        neuron1.first  -= dc_da;
        neuron1.second -= dc_db;
        i++;
      }

      std::advance(it, 1);
      std::advance(sumIt, 1);
    }
  }

  void dumpToFile() {
  }
  
  void loadFromFile() {
  }
};


void printIterablePair(const auto& vector) {
  for(const auto& [first, second] : vector) {
    std::printf("%f %f", first, second);
  }
  std::puts("");
}


int main(int argc, char** argv) {
  int count = 3000;
  NeuralNetwork n;
  std::vector<
  std::vector<float>> A;
  for(int i = 0; i < count; i++) {
    auto v = std::vector<float>(10, 0.f);
    v[digits(mt)] = 1.0f;
    A.push_back(std::move(v));
  }

  n.addLayer(10);
  n.addLayer(8);
  n.addLayer(8);
  n.addLayer(10);
  n.printInfo();
  //n.printWeightState();
  //n.printBiasState();
  auto desiredVector = std::vector<float>(10, 0.0f);
  desiredVector[1] = 1.0f;
  for(int i = 0;; i++) {
    if(i >= count) {
      i = 0;
    }
    auto score = n.execute(
      A[i],
      A[i]
    );
    if(i == 0) {
      printf("Score %f\n", score);
    }
  }
  return 0;
}
