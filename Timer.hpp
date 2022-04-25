#pragma once
#include <chrono>

namespace {
using namespace std::chrono;

class Timer {
public:
  Timer() {
    start = steady_clock::now();
  }

  double count() {
    auto now = steady_clock::now();
    return duration<double>(now - start).count();
  }

  double operator()() {
    return count();
  }

  double restart() {
    auto now = steady_clock::now();
    auto result = count();
    start = now;
    return result;
  }
private:
  std::chrono::time_point<std::chrono::steady_clock> start;
};


}