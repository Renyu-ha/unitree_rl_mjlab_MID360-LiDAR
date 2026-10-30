// Go2W SDK extension; source of truth: patches/unitree_sdk2/include.
#pragma once
#include <chrono>
#include <stdexcept>
#include <string>
namespace unitree::robot::go2w::detail {
inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
}
