// Go2W-specific adapters for the original G1 deployment pipeline.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

namespace go2w {
inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
constexpr int kMotors = 16;
inline const std::array<std::string, kMotors> joint_names = {
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"};
// SDK: FR, FL, RR, RL; policy: FL, FR, RL, RR.
inline const std::array<int, kMotors> sdk_ids = {3,4,5,0,1,2,9,10,11,6,7,8,13,12,15,14};

inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
inline void finite(const std::vector<float>& values, const std::string& name) {
    for (float x : values) require(std::isfinite(x), name + " contains NaN/Inf");
}
inline void validate_map(const std::vector<int>& ids) {
    require(ids.size() == kMotors, "Go2W requires 16 motor IDs");
    auto sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < kMotors; ++i)
        require(sorted[i] == i, "Motor IDs must be a permutation of 0..15");
}
struct MotorTarget { float q = 0, dq = 0, kp = 0, kd = 0, tau = 0; };

// Actions have ALREADY been scaled/offset by the shared ActionManager.
inline std::array<MotorTarget, kMotors> motor_targets(
    const std::vector<float>& actions, const std::vector<int>& ids,
    const std::vector<float>& kp_sdk, const std::vector<float>& kd_sdk) {
    validate_map(ids);
    require(actions.size() == kMotors && kp_sdk.size() >= kMotors && kd_sdk.size() >= kMotors,
            "Invalid Go2W action/gain dimensions");
    finite(actions, "Actions");
    std::array<MotorTarget, kMotors> targets{};
    for (int i = 0; i < kMotors; ++i) {
        auto& m = targets[ids[i]];
        m.kp = i < 12 ? kp_sdk[ids[i]] : 0.0f;
        m.kd = kd_sdk[ids[i]];
        if (i < 12) m.q = actions[i];
        else m.dq = actions[i];
    }
    return targets;
}

struct Grid {
    float size_x = 1.6f, size_y = 1.0f, resolution = 0.1f;
    float offset_x = 0.35f, offset_y = 0, miss = 5;
    float visible_min_x = 0.25f, visible_max_x = 1.15f, visible_abs_y = 0.5f;
    bool mask_visible = true;
    std::vector<int> visible_mask;
    int nx() const { return int(std::lround(size_x / resolution)) + 1; }
    int ny() const { return int(std::lround(size_y / resolution)) + 1; }
    int size() const { return nx() * ny(); }
    static float coordinate(int i, float resolution, float size, float offset) {
        // Match the separate float32 tensor operations, including boundary cells.
        volatile float scaled = i * resolution;
        volatile float centred = scaled - size / 2;
        return centred + offset;
    }
    float x(int ix) const { return coordinate(ix, resolution, size_x, offset_x); }
    float y(int iy) const { return coordinate(iy, resolution, size_y, offset_y); }
    int index(float px, float py) const {
        // Grid values represent sample centres, including both endpoints.
        int ix = int(std::lround((px - offset_x + size_x / 2) / resolution));
        int iy = int(std::lround((py - offset_y + size_y / 2) / resolution));
        return ix < 0 || ix >= nx() || iy < 0 || iy >= ny() ? -1 : iy * nx() + ix;
    }
    bool visible(int i) const {
        if (!visible_mask.empty()) return visible_mask.at(i) != 0;
        return !mask_visible || (x(i % nx()) >= visible_min_x &&
            x(i % nx()) <= visible_max_x && std::abs(y(i / nx())) <= visible_abs_y);
    }
    void validate() const {
        require(std::isfinite(resolution) && resolution > 0 && size_x > 0 && size_y > 0 &&
                size_x / resolution < 100 && size_y / resolution < 100 && miss > 0,
                "Invalid height-scan grid");
        require(visible_mask.empty() || visible_mask.size() == size_t(size()), "Invalid visibility mask");
    }
};
struct Pose { float x = 0, y = 0, z = 0, yaw = 0; };

class HeightMemory {
public:
    HeightMemory(Grid grid, float max_age, bool compensate_pose = false)
        : grid_(grid), max_age_(max_age), pose_mode_(compensate_pose) {
        grid_.validate();
        reset();
    }
    void reset() {
        values_.assign(grid_.size(), grid_.miss);
        ages_.assign(grid_.size(), 1e6f);
        residual_ = 0;
        have_pose_ = false;
    }
    std::vector<float> update(const std::vector<float>& scan, bool fresh,
                             float vx, float dt, const Pose& pose) {
        require(scan.size() == values_.size(), "Height scan dimension mismatch");
        finite(scan, "Height scan");
        require(std::isfinite(vx) && std::isfinite(dt) && dt > 0, "Invalid memory motion");
        if (pose_mode_ && have_pose_) reproject(pose);
        else if (!pose_mode_) {
            // Exactly the training convention: positive forward displacement only.
            residual_ += std::max(vx * dt, 0.0f);
            const int shift = int(std::floor(residual_ / grid_.resolution));
            residual_ -= shift * grid_.resolution;
            if (shift > 0) {
                for (int iy = 0; iy < grid_.ny(); ++iy)
                    for (int ix = 0; ix < grid_.nx(); ++ix) {
                        int dst = iy * grid_.nx() + ix, src = dst + shift;
                        values_[dst] = ix + shift < grid_.nx() ? values_[src] : grid_.miss;
                        ages_[dst] = ix + shift < grid_.nx() ? ages_[src] : 1e6f;
                    }
            }
        }
        for (int i = 0; i < grid_.size(); ++i) {
            ages_[i] += dt;
            if (fresh && grid_.visible(i)) { values_[i] = scan[i]; ages_[i] = 0; }
            if (ages_[i] > max_age_ + 1e-6f) values_[i] = grid_.miss;
        }
        previous_pose_ = pose;
        have_pose_ = true;
        return values_;
    }
private:
    void reproject(const Pose& pose) {
        std::vector<float> next(grid_.size(), grid_.miss), ages(grid_.size(), 1e6f);
        float c0 = std::cos(previous_pose_.yaw), s0 = std::sin(previous_pose_.yaw);
        float c1 = std::cos(pose.yaw), s1 = std::sin(pose.yaw);
        for (int i = 0; i < grid_.size(); ++i) {
            if (ages_[i] > max_age_ || values_[i] >= grid_.miss) continue;
            float x = grid_.x(i % grid_.nx()), y = grid_.y(i / grid_.nx());
            float wx = previous_pose_.x + c0*x - s0*y - pose.x;
            float wy = previous_pose_.y + s0*x + c0*y - pose.y;
            int dst = grid_.index(c1*wx + s1*wy, -s1*wx + c1*wy);
            float distance = values_[i] + pose.z - previous_pose_.z;
            if (dst >= 0 && distance >= 0 && distance < next[dst]) {
                next[dst] = distance;
                ages[dst] = ages_[i];
            }
        }
        values_ = std::move(next);
        ages_ = std::move(ages);
    }
    Grid grid_;
    float max_age_, residual_ = 0;
    bool pose_mode_, have_pose_ = false;
    Pose previous_pose_;
    std::vector<float> values_, ages_;
};
} // namespace go2w
