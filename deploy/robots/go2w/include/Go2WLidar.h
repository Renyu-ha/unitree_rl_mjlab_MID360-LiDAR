// Adapted from g1/src/State_RLBase.cpp's LidarHeightScanUpdater.
#pragma once
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <unitree/idl/ros2/PointCloud2_.hpp>
#include <unitree/idl/ros2/Odometry_.hpp>
#include "Types.h"
#include "go2w_core.h"
#include <unitree/dds_wrapper/robots/go2w/go2w_sensor.h>

namespace go2w {
inline std::string dds_topic(std::string topic) {
    if (topic.rfind("rt/", 0) == 0) return topic;
    while (!topic.empty() && topic.front() == '/') topic.erase(topic.begin());
    return "rt/" + topic;
}
inline Grid load_grid(const YAML::Node& cfg) {
    Grid grid;
    grid.size_x = cfg["size_x_m"].as<float>(); grid.size_y = cfg["size_y_m"].as<float>();
    grid.resolution = cfg["resolution_m"].as<float>();
    grid.offset_x = cfg["x_offset_m"].as<float>(); grid.offset_y = cfg["y_offset_m"].as<float>();
    grid.miss = cfg["miss_value_m"].as<float>(); grid.mask_visible = cfg["mask_visible"].as<bool>();
    grid.visible_min_x = cfg["visible_x_min_m"].as<float>();
    grid.visible_max_x = cfg["visible_x_max_m"].as<float>();
    grid.visible_abs_y = cfg["visible_y_abs_m"].as<float>();
    grid.visible_mask = cfg["visible_mask"].as<std::vector<int>>();
    grid.validate();
    return grid;
}

class LidarHeightScanUpdater {
public:
    using Cloud = sensor_msgs::msg::dds_::PointCloud2_;
    using Odom = nav_msgs::msg::dds_::Odometry_;
    LidarHeightScanUpdater(YAML::Node hardware, YAML::Node policy)
        : cfg_(hardware), grid_(load_grid(policy["height_scan"])),
          max_age_(policy["height_scan"]["memory_max_age_s"].as<float>()),
          pose_mode_(hardware["memory_mode"].as<std::string>("training_forward") == "pose"),
          memory_(grid_, max_age_, pose_mode_) {
        auto mode = hardware["memory_mode"].as<std::string>("training_forward");
        require(mode == "pose" || mode == "training_forward", "Unknown lidar memory_mode");
        auto t = cfg_["extrinsics"]["translation_m"].as<std::vector<float>>();
        auto r = cfg_["extrinsics"]["rpy_deg"].as<std::vector<float>>();
        require(t.size() == 3 && r.size() == 3, "Lidar extrinsics need three translation/RPY values");
        finite(t, "Lidar translation"); finite(r, "Lidar rotation");
        translation_ = Eigen::Vector3f(t[0], t[1], t[2]);
        constexpr float rad = 0.017453292519943295f;
        rotation_ = Eigen::AngleAxisf(r[2]*rad, Eigen::Vector3f::UnitZ()) *
                    Eigen::AngleAxisf(r[1]*rad, Eigen::Vector3f::UnitY()) *
                    Eigen::AngleAxisf(r[0]*rad, Eigen::Vector3f::UnitX());
        if (cfg_["self_filter_boxes"]) for (auto box : cfg_["self_filter_boxes"]) {
            auto values = box.as<std::vector<float>>();
            require(values.size() == 6, "Self-filter boxes need xmin,xmax,ymin,ymax,zmin,zmax");
            finite(values, "Self-filter box");
            std::array<float,6> bounds;
            std::copy(values.begin(), values.end(), bounds.begin());
            boxes_.push_back(bounds);
        }
        cloud_sub_ = std::make_unique<unitree::robot::go2w::subscription::Sensor<Cloud>>(cfg_["domain_id"].as<int>(),dds_topic(cfg_["topic"].as<std::string>()),[this](const void* data) {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto& msg = *static_cast<const Cloud*>(data);
            const auto& stamp = msg.header().stamp();
            int64_t key = int64_t(stamp.sec()) * 1000000000 + stamp.nanosec();
            if (key != 0 && key == cloud_stamp_) return;
            cloud_stamp_ = key; cloud_ = msg; cloud_at_ = now(); ++generation_;
        });
        if (max_age_ > 0 || pose_mode_) {
            odom_sub_ = std::make_unique<unitree::robot::go2w::subscription::Sensor<Odom>>(cfg_["domain_id"].as<int>(),dds_topic(cfg_["odometry_topic"].as<std::string>()),[this](const void* data) {
                std::lock_guard<std::mutex> lock(mutex_);
                const auto& msg = *static_cast<const Odom*>(data);
                const auto& stamp = msg.header().stamp();
                int64_t key = int64_t(stamp.sec()) * 1000000000 + stamp.nanosec();
                if (key != 0 && key == odom_stamp_) return;
                odom_stamp_ = key; odom_ = msg; odom_at_ = now();
            });
        }
    }
    ~LidarHeightScanUpdater() {
        cloud_sub_->CloseChannel();
        if (odom_sub_) odom_sub_->CloseChannel();
    }
    void reset() { memory_.reset(); last_generation_ = 0; }
    std::vector<float> update(const Eigen::Quaternionf& body, float dt) {
        Cloud cloud; Odom odom; uint64_t generation; double cloud_at, odom_at;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cloud = cloud_; odom = odom_; generation = generation_;
            cloud_at = cloud_at_; odom_at = odom_at_;
        }
        const double time = now();
        require(cloud_at > 0 && time - cloud_at < cfg_["pointcloud_timeout_ms"].as<float>()/1000,
                "MID360 point cloud missing/stale");
        require(cloud.header().frame_id() == cfg_["frame_id"].as<std::string>(), "Unexpected point cloud frame_id; check lidar extrinsics");
        Pose pose;
        float vx = 0;
        if (odom_sub_) {
            require(odom_at > 0 && time - odom_at < cfg_["odometry_timeout_ms"].as<float>()/1000, "Body odometry missing/stale");
            require(odom.child_frame_id() == cfg_["body_frame"].as<std::string>(), "Odometry twist must use the configured body frame");
            const auto& p = odom.pose().pose().position();
            const auto& q = odom.pose().pose().orientation();
            Eigen::Quaternionf attitude(q.w(), q.x(), q.y(), q.z());
            require(attitude.coeffs().allFinite() && attitude.norm() > 0.9f && attitude.norm() < 1.1f, "Invalid odometry orientation");
            attitude.normalize();
            auto matrix = attitude.toRotationMatrix();
            pose = {float(p.x()), float(p.y()), float(p.z()), std::atan2(matrix(1,0), matrix(0,0))};
            vx = float(odom.twist().twist().linear().x());
            require(std::isfinite(pose.x) && std::isfinite(pose.y) && std::isfinite(pose.z) && std::isfinite(vx), "Invalid body odometry");
        }
        bool fresh = generation != last_generation_;
        if (fresh) {
            current_scan_ = project(cloud, body);
            last_generation_ = generation;
        }
        require(observed_ratio_ >= cfg_["min_observed_ratio"].as<float>(), "Insufficient valid height-scan coverage");
        if (max_age_ <= 0) return current_scan_;
        return memory_.update(current_scan_, fresh, vx, dt, pose);
    }
private:
    std::vector<float> project(const Cloud& msg, const Eigen::Quaternionf& body) {
        std::array<int, 3> offsets{-1,-1,-1};
        for (const auto& field : msg.fields()) {
            for (int i = 0; i < 3; ++i)
                if (field.name() == std::string(1, "xyz"[i]) && field.datatype() == 7 && field.count() == 1)
                    offsets[i] = int(field.offset());
        }
        require(msg.point_step() > 0 && msg.width() > 0 && msg.height() > 0, "Empty PointCloud2");
        for (int offset : offsets) require(offset >= 0 && uint32_t(offset+4) <= msg.point_step(), "PointCloud2 needs float32 x/y/z");
        const uint64_t row_bytes = uint64_t(msg.width()) * msg.point_step();
        require(msg.row_step() >= row_bytes && uint64_t(msg.row_step()) * msg.height() <= msg.data().size(), "Truncated PointCloud2 rows");
        auto matrix = body.toRotationMatrix();
        float yaw = std::atan2(matrix(1,0), matrix(0,0));
        Eigen::Quaternionf inverse_yaw(Eigen::AngleAxisf(-yaw, Eigen::Vector3f::UnitZ()));
        std::vector<float> scan(grid_.size(), grid_.miss);
        uint16_t endian_test = 1;
        bool swap = msg.is_bigendian() == (*reinterpret_cast<uint8_t*>(&endian_test) == 1);
        for (uint32_t row = 0; row < msg.height(); ++row) {
            for (uint32_t col = 0; col < msg.width(); ++col) {
                Eigen::Vector3f point;
                size_t base = size_t(row)*msg.row_step() + size_t(col)*msg.point_step();
                for (int i = 0; i < 3; ++i) {
                    std::array<uint8_t,4> bytes;
                    std::copy_n(msg.data().data()+base+offsets[i], 4, bytes.data());
                    if (swap) std::reverse(bytes.begin(), bytes.end());
                    std::memcpy(&point[i], bytes.data(), 4);
                }
                if (!point.allFinite()) continue;
                Eigen::Vector3f local = rotation_*point + translation_;
                bool self = false;
                for (const auto& b : boxes_) {
                    if (local.x() >= b[0] && local.x() <= b[1] && local.y() >= b[2] && local.y() <= b[3] && local.z() >= b[4] && local.z() <= b[5]) self = true;
                }
                if (self) continue;
                Eigen::Vector3f level = inverse_yaw * (body * local);
                int index = grid_.index(level.x(), level.y());
                if (index < 0 || !grid_.visible(index) || -level.z() >= grid_.miss) continue;
                scan[index] = std::min(scan[index], std::max(0.0f,-level.z()));
            }
        }
        int observed = 0;
        for (float value : scan) if (value < grid_.miss) ++observed;
        observed_ratio_ = float(observed) / grid_.size();
        return scan;
    }
    YAML::Node cfg_;
    Grid grid_;
    float max_age_, observed_ratio_ = 0;
    bool pose_mode_;
    HeightMemory memory_;
    Eigen::Vector3f translation_;
    Eigen::Quaternionf rotation_;
    std::vector<std::array<float,6>> boxes_;
    std::unique_ptr<unitree::robot::go2w::subscription::Sensor<Cloud>> cloud_sub_;
    std::unique_ptr<unitree::robot::go2w::subscription::Sensor<Odom>> odom_sub_;
    std::mutex mutex_;
    Cloud cloud_; Odom odom_;
    double cloud_at_ = 0, odom_at_ = 0;
    int64_t cloud_stamp_ = 0, odom_stamp_ = 0;
    uint64_t generation_ = 0, last_generation_ = 0;
    std::vector<float> current_scan_;
};
} // namespace go2w
