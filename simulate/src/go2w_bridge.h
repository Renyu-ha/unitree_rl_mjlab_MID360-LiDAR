#pragma once
#include <cmath>
#include <mujoco/mujoco.h>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include <unitree/dds_wrapper/robots/go2w/go2w_sensor.h>
#include <unitree/dds_wrapper/robots/go2w/go2w_crc.h>
#include <unitree/dds_wrapper/robots/go2w/go2w_channel.h>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/ros2/PointCloud2_.hpp>
#include <unitree/idl/ros2/Odometry_.hpp>
#include <unitree/idl/ros2/String_.hpp>
#include <atomic>
#include <cstring>
#include <fstream>
#include <mutex>
#include "param.h"
#include "go2w_core.h"
#include "physics_joystick.h"

// Like G1Bridge, this is a robot-specific SDK/MuJoCo adapter. The physics
// thread calls it under the simulator mutex, avoiding concurrent mjData access.
class Go2WBridge {
public:
    using State = unitree_go::msg::dds_::LowState_;
    using Cmd = unitree_go::msg::dds_::LowCmd_;
    using Cloud = sensor_msgs::msg::dds_::PointCloud2_;
    using Odom = nav_msgs::msg::dds_::Odometry_;
    Go2WBridge(mjModel* m, mjData* d)
        : states_("rt/lowstate"), clouds_(param::config.domain_id,"rt/go2w/sim/cloud"), odometry_(param::config.domain_id,"rt/go2w/odometry"), commands_("rt/lowcmd",param::config.domain_id) {
        go2w::require(param::config.domain_id > 0 && param::config.interface == "lo", "Go2W simulator requires loopback and nonzero DDS domain");
        states_.InitChannel();
        commands_.InitChannel([this](const void* value) {
            const auto& message = *static_cast<const Cmd*>(value);
            if (message.crc() != unitree::robot::go2w::protocol_crc(message)) return;
            std::lock_guard<std::mutex> lock(mutex_);
            command_ = message; command_at_ = go2w::now(); ++command_count_; command_seen_=true;
        }, 1);
        readiness_=std::make_unique<unitree::robot::go2w::subscription::Sensor<std_msgs::msg::dds_::String_>>(
            param::config.domain_id,"rt/go2w/sim/ready",[this](const void* value) {
                if (static_cast<const std_msgs::msg::dds_::String_*>(value)->data()=="ready") controller_ready_=true;
            });
        cloud_.header().frame_id() = "base_link";
        cloud_.height() = 1; cloud_.is_bigendian() = false; cloud_.is_dense() = true; cloud_.point_step() = 12;
        for (int i = 0; i < 3; ++i) {
            sensor_msgs::msg::dds_::PointField_ field;
            field.name() = std::string(1, "xyz"[i]); field.offset() = i*4; field.datatype() = 7; field.count() = 1;
            cloud_.fields().push_back(field);
        }
        if (param::config.use_joystick && !param::config.automatic) {
            if (param::config.joystick_type == "xbox") joystick_ = std::make_shared<XBoxJoystick>(param::config.joystick_device, param::config.joystick_bits);
            else joystick_ = std::make_shared<SwitchJoystick>(param::config.joystick_device, param::config.joystick_bits);
        }
        reset_model(m,d);
    }
    ~Go2WBridge() { readiness_->CloseChannel(); commands_.CloseChannel(); }
    void reset_model(mjModel* m, mjData* d) {
        go2w::require(m->nu == 16 && m->nq == 23, "Go2W scene must have 16 actuators and 23 qpos values");
        body_ = mj_name2id(m,mjOBJ_BODY,"base_link");
        go2w::require(body_ > 0, "Go2W base_link is missing");
        for (int i = 0; i < 16; ++i) {
            joints_[i] = mj_name2id(m,mjOBJ_JOINT,go2w::joint_names[i].c_str());
            go2w::require(joints_[i] >= 0, "Go2W joint missing: "+go2w::joint_names[i]);
            actuators_[i] = -1;
            for (int a = 0; a < m->nu; ++a)
                if (m->actuator_trntype[a] == mjTRN_JOINT && m->actuator_trnid[2*a] == joints_[i]) {
                    go2w::require(actuators_[i] < 0, "Multiple actuators for one Go2W joint");
                    actuators_[i] = a;
                }
            go2w::require(actuators_[i] >= 0, "No actuator for Go2W joint");
        }
        int home = mj_name2id(m,mjOBJ_KEY,"home");
        go2w::require(home >= 0, "Generate the Go2W SDK scene with prepare_go2w_sim.py");
        mj_resetDataKeyframe(m,d,home); mj_forward(m,d);
        std::copy_n(d->qpos,7,home_.begin());
        stand_started_ = -1; iteration_ = 0;
    }
    void before_step(mjModel* m, mjData* d) {
        Cmd cmd; bool fresh;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cmd = command_; fresh = command_at_ > 0 && go2w::now()-command_at_ < 0.1;
        }
        if (stand_started_ < 0 && fresh && cmd.motor_cmd()[0].kp() > 0) stand_started_ = d->time;
        for (int i = 0; i < 16; ++i) {
            int j = joints_[i], a = actuators_[i];
            const auto& motor = cmd.motor_cmd()[go2w::sdk_ids[i]];
            double q = d->qpos[m->jnt_qposadr[j]], dq = d->qvel[m->jnt_dofadr[j]];
            double torque = fresh && motor.mode() == 1 ? motor.tau()+motor.kp()*(motor.q()-q)+motor.kd()*(motor.dq()-dq) : -dq;
            go2w::require(std::isfinite(torque), "Non-finite simulated motor torque");
            d->ctrl[a] = std::clamp(torque,m->actuator_ctrlrange[2*a],m->actuator_ctrlrange[2*a+1]);
            if (i >= 12 && fresh) {
                wheel_position_gain_ = std::max(wheel_position_gain_, std::abs(motor.kp()));
                max_wheel_target_ = std::max(max_wheel_target_, std::abs(motor.dq()));
                if (std::abs(motor.dq())>0.001f) wheel_targets_[go2w::sdk_ids[i]-12]=motor.dq();
            }
        }
    }
    void after_step(mjModel* m, mjData* d) {
        if (stand_started_ < 0 || d->time-stand_started_ < param::config.support_seconds) {
            std::copy(home_.begin(),home_.end(),d->qpos);
            std::fill_n(d->qvel,6,0.0); mj_forward(m,d);
        }
        mjtNum velocity[6]; mj_objectVelocity(m,d,mjOBJ_BODY,body_,velocity,1);
        State state{};
        state.head() = {0xFE,0xEF}; state.level_flag() = 0xFF; state.tick() = uint32_t(d->time*1000);
        for (int i = 0; i < 16; ++i) {
            auto& motor = state.motor_state()[go2w::sdk_ids[i]];
            motor.q() = d->qpos[m->jnt_qposadr[joints_[i]]];
            motor.dq() = d->qvel[m->jnt_dofadr[joints_[i]]];
            motor.tau_est() = d->actuator_force[actuators_[i]];
            go2w::require(std::isfinite(motor.q()) && std::isfinite(motor.dq()), "Non-finite simulated joint state");
        }
        for (int i = 0; i < 4; ++i) state.imu_state().quaternion()[i] = d->xquat[4*body_+i];
        for (int i = 0; i < 3; ++i) state.imu_state().gyroscope()[i] = velocity[i];
        unitree::common::REMOTE_DATA_RX remote{};
        if (joystick_) { joystick_->update(); remote = joystick_->combine(); }
        if (param::config.automatic) {
            if (auto_started_<0 && controller_ready_) auto_started_=d->time;
            double phase = auto_started_<0 ? -1 : d->time-auto_started_;
            remote.RF_RX.btn.components.L2 = phase > 0.1 && phase < 1.0;
            remote.RF_RX.btn.components.up = phase > 0.6 && phase < 1.0;
            remote.RF_RX.btn.components.R2 = phase > 1.2;
            remote.RF_RX.btn.components.A = phase > 4.6 && phase < 5.0;
            if (phase > 5.0) {
                remote.RF_RX.ly=param::config.command_vx;
                remote.RF_RX.lx=-param::config.command_vy;
                remote.RF_RX.rx=-param::config.command_wz;
            }
            if (phase>=param::config.estop_after) {
                remote.RF_RX.btn.components.L2=true;
                remote.RF_RX.btn.components.B=true;
            }
        }
        std::memcpy(state.wireless_remote().data(),remote.buff,40);
        state.crc() = unitree::robot::go2w::protocol_crc(state);
        const double scenario_time=elapsed(d->time);
        if (scenario_time >= param::config.corrupt_state_after) state.crc() ^= 1;
        if (scenario_time < param::config.drop_state_after) { states_.Write(state); ++state_count_; }
        if (iteration_ % 10 == 0 && scenario_time < param::config.drop_odom_after) publish_odometry(d,velocity);
        if (iteration_ % 50 == 0 && scenario_time < param::config.drop_cloud_after) publish_cloud(m,d);
        ++iteration_;
        min_height_ = std::min(min_height_,float(d->qpos[2]));
    }
    void write_report(const std::filesystem::path& file, double time) {
        if (file.empty()) return;
        std::lock_guard<std::mutex> lock(mutex_);
        std::ofstream out(file);
        go2w::require(bool(out), "Cannot write simulation report");
        out << "{\"simulation_seconds\":" << time << ",\"commands\":" << command_count_
            << ",\"states\":" << state_count_ << ",\"clouds\":" << cloud_count_ << ",\"odometry\":" << odom_count_
            << ",\"max_wheel_kp\":" << wheel_position_gain_ << ",\"max_wheel_target\":" << max_wheel_target_
            << ",\"min_base_height\":" << min_height_
            << ",\"wheel_targets_sdk\":[" << wheel_targets_[0] << ',' << wheel_targets_[1] << ','
            << wheel_targets_[2] << ',' << wheel_targets_[3] << "]}\n";
    }
    double elapsed(double time) const { return param::config.automatic ? (auto_started_<0 ? 0 : time-auto_started_) : time; }
private:
    void publish_odometry(mjData* d, const mjtNum* velocity) {
        Odom odom;
        odom.header().frame_id() = "odom"; odom.child_frame_id() = "base_link";
        odom.header().stamp().sec() = int32_t(d->time);
        odom.header().stamp().nanosec() = uint32_t((d->time-int(d->time))*1e9);
        auto& p = odom.pose().pose().position();
        p.x()=d->xpos[3*body_]; p.y()=d->xpos[3*body_+1]; p.z()=d->xpos[3*body_+2];
        auto& q = odom.pose().pose().orientation();
        q.w()=d->xquat[4*body_]; q.x()=d->xquat[4*body_+1]; q.y()=d->xquat[4*body_+2]; q.z()=d->xquat[4*body_+3];
        auto& v = odom.twist().twist().linear(); v.x()=velocity[3]; v.y()=velocity[4]; v.z()=velocity[5];
        odometry_.Write(odom); ++odom_count_;
    }
    void publish_cloud(mjModel* m, mjData* d) {
        cloud_.data().clear();
        const float yaw = std::atan2(d->xmat[9*body_+3],d->xmat[9*body_]);
        const mjtNum down[3] = {0,0,-1}; const mjtByte groups[6] = {1,1,0,0,0,0};
        mjtNum inverse[4]; mju_negQuat(inverse,d->xquat+4*body_);
        for (float offset : {0.0f,0.35f}) {
            go2w::Grid grid; grid.offset_x=offset;
            for (int iy=0; iy<grid.ny(); ++iy) for (int ix=0; ix<grid.nx(); ++ix) {
                float x=grid.x(ix), y=grid.y(iy);
                mjtNum start[3] = {d->xpos[3*body_]+std::cos(yaw)*x-std::sin(yaw)*y,
                    d->xpos[3*body_+1]+std::sin(yaw)*x+std::cos(yaw)*y,d->xpos[3*body_+2]};
                int hit;
#if mjVERSION_HEADER >= 3016000
                mjtNum distance=mj_ray(m,d,start,down,groups,1,body_,&hit,nullptr);
#else
                mjtNum distance=mj_ray(m,d,start,down,groups,1,body_,&hit);
#endif
                if (distance<0 || distance>=5) continue;
                mjtNum relative[3]={start[0]-d->xpos[3*body_],start[1]-d->xpos[3*body_+1],-distance}, local[3];
                mju_rotVecQuat(local,relative,inverse);
                for (double value : local) {
                    float scalar=float(value); auto bytes=reinterpret_cast<const uint8_t*>(&scalar);
                    cloud_.data().insert(cloud_.data().end(),bytes,bytes+4);
                }
            }
        }
        cloud_.width()=cloud_.data().size()/12; cloud_.row_step()=cloud_.data().size();
        cloud_.header().stamp().sec()=int32_t(d->time);
        cloud_.header().stamp().nanosec()=uint32_t((d->time-int(d->time))*1e9);
        clouds_.Write(cloud_); ++cloud_count_;
    }
    unitree::robot::ChannelPublisher<State> states_;
    unitree::robot::go2w::publisher::Sensor<Cloud> clouds_;
    unitree::robot::go2w::publisher::Sensor<Odom> odometry_;
    unitree::robot::go2w::SafeSubscriber<Cmd> commands_;
    std::unique_ptr<unitree::robot::go2w::subscription::Sensor<std_msgs::msg::dds_::String_>> readiness_;
    std::shared_ptr<unitree::common::UnitreeJoystick> joystick_;
    std::mutex mutex_;
    Cmd command_{}; double command_at_=0;
    Cloud cloud_;
    std::array<int,16> joints_{},actuators_{};
    std::array<mjtNum,7> home_{};
    std::array<float,4> wheel_targets_{};
    int body_=0,iteration_=0;
    double stand_started_=-1;
    double auto_started_=-1;
    std::atomic<bool> command_seen_{false};
    std::atomic<bool> controller_ready_{false};
    uint64_t command_count_=0,state_count_=0,cloud_count_=0,odom_count_=0;
    float wheel_position_gain_=0,max_wheel_target_=0,min_height_=1000;
};
