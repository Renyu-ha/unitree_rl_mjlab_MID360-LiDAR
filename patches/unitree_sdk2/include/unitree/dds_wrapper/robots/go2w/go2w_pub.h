// Go2W SDK extension; source of truth: patches/unitree_sdk2/include.
#pragma once
#include "go2w_sub.h"
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/dds_wrapper/common/crc.h>
#include <functional>
namespace unitree::robot::go2w::publisher {
class LowCmd {
public:
    using Message = unitree_go::msg::dds_::LowCmd_;
    Message msg_{};
    inline static bool enabled = false;
    inline static double heartbeat_timeout_s = 0.05;
    inline static std::atomic<bool> watchdog_fault{false};
    std::function<bool()> healthy;
    explicit LowCmd(subscription::LowState::SharedPtr state) : state_(std::move(state)) {
        msg_.head() = {0xFE, 0xEF};
        msg_.level_flag() = 0xFF;
        for (int i = 0; i < 20; ++i) msg_.motor_cmd()[i].mode() = i < 16 ? 1 : 0;
        if (enabled) {
            pub_ = std::make_unique<unitree::robot::ChannelPublisher<Message>>("rt/lowcmd");
            pub_->InitChannel();
            running_ = true;
            thread_ = std::thread([this] { publish_loop(); });
        }
    }
    ~LowCmd() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
    }
    void unlockAndPublish() {
        std::lock_guard<std::mutex> lock(mutex_);
        outgoing_ = msg_;
        heartbeat_ = detail::now();
    }
    void passive() {
        auto state = state_->snapshot();
        for (int i = 0; i < 20; ++i) {
            auto& m = msg_.motor_cmd()[i];
            m.q() = std::isfinite(state.motor_state()[i].q()) ? state.motor_state()[i].q() : 0;
            m.dq() = m.tau() = m.kp() = 0;
            m.kd() = i < 12 ? 3.0f : (i < 16 ? 1.0f : 0.0f);
        }
    }
private:
    void publish_loop() {
        auto tick = std::chrono::steady_clock::now();
        while (running_) {
            Message outgoing;
            bool expired;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                outgoing = outgoing_;
                expired = heartbeat_ == 0 || detail::now() - heartbeat_ > heartbeat_timeout_s;
            }
            // Do not publish before the first FSM command has been prepared.
            if (outgoing.head()[0] == 0xFE) {
                if (expired || state_->isTimeout() || (healthy && !healthy())) {
                    watchdog_fault = true;
                    auto state = state_->snapshot();
                    for (int i = 0; i < 20; ++i) {
                        auto& m = outgoing.motor_cmd()[i];
                        m.kp() = m.dq() = m.tau() = 0;
                        m.kd() = i < 12 ? 3.0f : (i < 16 ? 1.0f : 0.0f);
                        m.q() = std::isfinite(state.motor_state()[i].q()) ? state.motor_state()[i].q() : 0;
                    }
                }
                outgoing.crc() = protocol_crc(outgoing);
                pub_->Write(outgoing);
            }
            tick += std::chrono::milliseconds(2);
            std::this_thread::sleep_until(tick);
            if (tick < std::chrono::steady_clock::now() - std::chrono::milliseconds(10))
                tick = std::chrono::steady_clock::now();
        }
    }
    subscription::LowState::SharedPtr state_;
    std::unique_ptr<unitree::robot::ChannelPublisher<Message>> pub_;
    std::mutex mutex_;
    Message outgoing_{};
    double heartbeat_ = 0;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

}
