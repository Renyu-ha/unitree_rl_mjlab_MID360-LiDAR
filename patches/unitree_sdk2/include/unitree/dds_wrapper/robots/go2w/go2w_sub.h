// Go2W SDK extension; source of truth: patches/unitree_sdk2/include.
#pragma once
#include <cmath>
#include "defines.h"
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include <unitree/dds_wrapper/common/crc.h>
#include "go2w_crc.h"
#include "go2w_channel.h"
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
namespace unitree::robot::go2w::subscription {
class LowState {
public:
    using SharedPtr = std::shared_ptr<LowState>;
    using Message = unitree_go::msg::dds_::LowState_;
    Message msg_{};
    std::mutex mutex_;
    unitree::common::UnitreeJoystick joystick;
    explicit LowState(double timeout = 0.1,int domain = 0) : timeout_(timeout) {
        sub_ = std::make_unique<SafeSubscriber<Message>>("rt/lowstate",domain);
        sub_->InitChannel([this](const void* data) {
            const auto& received = *static_cast<const Message*>(data);
            if (received.crc() != protocol_crc(received)) return;
            std::lock_guard<std::mutex> lock(mutex_);
            received_ = received;
            received_at_ = detail::now();
        }, 1);
    }
    ~LowState() { sub_->CloseChannel(); }
    void update() {
        std::lock_guard<std::mutex> lock(mutex_);
        msg_ = received_;
        unitree::common::REMOTE_DATA_RX remote{};
        std::memcpy(remote.buff, msg_.wireless_remote().data(), sizeof(remote));
        joystick.extract(remote);
    }
    Message snapshot() {
        std::lock_guard<std::mutex> lock(mutex_);
        return received_;
    }
    bool isTimeout() {
        std::lock_guard<std::mutex> lock(mutex_);
        return received_at_ == 0 || detail::now() - received_at_ > timeout_;
    }
    void wait_for_connection() {
        double start = detail::now();
        while (isTimeout()) {
            detail::require(detail::now() - start < 15, "Timed out waiting for rt/lowstate");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        update();
    }
private:
    std::unique_ptr<SafeSubscriber<Message>> sub_;
    Message received_{};
    double timeout_, received_at_ = 0;
};


}
