// Go2W SDK extension; source of truth: patches/unitree_sdk2/include.
#pragma once
#include "go2w_sensor.h"
#include <memory>

namespace unitree::robot::go2w {
// Cyclone DDS 0.10 requires explicit close while a strong reader reference
// remains. Merely resetting the SDK channel can run its destructor on a
// callback thread, hitting EntityDelegate::prevent_callbacks' assertion.
template<class Message> class SafeSubscriber {
public:
    SafeSubscriber(std::string topic,int domain) : topic_(std::move(topic)),domain_(domain) {}
    void InitChannel(const std::function<void(const void*)>& callback,int /*queue_length*/=1) {
        reader_=std::make_unique<subscription::Sensor<Message>>(domain_,topic_,callback);
    }
    void CloseChannel() {
        if (!reader_) return;
        reader_->CloseChannel();
        reader_.reset();
    }
    ~SafeSubscriber() { CloseChannel(); }
private:
    std::string topic_;
    int domain_;
    std::unique_ptr<subscription::Sensor<Message>> reader_;
};
}
