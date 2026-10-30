// Go2W SDK extension; source of truth: patches/unitree_sdk2/include.
#pragma once
#include <dds/dds.hpp>
#include <functional>

namespace unitree::robot::go2w {
namespace subscription {
// ROS2 sensor publishers commonly offer BestEffort. Request it explicitly,
// which also matches Reliable publishers, instead of relying on SDK defaults.
template<typename Message>
class Sensor : private dds::sub::NoOpDataReaderListener<Message> {
public:
    using Callback=std::function<void(const void*)>;
    Sensor(int domain,const std::string& topic,Callback callback)
        : callback_(std::move(callback)),participant_(domain),subscriber_(participant_),topic_(participant_,topic),
          reader_(subscriber_,topic_,subscriber_.default_datareader_qos()
              << dds::core::policy::Reliability::BestEffort()
              << dds::core::policy::Durability::Volatile()
              << dds::core::policy::History::KeepLast(1)) {
        reader_.listener(this,dds::core::status::StatusMask::data_available());
    }
    ~Sensor() { CloseChannel(); }
    void CloseChannel() {
        if (closed_) return;
        // close() waits for in-flight callbacks while their listener is alive.
        // Replacing the listener with nullptr first races a callback already
        // entering Cyclone DDS 0.10's listener dispatch.
        reader_.close();
        closed_=true;
    }
private:
    void on_data_available(dds::sub::DataReader<Message>& reader) override {
        auto samples=reader.take();
        for (const auto& sample:samples) if (sample.info().valid()) callback_(&sample.data());
    }
    Callback callback_;
    dds::domain::DomainParticipant participant_;
    dds::sub::Subscriber subscriber_;
    dds::topic::Topic<Message> topic_;
    dds::sub::DataReader<Message> reader_;
    bool closed_=false;
};
}
namespace publisher {
template<typename Message>
class Sensor {
public:
    Sensor(int domain,const std::string& topic)
        : participant_(domain),publisher_(participant_),topic_(participant_,topic),
          writer_(publisher_,topic_,publisher_.default_datawriter_qos()
              << dds::core::policy::Reliability::BestEffort()
              << dds::core::policy::Durability::Volatile()
              << dds::core::policy::History::KeepLast(1)) {}
    void Write(const Message& value) { writer_.write(value); }
    bool Matched() { return writer_.publication_matched_status().current_count()>0; }
private:
    dds::domain::DomainParticipant participant_;
    dds::pub::Publisher publisher_;
    dds::topic::Topic<Message> topic_;
    dds::pub::DataWriter<Message> writer_;
};
}
}
