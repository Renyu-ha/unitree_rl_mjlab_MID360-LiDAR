#include <cmath>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/dds_wrapper/robots/go2w/go2w_channel.h>
#include <unitree/idl/go2/LowCmd_.hpp>

int main() {
    using Message=unitree_go::msg::dds_::LowCmd_;
    const int domain=200+getpid()%20;
    unitree::robot::ChannelFactory::Instance()->Init(domain,"lo");
    auto writer=unitree::robot::ChannelFactory::Instance()->CreateSendChannel<Message>("rt/go2w/test/shutdown");
    std::atomic<bool> running{true}; std::atomic<int> received{0};
    std::thread thread([&] {
        Message msg{};
        while (running) { writer->Write(msg,0); std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    });
    bool passed=true;
    for (int i=0;i<20;++i) {
        int before=received;
        unitree::robot::go2w::SafeSubscriber<Message> reader("rt/go2w/test/shutdown",domain);
        reader.InitChannel([&](const void*) { ++received; },1);
        auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
        while (received==before && std::chrono::steady_clock::now()<deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (received==before) passed=false;
        reader.CloseChannel();
    }
    running=false; thread.join(); writer.reset();
    if (!passed) { std::cerr << "DDS reader did not receive test traffic\n"; return 1; }
    std::cout << "PASS 20 DDS readers closed under active publication\n";
}
