// Go2W SDK extension; source of truth: patches/unitree_sdk2/include.
#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/dds_wrapper/common/crc.h>

namespace unitree::robot::go2w {
// Explicit little-endian protocol layout, including zero alignment bytes.
// Matches unitree_sdk2_python/unitree_sdk2py/utils/crc.py (Go messages).
// DDS does not transmit the C++ object's padding; never checksum that memory.
class CrcPacket {
public:
    template<class T> void add(T value) {
        static_assert(std::is_arithmetic_v<T> && sizeof(T)<=4);
        uint32_t bits;
        if constexpr (std::is_floating_point_v<T>) std::memcpy(&bits,&value,4);
        else bits=static_cast<uint32_t>(value);
        for (size_t i=0;i<sizeof(T);++i) data_[size_++]=uint8_t(bits>>(8*i));
    }
    template<class T,size_t N> void add(const std::array<T,N>& values) { for (T value:values) add(value); }
    void pad(size_t count) { size_+=count; }
    uint32_t finish(size_t expected) {
        if (size_!=expected || size_%4) throw std::runtime_error("Unexpected Go2 CRC protocol layout");
        std::array<uint32_t,294> words{};
        for (size_t i=0;i<size_;++i) words[i/4]|=uint32_t(data_[i])<<(8*(i%4));
        return crc32_core(words.data(),uint32_t(size_/4));
    }
private:
    std::array<uint8_t,1180> data_{};
    size_t size_=0;
};

template<class Message> inline void crc_header(CrcPacket& packet,const Message& msg) {
    packet.add(msg.head()); packet.add(msg.level_flag()); packet.add(msg.frame_reserve());
    packet.add(msg.sn()); packet.add(msg.version()); packet.add(msg.bandwidth()); packet.pad(2);
}

inline uint32_t protocol_crc(const unitree_go::msg::dds_::LowCmd_& msg) {
    CrcPacket p; crc_header(p,msg);
    for (const auto& motor:msg.motor_cmd()) {
        p.add(motor.mode()); p.pad(3);
        p.add(motor.q()); p.add(motor.dq()); p.add(motor.tau()); p.add(motor.kp()); p.add(motor.kd()); p.add(motor.reserve());
    }
    p.add(msg.bms_cmd().off()); p.add(msg.bms_cmd().reserve());
    p.add(msg.wireless_remote()); p.add(msg.led()); p.add(msg.fan()); p.add(msg.gpio()); p.pad(1); p.add(msg.reserve());
    return p.finish(808);
}

inline uint32_t protocol_crc(const unitree_go::msg::dds_::LowState_& msg) {
    CrcPacket p; crc_header(p,msg);
    const auto& imu=msg.imu_state();
    p.add(imu.quaternion()); p.add(imu.gyroscope()); p.add(imu.accelerometer()); p.add(imu.rpy()); p.add(imu.temperature()); p.pad(3);
    for (const auto& motor:msg.motor_state()) {
        p.add(motor.mode()); p.pad(3);
        p.add(motor.q()); p.add(motor.dq()); p.add(motor.ddq()); p.add(motor.tau_est());
        p.add(motor.q_raw()); p.add(motor.dq_raw()); p.add(motor.ddq_raw());
        p.add(motor.temperature()); p.pad(3); p.add(motor.lost()); p.add(motor.reserve());
    }
    const auto& bms=msg.bms_state();
    p.add(bms.version_high()); p.add(bms.version_low()); p.add(bms.status()); p.add(bms.soc()); p.add(bms.current());
    p.add(bms.cycle()); p.add(bms.bq_ntc()); p.add(bms.mcu_ntc()); p.add(bms.cell_vol());
    p.add(msg.foot_force()); p.add(msg.foot_force_est()); p.add(msg.tick()); p.add(msg.wireless_remote()); p.add(msg.bit_flag()); p.pad(3);
    p.add(msg.adc_reel()); p.add(msg.temperature_ntc1()); p.add(msg.temperature_ntc2()); p.pad(2);
    p.add(msg.power_v()); p.add(msg.power_a()); p.add(msg.fan_frequency()); p.add(msg.reserve());
    return p.finish(1176);
}
}
