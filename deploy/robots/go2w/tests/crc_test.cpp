#include <cmath>
#include <iostream>
#include <unitree/dds_wrapper/robots/go2w/go2w_crc.h>

using unitree::robot::go2w::protocol_crc;
void require(bool ok) { if(!ok) throw std::runtime_error("Go2W CRC regression"); }
int main() {
    unitree_go::msg::dds_::LowCmd_ cmd{};
    cmd.head()={0xFE,0xEF}; cmd.level_flag()=255; cmd.frame_reserve()=12;
    cmd.sn()={1,2}; cmd.version()={3,4}; cmd.bandwidth()=500;
    for (int i=0;i<20;++i) {
        auto& m=cmd.motor_cmd()[i]; m.mode()=1;
        m.q()=0.01f*i; m.dq()=2.0f*i; m.tau()=-0.5f*i; m.kp()=20+i; m.kd()=0.5f;
        m.reserve()={uint32_t(i+100),200,300};
    }
    cmd.bms_cmd().off()=1; cmd.bms_cmd().reserve()={5,6,7};
    for (int i=0;i<40;++i) cmd.wireless_remote()[i]=i;
    for (int i=0;i<12;++i) cmd.led()[i]=10+i;
    cmd.fan()={1,2}; cmd.gpio()=3; cmd.reserve()=42;
    auto other=cmd;
    auto* bytes=reinterpret_cast<unsigned char*>(&other.motor_cmd()[0]);
    require(reinterpret_cast<unsigned char*>(&other.motor_cmd()[0].q())-bytes==4);
    std::memset(bytes+1,0xA5,3);
    require(cmd==other && protocol_crc(cmd)==protocol_crc(other));
    other.motor_cmd()[0].q()+=1;
    require(protocol_crc(cmd)!=protocol_crc(other));

    unitree_go::msg::dds_::LowState_ state{};
    state.head()={0xFE,0xEF}; state.level_flag()=255; state.frame_reserve()=12;
    state.sn()={1,2}; state.version()={3,4}; state.bandwidth()=500;
    auto& imu=state.imu_state();
    imu.quaternion()={1,.1f,.2f,.3f}; imu.gyroscope()={.4f,.5f,.6f};
    imu.accelerometer()={1,2,3}; imu.rpy()={.7f,.8f,.9f}; imu.temperature()=40;
    for (int i=0;i<20;++i) {
        auto& m=state.motor_state()[i]; m.mode()=1;
        m.q()=.01f*i; m.dq()=2.0f*i; m.ddq()=3.0f*i; m.tau_est()=-.5f*i;
        m.q_raw()=4.0f*i; m.dq_raw()=5.0f*i; m.ddq_raw()=6.0f*i;
        m.temperature()=20+i; m.lost()=100+i; m.reserve()={uint32_t(i),44};
    }
    auto& bms=state.bms_state();
    bms.version_high()=1; bms.version_low()=2; bms.status()=3; bms.soc()=4; bms.current()=-123; bms.cycle()=12;
    bms.bq_ntc()={8,9}; bms.mcu_ntc()={10,11};
    for (int i=0;i<15;++i) bms.cell_vol()[i]=12+i;
    state.foot_force()={10,20,30,40}; state.foot_force_est()={11,22,33,44}; state.tick()=12345;
    for (int i=0;i<40;++i) state.wireless_remote()[i]=i;
    state.bit_flag()=1; state.adc_reel()=1.25f; state.temperature_ntc1()=20; state.temperature_ntc2()=21;
    state.power_v()=24.5f; state.power_a()=1.5f; state.fan_frequency()={1,2,3,4}; state.reserve()=99;
    auto second=state;
    bytes=reinterpret_cast<unsigned char*>(&second.motor_state()[0]);
    require(reinterpret_cast<unsigned char*>(&second.motor_state()[0].q())-bytes==4);
    std::memset(bytes+1,0x5A,3);
    require(state==second && protocol_crc(state)==protocol_crc(second));
    second.tick()+=1; require(protocol_crc(state)!=protocol_crc(second));
    std::cout << protocol_crc(cmd) << ' ' << protocol_crc(state) << '\n';
}
