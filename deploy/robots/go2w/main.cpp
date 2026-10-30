#include <csignal>
#include <unistd.h>
#include <unitree/dds_wrapper/robots/go2w/go2w_sensor.h>
#include <unitree/idl/ros2/String_.hpp>
#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "go2w_core.h"
#include "Go2WFSM.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd;
std::shared_ptr<LowState_t> FSMState::lowstate;
std::shared_ptr<Keyboard> FSMState::keyboard;
namespace { volatile std::sig_atomic_t stopped = 0; void stop(int) { stopped = 1; } }

int main(int argc, char** argv) {
    bool fsm_started=false;
    try {
        namespace po = boost::program_options;
        po::options_description options("Go2W controller (adapted from G1)");
        options.add_options()
            ("help,h", "Show options")
            ("config", po::value<std::string>(), "Hardware/FSM YAML")
            ("policy", po::value<std::string>(), "Exact policy bundle with exported/ and params/")
            ("network,n", po::value<std::string>()->default_value("lo"), "DDS network interface")
            ("domain", po::value<int>()->default_value(1), "DDS domain (real robot: 0)")
            ("sim", "Use synthetic base-frame point clouds on an isolated loopback domain")
            ("actuate", "Publish motor commands; default is receive/inference only");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, options), vm);
        po::notify(vm);
        if (vm.count("help")) { std::cout << options << '\n'; return 0; }
        param::bin_path = param::get_bin_path();
        param::proj_dir = param::bin_path.parent_path().parent_path();
        std::filesystem::path config = vm.count("config") ? std::filesystem::path(vm["config"].as<std::string>()) : param::proj_dir / "config/config.yaml";
        param::config = YAML::LoadFile(config.string());
        // Relative paths in a custom config resolve against its parent project.
        param::proj_dir = std::filesystem::absolute(config).parent_path().parent_path();
        if (vm.count("policy")) param::config["FSM"]["Velocity"]["policy_dir"] = std::filesystem::absolute(vm["policy"].as<std::string>()).string();
        auto stand = param::config["FSM"]["FixStand"];
        for (auto field : {"kp", "kd"}) {
            auto values = stand[field].as<std::vector<float>>();
            go2w::require(values.size() == 16, "FixStand gains must contain 16 SDK-order values");
            go2w::finite(values, "FixStand gains");
            for (float value : values) go2w::require(value >= 0, "Gains must be nonnegative");
            if (std::string(field) == "kp") for (int i = 12; i < 16; ++i)
                go2w::require(values[i] == 0, "Wheel position gains must be zero");
        }
        go2w::require(param::config["FSM"]["Passive"]["mode"].size() == 16 &&
                      param::config["FSM"]["Passive"]["kd"].size() == 16, "Passive arrays must contain 16 values");
        auto passive_kd = param::config["FSM"]["Passive"]["kd"].as<std::vector<float>>();
        go2w::finite(passive_kd, "Passive damping");
        for (float value : passive_kd) go2w::require(value >= 0, "Passive damping must be nonnegative");
        auto ts = stand["ts"].as<std::vector<float>>();
        auto qs = stand["qs"].as<std::vector<std::vector<float>>>();
        go2w::require(ts.size() >= 2 && ts.size() == qs.size() && ts.front() == 0, "Invalid stand interpolation");
        go2w::finite(ts, "Stand interpolation times");
        for (size_t i = 1; i < ts.size(); ++i) {
            go2w::require(ts[i] > ts[i-1] && qs[i].size() == 16, "Stand poses need 16 values and increasing times");
            go2w::finite(qs[i], "Stand pose");
        }
        const auto network = vm["network"].as<std::string>();
        const int domain = vm["domain"].as<int>();
        param::config["FSM"]["Velocity"]["lidar"]["domain_id"]=domain;
        go2w::require(!network.empty() && domain >= 0, "Specify a network interface and nonnegative DDS domain");
        if (vm.count("sim")) {
            go2w::require(network == "lo" && domain > 0, "--sim requires loopback and a nonzero DDS domain");
            auto lidar = param::config["FSM"]["Velocity"]["lidar"];
            lidar["topic"] = "/go2w/sim/cloud";
            lidar["frame_id"] = "base_link";
            lidar["extrinsics"]["translation_m"] = std::vector<float>{0,0,0};
            lidar["extrinsics"]["rpy_deg"] = std::vector<float>{0,0,0};
        }
        LowCmd_t::enabled = vm.count("actuate") != 0;
        LowCmd_t::heartbeat_timeout_s=param::config["safety"]["command_timeout_ms"].as<double>(50)/1000;
        go2w::require(std::isfinite(LowCmd_t::heartbeat_timeout_s) && LowCmd_t::heartbeat_timeout_s>0,"Set a positive command heartbeat timeout");
        param::config["simulation"]=vm.count("sim")!=0;
        unitree::robot::ChannelFactory::Instance()->Init(domain, network);
        if (LowCmd_t::enabled) {
            std::atomic<bool> occupied{false};
            unitree::robot::go2w::SafeSubscriber<unitree_go::msg::dds_::LowCmd_> check("rt/lowcmd",domain);
            check.InitChannel([&](const void*) { occupied = true; }, 1);
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            check.CloseChannel();
            go2w::require(!occupied, "rt/lowcmd already has a writer; stop the other controller first");
        }
        FSMState::lowstate = std::make_shared<LowState_t>(param::config["safety"]["lowstate_timeout_ms"].as<float>()/1000,domain);
        spdlog::info("Waiting for Go2W LowState on interface {}, domain {}", network, domain);
        FSMState::lowstate->wait_for_connection();
        FSMState::lowcmd = std::make_unique<LowCmd_t>(FSMState::lowstate);
        const float max_tilt = param::config["safety"]["max_tilt_rad"].as<float>();
        FSMState::lowcmd->healthy = [max_tilt] {
            auto state = FSMState::lowstate->snapshot();
            unitree::common::REMOTE_DATA_RX remote{};
            std::memcpy(remote.buff, state.wireless_remote().data(), sizeof(remote));
            if (remote.RF_RX.btn.components.L2 && remote.RF_RX.btn.components.B) return false;
            const auto& q = state.imu_state().quaternion();
            float norm = q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
            if (!std::isfinite(norm) || norm < 0.81f || norm > 1.21f) return false;
            if (1 - 2*(q[1]*q[1]+q[2]*q[2])/norm < std::cos(max_tilt)) return false;
            for (int i = 0; i < 16; ++i)
                if (!std::isfinite(state.motor_state()[i].q()) || !std::isfinite(state.motor_state()[i].dq())) return false;
            return true;
        };
        go2w::register_stand();
        auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
        go2w::protect_fsm(*fsm);
        fsm->start();
        fsm_started=true;
        spdlog::info("Go2W {}. L2+Up: FixStand; R2+A: Velocity; L2+B: Passive.",
                     LowCmd_t::enabled ? "motor output enabled" : "dry run: no LowCmd is published");
        std::signal(SIGINT, stop); std::signal(SIGTERM, stop);
        using Ready=std_msgs::msg::dds_::String_;
        std::unique_ptr<unitree::robot::go2w::publisher::Sensor<Ready>> simulation_status;
        Ready ready; ready.data()="ready";
        if (vm.count("sim")) simulation_status=std::make_unique<unitree::robot::go2w::publisher::Sensor<Ready>>(domain,"rt/go2w/sim/ready");
        while (!stopped) {
            if (simulation_status) simulation_status->Write(ready);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        fsm->stop();
        FSMState::lowcmd->passive();
        FSMState::lowcmd->unlockAndPublish();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        fsm.reset(); FSMState::lowcmd.reset(); FSMState::lowstate.reset();
    } catch (const std::exception& error) {
        spdlog::critical("Go2W startup/runtime error: {}", error.what());
        if (FSMState::lowcmd && fsm_started) {
            FSMState::lowcmd->passive(); FSMState::lowcmd->unlockAndPublish();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        FSMState::lowcmd.reset(); FSMState::lowstate.reset();
        return 1;
    }
}
