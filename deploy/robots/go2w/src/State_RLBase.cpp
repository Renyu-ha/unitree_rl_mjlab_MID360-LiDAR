#include "FSM/State_RLBase.h"
#include "Go2WLidar.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"


class LidarHeightScanUpdater : public go2w::LidarHeightScanUpdater {
public:
    using go2w::LidarHeightScanUpdater::LidarHeightScanUpdater;
};
struct State_RLBase::RobotExtension {
    std::vector<int> ids;
    YAML::Node safety;
    std::vector<float> min_q, max_q;
    std::atomic<bool> fault{false};
    std::mutex result_mutex;
    std::vector<float> result;
    double result_at = 0, entered_at = 0;
};

namespace isaaclab {
REGISTER_OBSERVATION(wheel_vel) {
    auto& v = env->robot->data.joint_vel;
    return std::vector<float>(v.data()+12, v.data()+16);
}
}

namespace {
bool equal_contract(const YAML::Node& a, const YAML::Node& b) {
    if (a.Type() != b.Type() || a.size() != b.size()) return false;
    if (a.IsScalar()) {
        if (a.as<std::string>() == b.as<std::string>()) return true;
        try { return std::abs(a.as<double>()-b.as<double>()) < 1e-8; }
        catch (...) { return false; }
    }
    if (a.IsSequence()) {
        for (size_t i = 0; i < a.size(); ++i) if (!equal_contract(a[i], b[i])) return false;
    } else if (a.IsMap()) {
        auto ai = a.begin(), bi = b.begin();
        for (; ai != a.end(); ++ai, ++bi)
            if (ai->first.as<std::string>() != bi->first.as<std::string>() || !equal_contract(ai->second, bi->second)) return false;
    }
    return true;
}
class Go2WArticulation : public unitree::BaseArticulation<LowState_t::SharedPtr> {
public:
    explicit Go2WArticulation(LowState_t::SharedPtr state) : BaseArticulation(state) {}
    void update() override {
        // Read the callback snapshot instead of the FSM's public message.
        auto state = lowstate->snapshot();
        data.root_quat_w = Eigen::Quaternionf(state.imu_state().quaternion()[0],
            state.imu_state().quaternion()[1], state.imu_state().quaternion()[2], state.imu_state().quaternion()[3]);
        go2w::require(data.root_quat_w.coeffs().allFinite() && data.root_quat_w.norm() > 0.9f && data.root_quat_w.norm() < 1.1f,
                      "Invalid IMU quaternion");
        data.root_quat_w.normalize();
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        for (int i = 0; i < 3; ++i) data.root_ang_vel_b[i] = state.imu_state().gyroscope()[i];
        for (int i = 0; i < 16; ++i) {
            int id = int(data.joint_ids_map[i]);
            data.joint_pos[i] = state.motor_state()[id].q();
            data.joint_vel[i] = state.motor_state()[id].dq();
        }
        unitree::common::REMOTE_DATA_RX remote{};
        std::memcpy(remote.buff, state.wireless_remote().data(), sizeof(remote));
        joystick_.extract(remote);
        data.joystick = &joystick_;
    }
private:
    unitree::common::UnitreeJoystick joystick_;
};
}

State_RLBase::State_RLBase(int mode, std::string name) : FSMState(mode, name), extension_(std::make_shared<RobotExtension>()) {
    auto cfg = param::config["FSM"][name];
    auto policy = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto deploy = YAML::LoadFile((policy / "params/deploy.yaml").string());
    go2w::require(deploy["go2w_deploy_schema"].as<int>(0) == 1, "Re-export a Go2W deployment bundle first");
    auto embedded = YAML::Load(deploy["onnx_metadata"]["go2w_deploy_contract"].as<std::string>());
    for (auto field : embedded)
        go2w::require(equal_contract(deploy[field.first.as<std::string>()], field.second),
                      "Deployment YAML differs from exported contract: " + field.first.as<std::string>());
    extension_->ids = deploy["joint_ids_map"].as<std::vector<int>>();
    go2w::validate_map(extension_->ids);
    auto names = deploy["action_joint_names"].as<std::vector<std::string>>();
    go2w::require(names == std::vector<std::string>(go2w::joint_names.begin(), go2w::joint_names.end()), "Unexpected Go2W action order");
    extension_->safety = param::config["safety"];
    extension_->min_q = extension_->safety["leg_position_min"].as<std::vector<float>>();
    extension_->max_q = extension_->safety["leg_position_max"].as<std::vector<float>>();
    go2w::require(extension_->min_q.size() == 12 && extension_->max_q.size() == 12, "Set all 12 leg target limits");
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(deploy, std::make_shared<Go2WArticulation>(lowstate));
    auto runner = std::make_unique<isaaclab::OrtRunner>((policy / "exported/policy.onnx").string());
    auto expected = deploy["onnx_metadata"].as<std::map<std::string,std::string>>();
    if (expected["policy_kind"].rfind("smoke_",0)==0 && LowCmd_t::enabled)
        go2w::require(param::config["simulation"].as<bool>(false),"The smoke policy is for --sim only; supply a trained actor for hardware");
    if (LowCmd_t::enabled && !param::config["simulation"].as<bool>(false))
        go2w::require(extension_->safety["hardware_calibrated"].as<bool>(false),"Complete the hardware calibration checklist and set safety.hardware_calibrated=true before actuating");
    go2w::require(expected.count("go2w_deploy_contract") != 0, "Deployment metadata is missing");
    runner->validate_deployment(deploy["observation_dim"].as<size_t>(), 16, expected);
    env->alg = std::move(runner);
    if (deploy["perception_required"].as<bool>())
        lidar_updater_ = std::make_unique<LidarHeightScanUpdater>(cfg["lidar"], deploy);
    registered_checks.insert(registered_checks.begin(), {
        [this] { return extension_->fault.load() || LowCmd_t::watchdog_fault.load(); }, FSMStringMap.right.at("Passive")});
}

void State_RLBase::enter() {
    exit();
    extension_->fault = false;
    LowCmd_t::watchdog_fault = false;
    extension_->entered_at = go2w::now();
    { std::lock_guard<std::mutex> lock(extension_->result_mutex); extension_->result.clear(); extension_->result_at = 0; }
    if (lidar_updater_) lidar_updater_->reset();
    policy_thread_running = true;
    policy_thread = std::thread(&State_RLBase::policy_loop, this);
}

void State_RLBase::policy_loop() {
    try {
        bool reset_needed = true;
        auto tick = std::chrono::steady_clock::now();
        while (policy_thread_running) {
            const double started = go2w::now();
            go2w::require(!lowstate->isTimeout(), "LowState timeout");
            env->robot->update();
            auto gravity = env->robot->data.projected_gravity_b;
            go2w::require(-gravity.z() > std::cos(extension_->safety["max_tilt_rad"].as<float>()), "Body tilt limit");
            if (lidar_updater_) {
                auto scan = lidar_updater_->update(env->robot->data.root_quat_w, env->step_dt);
                std::lock_guard<std::mutex> lock(env->robot->data.height_scan_mutex);
                env->robot->data.height_scan = std::move(scan);
                env->robot->data.height_scan_valid = true;
            }
            // Backfill all history slots from the first REAL sensor frame,
            // as mjlab's CircularBuffer does after reset.
            if (reset_needed) { env->reset(false); reset_needed = false; }
            env->step(false);
            if (env->episode_length==1 || env->episode_length%250==0)
                spdlog::info("Go2W policy steps={}",env->episode_length);
            auto actions = env->action_manager->processed_actions();
            go2w::require(actions.size() == 16, "Expected 16 Go2W actions");
            go2w::finite(actions, "Processed actions");
            for (int i = 0; i < 12; ++i)
                go2w::require(actions[i] >= extension_->min_q[i] && actions[i] <= extension_->max_q[i], "Leg target outside configured limits");
            for (int i = 12; i < 16; ++i)
                go2w::require(std::abs(actions[i]) <= extension_->safety["max_wheel_speed_rad_s"].as<float>(), "Wheel target outside configured limits");
            go2w::require(go2w::now() - started < extension_->safety["policy_timeout_ms"].as<float>() / 1000, "Inference deadline exceeded");
            {
                std::lock_guard<std::mutex> lock(extension_->result_mutex);
                extension_->result = std::move(actions); extension_->result_at = started;
            }
            tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(env->step_dt));
            std::this_thread::sleep_until(tick);
            if (tick < std::chrono::steady_clock::now() - std::chrono::milliseconds(20))
                tick = std::chrono::steady_clock::now();
        }
    } catch (const std::exception& error) {
        spdlog::error("Go2W policy stopped: {}", error.what());
        extension_->fault = true;
    }
}

void State_RLBase::run() {
    std::vector<float> actions;
    double age;
    { std::lock_guard<std::mutex> lock(extension_->result_mutex); actions = extension_->result; age = go2w::now()-extension_->result_at; }
    if (extension_->fault || LowCmd_t::watchdog_fault || lowstate->isTimeout()) { lowcmd->passive(); return; }
    if (actions.empty()) {
        // Keep the previous FixStand target only for a bounded warm-up window.
        if (go2w::now()-extension_->entered_at > std::max(0.1,extension_->safety["policy_timeout_ms"].as<double>()/1000)) {
            extension_->fault = true; lowcmd->passive();
        }
        return;
    }
    if (age > extension_->safety["policy_timeout_ms"].as<float>()/1000) {
        extension_->fault = true; lowcmd->passive(); return;
    }
    const auto targets = go2w::motor_targets(actions, extension_->ids, env->robot->data.joint_stiffness, env->robot->data.joint_damping);
    for (int i = 0; i < 16; ++i) {
        auto& m = lowcmd->msg_.motor_cmd()[i];
        m.q() = targets[i].q; m.dq() = targets[i].dq;
        m.kp() = targets[i].kp; m.kd() = targets[i].kd; m.tau() = 0;
    }
}

void State_RLBase::exit() {
    policy_thread_running = false;
    if (policy_thread.joinable()) policy_thread.join();
}

State_RLBase::~State_RLBase() { exit(); }
