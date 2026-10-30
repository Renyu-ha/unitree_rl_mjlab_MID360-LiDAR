#pragma once
#include <csignal>
#include <iostream>
#include <thread>
#include <unistd.h>
#include "go2w_bridge.h"

namespace go2w_sim {
inline volatile std::sig_atomic_t stopped=0;
inline void stop(int) { stopped=1; }
inline std::filesystem::path executable_directory() {
    std::array<char,4096> path{};
    auto count=readlink("/proc/self/exe",path.data(),path.size()-1);
    go2w::require(count>0,"Cannot resolve simulator executable");
    return std::filesystem::path(std::string(path.data(),size_t(count))).parent_path();
}
inline int run() {
    go2w::require(param::config.robot=="go2w","Headless mode currently supports Go2W");
    go2w::require(param::config.domain_id>0 && param::config.interface=="lo","Go2W simulation requires loopback and nonzero DDS domain");
    go2w::require(std::isfinite(param::config.duration) && param::config.duration>0,"Set a positive simulation duration");
    unitree::robot::ChannelFactory::Instance()->Init(param::config.domain_id,param::config.interface);
    char error[2048]{};
    std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_loadXML(param::config.robot_scene.c_str(),nullptr,error,sizeof(error)),mj_deleteModel);
    go2w::require(bool(model),std::string("Cannot load Go2W scene: ")+error);
    std::unique_ptr<mjData,decltype(&mj_deleteData)> data(mj_makeData(model.get()),mj_deleteData);
    go2w::require(mjVERSION_HEADER==mj_version(),"MuJoCo headers/library mismatch");
    Go2WBridge bridge(model.get(),data.get());
    std::signal(SIGINT,stop); std::signal(SIGTERM,stop);
    auto tick=std::chrono::steady_clock::now();
    std::cout << "unitree_mujoco Go2W headless; MuJoCo " << mj_versionString() << std::endl;
    const double started=go2w::now();
    while (!stopped && bridge.elapsed(data->time)<param::config.duration) {
        go2w::require(go2w::now()-started<std::max(60.0,param::config.duration*30),"Headless simulation deadline exceeded; check controller connection or host speed");
        bridge.before_step(model.get(),data.get());
        if (param::config.freeze_physics) data->time+=model->opt.timestep;
        else mj_step(model.get(),data.get());
        bridge.after_step(model.get(),data.get());
        tick+=std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(model->opt.timestep));
        std::this_thread::sleep_until(tick);
        if (tick<std::chrono::steady_clock::now()-std::chrono::milliseconds(20)) tick=std::chrono::steady_clock::now();
    }
    bridge.write_report(param::config.report,data->time);
    std::cout << "Go2W simulation completed: " << data->time << " seconds" << std::endl;
    return 0;
}
inline int main(int argc,char** argv) {
    try {
        param::load(argc,argv,executable_directory().parent_path()/"config.yaml");
        return run();
    } catch (const std::exception& error) {
        std::cerr << "unitree_mujoco: " << error.what() << std::endl;
        return 1;
    }
}
}
