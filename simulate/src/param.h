#pragma once

#include <iostream>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace param
{

inline struct SimulationConfig
{
    std::string robot;
    std::filesystem::path robot_scene;

    int domain_id;
    std::string interface;

    int use_joystick;
    std::string joystick_type;
    std::string joystick_device;
    int joystick_bits;

    int print_scene_information;

    int enable_elastic_band;
    int band_attached_link = 0;
    bool headless = false, automatic = false, freeze_physics = false;
    double duration = 30.0, drop_cloud_after = 1e30, drop_odom_after = 1e30;
    double drop_state_after = 1e30, corrupt_state_after = 1e30;
    double estop_after = 1e30;
    double support_seconds = 3.2;
    float command_vx = 0, command_vy = 0, command_wz = 0;
    std::filesystem::path report;

    void load_from_yaml(const std::string &filename)
    {
        auto cfg = YAML::LoadFile(filename);
        try
        {
            robot = cfg["robot"].as<std::string>();
            robot_scene = cfg["robot_scene"].as<std::string>();
            domain_id = cfg["domain_id"].as<int>();
            interface = cfg["interface"].as<std::string>();
            use_joystick = cfg["use_joystick"].as<int>();
            joystick_type = cfg["joystick_type"].as<std::string>();
            joystick_device = cfg["joystick_device"].as<std::string>();
            joystick_bits = cfg["joystick_bits"].as<int>();
            print_scene_information = cfg["print_scene_information"].as<int>();
            enable_elastic_band = cfg["enable_elastic_band"].as<int>();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }
} config;

/* ---------- Command Line Parameters ---------- */
namespace po = boost::program_options;

//※ This function must be called at the beginning of main() function
inline po::variables_map helper(int argc, char** argv)
{
    po::options_description desc("Unitree Mujoco");
    desc.add_options()
        ("help,h", "Show help message")
        ("domain_id,i", po::value<int>(&config.domain_id), "DDS domain ID; -i 0")
        ("network,n", po::value<std::string>(&config.interface), "DDS network interface; -n eth0")
        ("robot,r", po::value<std::string>(&config.robot), "Robot type; -r go2")
        ("scene,s", po::value<std::filesystem::path>(&config.robot_scene), "Robot scene file; -s scene_terrain.xml")
        ("config", po::value<std::string>(), "Simulation YAML file")
        ("headless", po::bool_switch(&config.headless), "Run without a display (Go2W)")
        ("auto", po::bool_switch(&config.automatic), "Synthetic stand/run buttons (isolated Go2W simulation only)")
        ("duration", po::value<double>(&config.duration), "Headless simulation seconds")
        ("vx", po::value<float>(&config.command_vx), "Synthetic forward command")
        ("vy", po::value<float>(&config.command_vy), "Synthetic lateral command")
        ("wz", po::value<float>(&config.command_wz), "Synthetic yaw command")
        ("drop-cloud-after", po::value<double>(&config.drop_cloud_after), "Stop synthetic point clouds after simulation seconds")
        ("drop-odom-after", po::value<double>(&config.drop_odom_after), "Stop synthetic odometry")
        ("drop-state-after", po::value<double>(&config.drop_state_after), "Stop LowState")
        ("corrupt-state-after", po::value<double>(&config.corrupt_state_after), "Inject bad LowState CRC")
        ("estop-after", po::value<double>(&config.estop_after), "Hold the raw L2+B stop combination")
        ("freeze-physics", po::bool_switch(&config.freeze_physics), "Protocol test only: keep the robot at its home pose")
        ("report", po::value<std::filesystem::path>(&config.report), "Write Go2W simulation metrics as JSON")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    return vm;
}

inline void load(int argc, char** argv, const std::filesystem::path& default_config)
{
    std::filesystem::path filename = default_config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { helper(argc, argv); return; }
        if (arg == "--config" && i+1 < argc) filename = argv[++i];
        else if (arg.rfind("--config=", 0) == 0) filename = arg.substr(9);
    }
    filename = std::filesystem::absolute(filename);
    config.load_from_yaml(filename.string());
    helper(argc, argv);
    if (config.robot_scene.is_relative()) {
        auto local = filename.parent_path()/config.robot_scene;
        auto legacy = filename.parent_path().parent_path()/config.robot_scene;
        config.robot_scene = std::filesystem::exists(local) ? local : legacy;
    }
}

}
