#include "go2w_core.h"
#include <iostream>
#include <limits>
#include <iomanip>

static void check(bool value, const char* message) { go2w::require(value, message); }
static bool near(float a, float b) { return std::abs(a-b) < 1e-5f; }

int main(int argc, char**) {
    try {
        go2w::Grid grid;
        if (argc > 1) {
            // Numerical comparison against the actual training memory function.
            go2w::HeightMemory memory(grid, 0.8f);
            std::cout << std::setprecision(9);
            for (int t = 0; t < 60; ++t) {
                std::vector<float> scan(grid.size());
                for (int i = 0; i < grid.size(); ++i) scan[i] = 0.3f + 0.001f*i + 0.0001f*t;
                auto result = memory.update(scan, true, t < 30 ? 0.7f : 0.0f, 0.02f, {});
                for (float value : result) std::cout << value << ' ';
                std::cout << '\n';
            }
            return 0;
        }
        check(grid.size() == 187, "17x11 grid");
        check(grid.index(grid.x(0), grid.y(0)) == 0, "first sample centre");
        check(grid.index(grid.x(16), grid.y(10)) == 186, "last sample centre");
        check(grid.index(20, 0) == -1, "reject distant points");
        std::vector<int> ids(go2w::sdk_ids.begin(), go2w::sdk_ids.end());
        std::vector<float> kp(16, 20), kd(16, 2), actions(16);
        for (int i = 0; i < 16; ++i) actions[i] = float(i+1);
        auto targets = go2w::motor_targets(actions, ids, kp, kd);
        check(targets[3].q == 1 && targets[0].q == 4 && targets[9].q == 7 && targets[6].q == 10, "leg mapping");
        check(targets[13].dq == 13 && targets[12].dq == 14 && targets[15].dq == 15 && targets[14].dq == 16, "wheel mapping");
        for (int id = 12; id < 16; ++id) check(targets[id].kp == 0 && targets[id].q == 0, "wheels have no position gain");
        bool rejected = false;
        try { ids[0] = ids[1]; go2w::validate_map(ids); } catch (...) { rejected = true; }
        check(rejected, "duplicate IDs rejected");
        rejected = false;
        try { actions[0] = std::numeric_limits<float>::quiet_NaN(); go2w::finite(actions, "test"); } catch (...) { rejected = true; }
        check(rejected, "NaN rejected");
        go2w::HeightMemory memory(grid, 0.8f);
        auto scan = std::vector<float>(187, 0.4f);
        auto result = memory.update(scan, true, 0, 0.02f, {});
        check(result[0] == 5 && near(result[10], 0.4f), "visibility mask");
        for (int t = 0; t < 42; ++t) result = memory.update(scan, false, 0, 0.02f, {});
        check(result[10] == 5, "memory expires without repeated-cloud refresh");
        memory.reset();
        result = memory.update(scan, true, 0, 0.02f, {});
        result = memory.update(scan, false, 5.1f, 0.02f, {});
        check(near(result[9], 0.4f), "forward shift moves samples toward rear");
        go2w::Grid full = grid; full.mask_visible = false;
        go2w::HeightMemory pose(full, 0.8f, true);
        pose.update(scan, true, 0, 0.02f, {});
        result = pose.update(scan, false, 0, 0.02f, {0,0,0.1f,0});
        check(near(result[50], 0.5f), "pose mode compensates body height");
        result = pose.update(scan, false, 0, 0.02f, {0.1f,0,0.1f,0});
        check(near(result[40], 0.5f) && result[16] == 5, "pose mode compensates translation");
        pose.reset();
        result = pose.update(scan, false, 0, 0.02f, {});
        check(result[50] == 5, "reset clears historical terrain");
        std::cout << "Go2W core: mapping, mixed targets, grid, expiry and pose compensation passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
