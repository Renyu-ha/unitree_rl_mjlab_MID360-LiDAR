#pragma once
#include "FSM/CtrlFSM.h"
#include "FSM/State_FixStand.h"

namespace go2w {
class FixStand final : public State_FixStand {
public:
    using State_FixStand::State_FixStand;
    void enter() override {
        // A new manual stand request acknowledges the previous watchdog fault.
        LowCmd_t::watchdog_fault=false;
        State_FixStand::enter();
    }
};
inline void register_stand() {
    getFsmMap()["State_FixStand"]=[](int id,std::string name) -> std::shared_ptr<BaseState> {
        return std::make_shared<FixStand>(id,std::move(name));
    };
}
inline void protect_fsm(CtrlFSM& fsm) {
    const int passive=FSMStringMap.right.at("Passive");
    for (auto& state:fsm.states) if (!state->isState(passive))
        state->registered_checks.insert(state->registered_checks.begin(),{[] {
            if (LowCmd_t::watchdog_fault) return true;
            auto snapshot=FSMState::lowstate->snapshot();
            unitree::common::REMOTE_DATA_RX remote{};
            std::memcpy(remote.buff,snapshot.wireless_remote().data(),sizeof(remote));
            return remote.RF_RX.btn.components.L2 && remote.RF_RX.btn.components.B;
        },passive});
}
}
