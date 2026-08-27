use crate::common::prv::Prv;
use crate::frontend::runtime_cfg::DecoderRuntimeCfg;
use crate::frontend::trap_type::TrapType;
use rvdasm::insn::Insn;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub enum Entry {
    Instruction { insn: Insn, pc: u64 },
    Event { timestamp: u64, kind: EventKind },
}

#[derive(Debug, Clone, Serialize)]
pub enum EventKind {
    TakenBranch {
        arc: (u64, u64),
    },
    NonTakenBranch {
        arc: (u64, u64),
    },
    UninferableJump {
        arc: (u64, u64),
    },
    InferrableJump {
        arc: (u64, u64),
    },
    Trap {
        reason: TrapReason,
        prv_arc: (Prv, Prv),
        arc: (u64, u64),
        ctx: Option<u64>,
    },
    SyncStart {
        runtime_cfg: DecoderRuntimeCfg,
        start_pc: u64,
        start_prv: Prv,
        start_ctx: u64,
    },
    SyncEnd {
        end_pc: u64,
    },
    SyncPeriodic,
    BPHit {
        hit_count: u64,
    },
    BPMiss,
    Panic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum TrapReason {
    Exception,
    Interrupt,
    Return,
}

impl Entry {
    pub fn event(event: EventKind, timestamp: u64) -> Self {
        Entry::Event {
            timestamp,
            kind: event,
        }
    }

    pub fn instruction(insn: &Insn, pc: u64) -> Self {
        Entry::Instruction {
            insn: insn.clone(),
            pc,
        }
    }
}

impl EventKind {
    // constructors
    pub fn taken_branch(arc: (u64, u64)) -> Self {
        EventKind::TakenBranch { arc }
    }

    pub fn non_taken_branch(arc: (u64, u64)) -> Self {
        EventKind::NonTakenBranch { arc }
    }

    pub fn uninferable_jump(arc: (u64, u64)) -> Self {
        EventKind::UninferableJump { arc }
    }

    pub fn inferrable_jump(arc: (u64, u64)) -> Self {
        EventKind::InferrableJump { arc }
    }

    pub fn trap(reason: TrapReason, prv_arc: (Prv, Prv), arc: (u64, u64)) -> Self {
        EventKind::Trap {
            reason,
            prv_arc,
            arc,
            ctx: None,
        }
    }

    pub fn trap_with_ctx(
        reason: TrapReason,
        prv_arc: (Prv, Prv),
        arc: (u64, u64),
        ctx: u64,
    ) -> Self {
        EventKind::Trap {
            reason,
            prv_arc,
            arc,
            ctx: Some(ctx),
        }
    }

    pub fn sync_start(
        runtime_cfg: DecoderRuntimeCfg,
        start_pc: u64,
        start_prv: Prv,
        start_ctx: u64,
    ) -> Self {
        EventKind::SyncStart {
            runtime_cfg,
            start_pc,
            start_prv,
            start_ctx,
        }
    }

    pub fn sync_end(end_pc: u64) -> Self {
        EventKind::SyncEnd { end_pc }
    }

    // pub fn sync_periodic() -> Self {
    //     EventKind::SyncPeriodic
    // }

    pub fn bphit(hit_count: u64) -> Self {
        EventKind::BPHit { hit_count }
    }

    pub fn bpmiss() -> Self {
        EventKind::BPMiss
    }

    // pub fn context_change(ctx: u64) -> Self {
    //     EventKind::ContextChange { ctx }
    // }

    pub fn panic() -> Self {
        EventKind::Panic
    }

    // getters
    // pub fn get_from_addr(&self) -> Option<u64> {
    //     match self {
    //         EventKind::TakenBranch { arc } => Some(arc.0),
    //         EventKind::UninferableJump { arc } => Some(arc.0),
    //         EventKind::InferrableJump { arc } => Some(arc.0),
    //         _ => None,
    //     }
    // }

    // pub fn get_to_addr(&self) -> Option<u64> {
    //     match self {
    //         EventKind::TakenBranch { arc } => Some(arc.1),
    //         EventKind::UninferableJump { arc } => Some(arc.1),
    //         EventKind::InferrableJump { arc } => Some(arc.1),
    //         _ => None,
    //     }
    // }

    // pub fn get_trap_reason(&self) -> Option<TrapReason> {
    //     match self {
    //         EventKind::Trap { reason, .. } => Some(reason.clone()),
    //         _ => None,
    //     }
    // }

    // pub fn get_prv_arc(&self) -> Option<(Prv, Prv)> {
    //     match self {
    //         EventKind::Trap { prv_arc, .. } => Some(prv_arc.clone()),
    //         _ => None,
    //     }
    // }

    // pub fn get_ctx(&self) -> Option<u64> {
    //     match self {
    //         EventKind::ContextChange { ctx } => Some(ctx.clone()),
    //         _ => None,
    //     }
    // }
}

impl From<TrapType> for TrapReason {
    fn from(trap_type: TrapType) -> Self {
        match trap_type {
            TrapType::TException => TrapReason::Exception,
            TrapType::TInterrupt => TrapReason::Interrupt,
            TrapType::TReturn => TrapReason::Return,
            TrapType::TNone => panic!("TNone should not be converted to TrapReason"),
        }
    }
}
