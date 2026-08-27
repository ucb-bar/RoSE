use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash, PartialOrd, Ord)]
pub enum Prv {
    PrvUser = 0b000,
    PrvSupervisor = 0b001,
    PrvHypervisor = 0b010,
    PrvMachine = 0b011,
}

impl From<u64> for Prv {
    fn from(value: u64) -> Self {
        match value {
            0b000 => Prv::PrvUser,       // baremetal, no context
            0b001 => Prv::PrvSupervisor, // User-space only context
            0b010 => Prv::PrvHypervisor, // watch for a specific context
            0b011 => Prv::PrvMachine,    // all contexts
            _ => panic!("Invalid Privllege Level value, got: {}", value),
        }
    }
}
