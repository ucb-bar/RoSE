use crate::frontend::br_mode::*;

use serde::Serialize;

pub const BP_MODE_MASK: u8 = 0b11;
pub const BP_ENTRY_MASK: u8 = 0b1111_1100;
pub const BP_ENTRY_OFFSET: u8 = 2;
pub const BP_BASE_VALUE: u64 = 64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DecoderRuntimeCfg {
    pub br_mode: BrMode,
    pub bp_entries: u64,
}
