/// The tag for the end of file.
pub const EOF_TAG: u32 = 0;
/// The tag for an [`ANNOUNCE_FUNCTION` record](./struct.Function.html).
pub const FUNCTION_TAG: u32 = 0x01_00_00_00;
/// The tag for a [`BASIC_BLOCK` record](./struct.Blocks.html).
pub const BLOCKS_TAG: u32 = 0x01_41_00_00;
/// The tag for an [`ARCS` record](./struct.Arcs.html).
pub const ARCS_TAG: u32 = 0x01_43_00_00;
/// The tag for a [`LINES` record](./struct.Lines.html).
pub const LINES_TAG: u32 = 0x01_45_00_00;
/// The tag for a [`COUNTS` record](./struct.ArcCounts.html).
pub const COUNTER_BASE_TAG: u32 = 0x01_a1_00_00;
/// The tag for a [`SUMMARY` record](./struct.Summary.html).
pub const OBJECT_SUMMARY_TAG: u32 = 0xa1_00_00_00;
/// The tag for a program-`SUMMARY` record, which has been deprecated and is always skipped when present.
pub const PROGRAM_SUMMARY_TAG: u32 = 0xa3_00_00_00;

pub const FLAG_TREE: u32 = 0x1;
pub const FLAG_FAKE: u32 = 0x2;
pub const FLAG_FALL: u32 = 0x4;
