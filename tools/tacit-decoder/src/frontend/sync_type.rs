pub const SYNC_TYPE_MASK: u8 = 0b1110_0000;
pub const SYNC_TYPE_OFFSET: u8 = 5;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SyncType {
    SyncNone = 0b000,
    SyncStart = 0b001,
    SyncPeriodic = 0b010,
    SyncEnd = 0b011,
}

impl From<u8> for SyncType {
    fn from(value: u8) -> Self {
        match value {
            0b000 => SyncType::SyncNone,
            0b001 => SyncType::SyncStart,
            0b010 => SyncType::SyncPeriodic,
            0b011 => SyncType::SyncEnd,
            _ => panic!("Invalid SyncType value"),
        }
    }
}
