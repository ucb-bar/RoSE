use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BrMode {
    BrTarget = 0b00,
    BrHistory = 0b01,
    BrPredict = 0b10,
    BrReserved = 0b11,
}

impl From<u64> for BrMode {
    fn from(value: u64) -> Self {
        match value {
            0b00 => BrMode::BrTarget,
            0b01 => BrMode::BrHistory,
            0b10 => BrMode::BrPredict,
            0b11 => BrMode::BrReserved,
            _ => panic!("Invalid BrMode value, got: {}", value),
        }
    }
}
