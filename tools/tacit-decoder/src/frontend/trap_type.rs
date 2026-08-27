pub const TRAP_TYPE_MASK: u8 = 0b1110_0000;
pub const TRAP_TYPE_OFFSET: u8 = 5;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TrapType {
    TNone = 0b000,
    TException = 0b001,
    TInterrupt = 0b010,
    TReturn = 0b100,
}

impl From<u8> for TrapType {
    fn from(value: u8) -> Self {
        match value {
            0b000 => TrapType::TNone,
            0b001 => TrapType::TException,
            0b010 => TrapType::TInterrupt,
            0b100 => TrapType::TReturn,
            _ => panic!("Invalid TrapType value"),
        }
    }
}
