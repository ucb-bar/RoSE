pub const C_HEADER_MASK: u8 = 0b0000_0011;
pub const C_TIMESTAMP_MASK: u8 = 0b1111_1100;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CHeader {
    CTb = 0b00, // taken branch
    CNt = 0b01, // not taken branch
    CNa = 0b10, // not applicable
    CIj = 0b11, // inferable jump
}

impl From<u8> for CHeader {
    fn from(value: u8) -> Self {
        match value {
            0b00 => CHeader::CTb,
            0b01 => CHeader::CNt,
            0b10 => CHeader::CNa,
            0b11 => CHeader::CIj,
            _ => panic!("Invalid CHeader value"),
        }
    }
}
