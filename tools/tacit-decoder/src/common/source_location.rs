use crate::common::prv::Prv;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub lines: u32,
    pub prv: Prv,
}

impl SourceLocation {
    pub fn from_addr2line(loc: addr2line::Location, prv: Prv) -> Self {
        if let Some(file) = loc.file {
            SourceLocation {
                file: file.to_string(),
                lines: loc.line.unwrap_or(0),
                prv: prv,
            }
        } else {
            SourceLocation {
                file: String::new(),
                lines: 0,
                prv: prv,
            }
        }
    }
}
