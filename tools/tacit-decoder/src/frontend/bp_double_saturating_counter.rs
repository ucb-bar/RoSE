#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BpState {
    StrongNotTaken = 0b00,
    WeakNotTaken = 0b01,
    WeakTaken = 0b10,
    StrongTaken = 0b11,
}

impl From<u8> for BpState {
    fn from(value: u8) -> Self {
        match value {
            0b00 => BpState::StrongNotTaken,
            0b01 => BpState::WeakNotTaken,
            0b10 => BpState::WeakTaken,
            0b11 => BpState::StrongTaken,
            _ => panic!("Invalid BpState value"),
        }
    }
}

impl BpState {
    fn _increment(self) -> BpState {
        match self {
            BpState::StrongNotTaken => BpState::WeakNotTaken,
            BpState::WeakNotTaken => BpState::WeakTaken,
            BpState::WeakTaken => BpState::StrongTaken,
            BpState::StrongTaken => BpState::StrongTaken,
        }
    }

    fn _decrement(self) -> BpState {
        match self {
            BpState::StrongNotTaken => BpState::StrongNotTaken,
            BpState::WeakNotTaken => BpState::StrongNotTaken,
            BpState::WeakTaken => BpState::WeakNotTaken,
            BpState::StrongTaken => BpState::WeakTaken,
        }
    }

    fn judge(self) -> bool {
        match self {
            BpState::StrongNotTaken => false,
            BpState::WeakNotTaken => false,
            BpState::WeakTaken => true,
            BpState::StrongTaken => true,
        }
    }
}

pub struct BpDoubleSaturatingCounter {
    num_entries: u64,
    counters: Vec<BpState>,
}

impl BpDoubleSaturatingCounter {
    pub fn new(num_entries: u64) -> Self {
        Self {
            num_entries,
            counters: vec![BpState::WeakNotTaken; num_entries as usize],
        }
    }

    pub fn predict(&mut self, pc: u64, hit: bool) -> bool {
        let index = (pc >> 1) % self.num_entries;
        let state = self.counters[index as usize];
        let prediction = state.judge();
        if hit == false {
            if prediction == true {
                // predicted taken, but miss
                self.counters[index as usize] = state._decrement();
            } else {
                // predicted not taken, but hit
                self.counters[index as usize] = state._increment();
            }
        } else {
            // hit
            if prediction == true {
                // predicted taken, and hit
                self.counters[index as usize] = state._increment();
            } else {
                // predicted not taken, and hit
                self.counters[index as usize] = state._decrement();
            }
        }
        prediction // return the prediction before update
    }
}
