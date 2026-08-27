use crate::backend::abstract_receiver::{AbstractReceiver, BusReceiver};
use crate::backend::event::{Entry, EventKind};
use bus::BusReader;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

pub struct AfdoReceiver {
    writer: BufWriter<File>,
    receiver: BusReceiver,
    range_map: HashMap<(u64, u64), usize>,
    branch_map: HashMap<(u64, u64), usize>,
    last_record: (u64, u64),
    elf_start: u64,
}

impl AfdoReceiver {
    pub fn new(bus_rx: BusReader<Entry>, elf_start: u64) -> Self {
        Self {
            writer: BufWriter::new(File::create("trace_afdo.txt").unwrap()),
            receiver: BusReceiver {
                name: "afdo".to_string(),
                bus_rx: bus_rx,
                checksum: 0,
            },
            range_map: HashMap::new(),
            branch_map: HashMap::new(),
            last_record: (0, 0),
            elf_start: elf_start,
        }
    }

    pub fn update_records(&mut self, from_addr: u64, to_addr: u64, timestamp: u64) {
        self.range_map
            .entry((self.last_record.1, to_addr))
            .and_modify(|v| *v += 1)
            .or_insert(1);
        self.branch_map
            .entry((from_addr, to_addr))
            .and_modify(|v| *v += 1)
            .or_insert(1);
        self.last_record = (to_addr, timestamp);
    }
}

impl AbstractReceiver for AfdoReceiver {
    fn bus_rx(&mut self) -> &mut BusReader<Entry> {
        &mut self.receiver.bus_rx
    }

    fn _bump_checksum(&mut self) {
        self.receiver.checksum += 1;
    }

    fn _receive_entry(&mut self, entry: Entry) {
        match entry {
            Entry::Event {
                timestamp: _,
                kind:
                    EventKind::SyncStart {
                        runtime_cfg: _,
                        start_pc,
                        start_prv: _,
                        start_ctx: _,
                    },
            } => {
                self.last_record = (0, start_pc);
            }
            Entry::Event {
                timestamp,
                kind: EventKind::TakenBranch { arc },
            } => {
                self.update_records(arc.0, arc.1, timestamp);
            }
            Entry::Event {
                timestamp,
                kind: EventKind::InferrableJump { arc },
            } => {
                self.update_records(arc.0, arc.1, timestamp);
            }
            Entry::Event {
                timestamp,
                kind: EventKind::UninferableJump { arc },
            } => {
                self.update_records(arc.0, arc.1, timestamp);
            }
            _ => {}
        }
    }

    fn _flush(&mut self) {
        // write the range map
        self.writer
            .write_all(format!("{}\n", self.range_map.len()).as_bytes())
            .unwrap();
        for (key, value) in self.range_map.iter() {
            self.writer
                .write_all(
                    format!(
                        "{:x}-{:x}:{}\n",
                        key.0 - self.elf_start,
                        key.1 - self.elf_start,
                        value
                    )
                    .as_bytes(),
                )
                .unwrap();
        }
        // write the sample record, which should always be 0
        self.writer.write_all(b"0\n").unwrap();
        // write the branch map
        self.writer
            .write_all(format!("{}\n", self.branch_map.len()).as_bytes())
            .unwrap();
        for (key, value) in self.branch_map.iter() {
            self.writer
                .write_all(
                    format!(
                        "{:x}->{:x}:{}\n",
                        key.0 - self.elf_start,
                        key.1 - self.elf_start,
                        value
                    )
                    .as_bytes(),
                )
                .unwrap();
        }
        self.writer.flush().unwrap();
    }
}
