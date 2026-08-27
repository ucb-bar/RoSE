use crate::backend::abstract_receiver::{AbstractReceiver, BusReceiver};
use crate::backend::event::{Entry, EventKind};
use crate::backend::stack_unwinder::{Frame, StackUnwinder, StackUpdateResult};
use crate::common::symbol_index::SymbolIndex;
use bus::BusReader;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;

pub struct AtomicReceiver {
    writer: BufWriter<File>,
    receiver: BusReceiver,
    unwinder: StackUnwinder,
    last_ts: u64,
}

impl AtomicReceiver {
    pub fn new(
        bus_rx: BusReader<Entry>,
        symbols: Arc<SymbolIndex>,
    ) -> Self {
        let unwinder = StackUnwinder::new(symbols).expect("stack unwinder");
        Self {
            writer: BufWriter::new(File::create("trace.atomics.txt").unwrap()),
            receiver: BusReceiver {
                name: "atomics".into(),
                bus_rx,
                checksum: 0,
            },
            unwinder,
            last_ts: 0,
        }
    }

    fn is_atomic_insn(insn: &rvdasm::insn::Insn) -> bool {
        let name = insn.get_name();
        name.starts_with("lr.") || name.starts_with("sc.") || name.starts_with("amo")
    }

    fn drain_update(&mut self, update: StackUpdateResult) {
        let _ = update;
    }

    fn write_stack_snapshot(&mut self) {
        writeln!(self.writer, "  Call stack:").unwrap();
        for frame in self.unwinder.peek_all_frames() {
            self.write_frame(&frame);
        }
        writeln!(self.writer).unwrap();
    }

    fn write_frame(&mut self, frame: &Frame) {
        writeln!(
            self.writer,
            "    {:?} :: {} @ 0x{:x}",
            frame.symbol.prv, frame.symbol.name, frame.addr
        )
        .unwrap();
    }
}

impl AbstractReceiver for AtomicReceiver {
    fn bus_rx(&mut self) -> &mut BusReader<Entry> {
        &mut self.receiver.bus_rx
    }

    fn _bump_checksum(&mut self) {
        self.receiver.checksum += 1;
    }

    fn _receive_entry(&mut self, entry: Entry) {
        match entry {
            Entry::Instruction { insn, pc } => {
                if Self::is_atomic_insn(&insn) {
                    writeln!(
                        self.writer,
                        "[{:>10}] 0x{:08x}: {}",
                        self.last_ts,
                        pc,
                        insn.to_string()
                    )
                    .unwrap();
                    self.write_stack_snapshot();
                }
            }
            Entry::Event { timestamp, kind } => {
                self.last_ts = timestamp;
                if let Some(update) = self.unwinder.step(&Entry::Event {
                    timestamp,
                    kind: kind.clone(),
                }) {
                    self.drain_update(update);
                }
                if matches!(kind, EventKind::SyncEnd { .. }) {
                    // nothing additional to emit
                }
            }
        }
    }

    fn _flush(&mut self) {
        self.writer.flush().unwrap();
    }
}
