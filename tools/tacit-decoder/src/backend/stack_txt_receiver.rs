use crate::backend::abstract_receiver::{AbstractReceiver, BusReceiver};
use crate::backend::event::Entry;
use crate::backend::stack_unwinder::{StackUnwinder, StackUpdateResult};
use bus::BusReader;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;

use crate::common::symbol_index::SymbolIndex;

pub struct StackTxtReceiver {
    writer: BufWriter<File>,
    receiver: BusReceiver,
    unwinder: StackUnwinder,
}

impl StackTxtReceiver {
    pub fn new(
        bus_rx: BusReader<Entry>,
        symbols: Arc<SymbolIndex>,
    ) -> Self {
        let unwinder = StackUnwinder::new(symbols).expect("init unwinder");
        Self {
            writer: BufWriter::new(File::create("trace.stack.txt").unwrap()),
            receiver: BusReceiver {
                name: "stacktxt".into(),
                bus_rx,
                checksum: 0,
            },
            unwinder,
        }
    }

    fn dump_current_stack(&mut self) -> std::io::Result<()> {
        let all_frames = self.unwinder.peek_all_frames();
        let size = all_frames.len();
        writeln!(self.writer, "  Stack (size: {})", size)?;
        for frame in all_frames {
            writeln!(
                self.writer,
                "    {:?} :: {} @ {:?}",
                frame.symbol.prv, frame.symbol.name, frame.symbol.src
            )?;
        }
        writeln!(self.writer)?;
        Ok(())
    }

    fn handle_stack_update(&mut self, ts: u64, update: StackUpdateResult) {
        for frame in update.frames_closed {
            writeln!(
                self.writer,
                "[ts {ts}] pop {:?} :: {} @ 0x{:x}",
                frame.symbol.prv, frame.symbol.name, frame.addr
            )
            .unwrap();
        }

        if let Some(frame) = update.frames_opened {
            writeln!(
                self.writer,
                "[ts {ts}] push {:?} :: {} @ 0x{:x}",
                frame.symbol.prv, frame.symbol.name, frame.addr
            )
            .unwrap();
        }

        self.dump_current_stack().unwrap();
    }
}

impl AbstractReceiver for StackTxtReceiver {
    fn bus_rx(&mut self) -> &mut BusReader<Entry> {
        &mut self.receiver.bus_rx
    }

    fn _bump_checksum(&mut self) {
        self.receiver.checksum += 1;
    }

    fn _receive_entry(&mut self, entry: Entry) {
        match entry {
            Entry::Instruction { insn: _, pc: _ } => {}
            Entry::Event { timestamp, kind } => {
                // log the event
                writeln!(self.writer, "[ts {timestamp}] {:?}", kind).unwrap();
                if let Some(update) = self.unwinder.step(&Entry::Event { timestamp, kind }) {
                    self.handle_stack_update(timestamp, update);
                }
            }
        }
    }

    fn _flush(&mut self) {
        self.writer.flush().unwrap();
    }
}
