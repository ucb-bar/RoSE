use crate::backend::abstract_receiver::{AbstractReceiver, BusReceiver};
use crate::backend::event::{Entry, EventKind};
use crate::backend::stack_unwinder::{Frame, StackUnwinder, StackUpdateResult};
use crate::common::symbol_index::SymbolIndex;

use bus::BusReader;
use serde_json::json;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;

/// Emit a Perfetto/Chrome trace by following the unwinder's stack updates.
pub struct PerfettoReceiver {
    writer: BufWriter<File>,
    receiver: BusReceiver,
    unwinder: StackUnwinder,
    events: Vec<String>,
    start_ts: Option<u64>,
    end_ts: Option<u64>,
    last_ts: u64,
}

impl PerfettoReceiver {
    pub fn new(
        bus_rx: BusReader<Entry>,
        symbols: Arc<SymbolIndex>,
    ) -> Self {
        let unwinder = StackUnwinder::new(symbols).expect("stack unwinder");
        PerfettoReceiver {
            writer: BufWriter::new(File::create("trace.perfetto.json").unwrap()),
            receiver: BusReceiver {
                name: "perfetto".into(),
                bus_rx,
                checksum: 0,
            },
            unwinder,
            events: Vec::new(),
            start_ts: None,
            end_ts: None,
            last_ts: 0,
        }
    }

    fn drain_update(&mut self, ts: u64, update: StackUpdateResult) {
        for frame in update.frames_closed {
            self.emit_end(ts, &frame);
        }
        if let Some(frame) = update.frames_opened {
            self.emit_begin(ts, &frame);
        }
    }

    fn emit_begin(&mut self, ts: u64, frame: &Frame) {
        let evt = json!({
            "name": frame.symbol.name,
            "cat": "function",
            "ph": "B",
            "ts": ts,
            "pid": 0,
            "tid": 0,
            "args": {
                "addr": format!("0x{:x}", frame.addr),
                "prv": format!("{:?}", frame.symbol.prv),
                "file": frame.symbol.src.file,
                "line": frame.symbol.src.lines,
            }
        });
        self.events.push(evt.to_string());
    }

    fn emit_end(&mut self, ts: u64, frame: &Frame) {
        let evt = json!({
            "name": frame.symbol.name,
            "cat": "function",
            "ph": "E",
            "ts": ts,
            "pid": 0,
            "tid": 0,
            "args": {}
        });
        self.events.push(evt.to_string());
    }

    fn maybe_record_sync_markers(&mut self, ts: u64, kind: &EventKind) {
        match kind {
            EventKind::SyncStart { .. } => {
                self.start_ts.get_or_insert(ts);
            }
            EventKind::SyncEnd { .. } => {
                self.end_ts = Some(ts);
            }
            _ => {}
        }
    }

    fn close_remaining_frames(&mut self, ts: u64) {
        if let Some(update) = self.unwinder.flush() {
            // flush() returns the currently open frames as closed ones.
            self.drain_update(ts, update);
        }
    }
}

impl AbstractReceiver for PerfettoReceiver {
    fn bus_rx(&mut self) -> &mut BusReader<Entry> {
        &mut self.receiver.bus_rx
    }

    fn _bump_checksum(&mut self) {
        self.receiver.checksum += 1;
    }

    fn _receive_entry(&mut self, entry: Entry) {
        match entry {
            Entry::Instruction { .. } => {}
            Entry::Event { timestamp, kind } => {
                self.last_ts = timestamp;
                self.maybe_record_sync_markers(timestamp, &kind);
                let event_entry = Entry::Event {
                    timestamp,
                    kind: kind.clone(),
                };
                if let Some(update) = self.unwinder.step(&event_entry) {
                    self.drain_update(timestamp, update);
                }
            }
        }
    }

    fn _flush(&mut self) {
        let final_ts = self.end_ts.or(self.start_ts).unwrap_or(self.last_ts);
        self.close_remaining_frames(final_ts);

        if self.end_ts.is_none() {
            self.end_ts = Some(final_ts);
        }
        if self.start_ts.is_none() {
            self.start_ts = Some(final_ts);
        }

        writeln!(self.writer, "{{").unwrap();
        writeln!(self.writer, "  \"traceEvents\": [").unwrap();
        for (i, ev) in self.events.iter().enumerate() {
            let comma = if i + 1 < self.events.len() { "," } else { "" };
            writeln!(self.writer, "    {}{}", ev, comma).unwrap();
        }
        writeln!(self.writer, "  ]").unwrap();
        writeln!(self.writer, "}}").unwrap();
        self.writer.flush().unwrap();
    }
}
