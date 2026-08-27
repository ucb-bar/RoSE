use crate::backend::abstract_receiver::{AbstractReceiver, BusReceiver};
use crate::backend::event::{Entry, EventKind};
use crate::backend::stack_unwinder::{Frame, StackUnwinder, StackUpdateResult};
use crate::common::prv::Prv;
use crate::common::symbol_index::SymbolIndex;
use bus::BusReader;
use log::{debug, warn};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;

#[derive(Serialize)]
struct ProfileEvent {
    #[serde(rename = "type")]
    kind: String,
    frame: u32,
    at: u64,
}

struct Lookup {
    u_lookup: HashMap<u64, HashMap<u64, u32>>,
    k_lookup: HashMap<u64, u32>,
    m_lookup: HashMap<u64, u32>,
}

impl Lookup {
    fn lookup(&self, prv: Prv, ctx: u64, addr: u64) -> Option<u32> {
        match prv {
            Prv::PrvUser => self
                .u_lookup
                .get(&ctx)
                .and_then(|lookup| lookup.get(&addr))
                .copied(),
            Prv::PrvSupervisor => self.k_lookup.get(&addr).copied(),
            Prv::PrvMachine => self.m_lookup.get(&addr).copied(),
            _ => panic!("Unsupported privilege level: {:?}", prv),
        }
    }
}

pub struct SpeedscopeReceiver {
    writer: BufWriter<File>,
    receiver: BusReceiver,
    frames: Vec<Value>,
    frame_lookup: Lookup,
    start: u64,
    end: u64,
    events: Vec<ProfileEvent>,
    unwinder: StackUnwinder,
}

impl SpeedscopeReceiver {
    pub fn new(
        bus_rx: BusReader<Entry>,
        symbols: Arc<SymbolIndex>,
    ) -> Self {
        debug!("SpeedscopeReceiver::new");

        let unwinder =
            StackUnwinder::new(Arc::clone(&symbols)).expect("stack unwinder");

        let (frames, frame_lookup) = build_frames(&symbols);

        Self {
            writer: BufWriter::new(File::create("trace.speedscope.json").unwrap()),
            receiver: BusReceiver {
                name: "speedscope".into(),
                bus_rx,
                checksum: 0,
            },
            frames,
            frame_lookup,
            start: 0,
            end: 0,
            events: Vec::new(),
            unwinder,
        }
    }

    fn record_stack_update(&mut self, ts: u64, update: StackUpdateResult) {
        for frame in update.frames_closed {
            if let Some(id) = self.lookup_frame(&frame) {
                self.events.push(ProfileEvent {
                    kind: "C".into(),
                    frame: id,
                    at: ts,
                });
            } else {
                warn!("Frame not found: {:?}", frame);
            }
        }

        if let Some(frame) = update.frames_opened {
            if let Some(id) = self.lookup_frame(&frame) {
                self.events.push(ProfileEvent {
                    kind: "O".into(),
                    frame: id,
                    at: ts,
                });
            } else {
                warn!("Frame not found: {:?}", frame);
            }
        }
    }

    fn lookup_frame(&self, frame: &Frame) -> Option<u32> {
        let id = self
            .frame_lookup
            .lookup(frame.symbol.prv, frame.symbol.ctx, frame.addr);
        id
    }
}

impl AbstractReceiver for SpeedscopeReceiver {
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
                match &kind {
                    EventKind::SyncStart {
                        start_prv: _,
                        start_pc: _,
                        ..
                    } => {
                        self.start = timestamp;
                    }
                    EventKind::SyncEnd { .. } => {
                        self.end = timestamp;
                    }
                    _ => {}
                }

                if let Some(update) = self.unwinder.step(&Entry::Event {
                    timestamp,
                    kind: kind.clone(),
                }) {
                    self.record_stack_update(timestamp, update);
                }
            }
        }
    }

    fn _flush(&mut self) {
        if self.end == 0 {
            if let Some(last) = self.events.last() {
                self.end = last.at;
            }
        }

        if let Some(update) = self.unwinder.flush() {
            // close any remaining frames at self.end
            for frame in update.frames_closed {
                if let Some(id) = self.lookup_frame(&frame) {
                    self.events.push(ProfileEvent {
                        kind: "C".into(),
                        frame: id,
                        at: self.end,
                    });
                }
            }
        }

        write_speedscope(
            &mut self.writer,
            &self.frames,
            &self.events,
            self.start,
            self.end,
        )
        .expect("write speedscope");

        self.writer.flush().unwrap();
    }
}

/// Build Speedscope frames and an address→frame-id lookup.
fn build_frames(symbols: &SymbolIndex) -> (Vec<Value>, Lookup) {
    let mut frames = Vec::new();
    let mut u_lookups: HashMap<u64, HashMap<u64, u32>> = HashMap::new();
    let mut k_lookup: HashMap<u64, u32> = HashMap::new();
    let mut m_lookup: HashMap<u64, u32> = HashMap::new();

    for (&addr, info) in symbols.get(Prv::PrvSupervisor, 0).iter() {
        let id = frames.len() as u32;
        frames.push(json!({
            "name": format!("{}:{}", "k", info.name),
            "file": info.src.file,
            "line": info.src.lines,
        }));
        k_lookup.insert(addr, id);
    }

    for (&addr, info) in symbols.get(Prv::PrvMachine, 0).iter() {
        let id = frames.len() as u32;
        frames.push(json!({
            "name": format!("{}:{}", "m", info.name),
            "file": info.src.file,
            "line": info.src.lines,
        }));
        m_lookup.insert(addr, id);
    }
    // iterate over the user space symbol map
    for (&asid, user_symbol_map) in symbols.get_user_symbol_map().iter() {
        let mut u_lookup: HashMap<u64, u32> = HashMap::new();
        for (&addr, info) in user_symbol_map.iter() {
            let id = frames.len() as u32;
            frames.push(json!({
                "name": format!("{}:{}", asid, info.name),
                "file": info.src.file,
                "line": info.src.lines,
            }));
            u_lookup.insert(addr, id);
        }
        u_lookups.insert(asid, u_lookup);
    }

    let lookup = Lookup {
        u_lookup: u_lookups,
        k_lookup: k_lookup,
        m_lookup: m_lookup,
    };

    (frames, lookup)
}

fn write_speedscope(
    writer: &mut BufWriter<File>,
    frames: &[Value],
    events: &[ProfileEvent],
    start: u64,
    end: u64,
) -> std::io::Result<()> {
    writeln!(writer, "{{")?;
    writeln!(writer, "  \"version\": \"0.0.1\",")?;
    writeln!(
        writer,
        "  \"$schema\": \"https://www.speedscope.app/file-format-schema.json\","
    )?;
    writeln!(writer, "  \"shared\": {{")?;
    writeln!(writer, "    \"frames\": [")?;
    for (i, frame) in frames.iter().enumerate() {
        let comma = if i + 1 < frames.len() { "," } else { "" };
        writeln!(writer, "      {}{}", frame.to_string(), comma)?;
    }
    writeln!(writer, "    ]")?;
    writeln!(writer, "  }},")?;
    writeln!(writer, "  \"profiles\": [")?;
    writeln!(writer, "    {{")?;
    writeln!(writer, "      \"name\": \"tacit\",")?;
    writeln!(writer, "      \"type\": \"evented\",")?;
    writeln!(writer, "      \"unit\": \"none\",")?;
    writeln!(writer, "      \"startValue\": {},", start)?;
    writeln!(writer, "      \"endValue\": {},", end)?;
    writeln!(writer, "      \"events\": [")?;

    for (i, ev) in events.iter().enumerate() {
        let comma = if i + 1 < events.len() { "," } else { "" };
        writeln!(
            writer,
            "        {{\"type\": \"{}\", \"frame\": {}, \"at\": {}}}{}",
            ev.kind, ev.frame, ev.at, comma
        )?;
    }

    writeln!(writer, "      ]")?;
    writeln!(writer, "    }}")?;
    writeln!(writer, "  ]")?;
    writeln!(writer, "}}")?;

    Ok(())
}
