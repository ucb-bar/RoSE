use crate::backend::abstract_receiver::{AbstractReceiver, BusReceiver};
use crate::backend::event::{Entry, EventKind};
use crate::frontend::br_mode;
use crate::frontend::runtime_cfg::DecoderRuntimeCfg;
use bus::BusReader;
use std::fs::File;
use std::io::{BufWriter, Write};

pub struct StatsReceiver {
    writer: BufWriter<File>,
    receiver: BusReceiver,
    runtime_cfg: DecoderRuntimeCfg,
    file_size: u64,
    packet_count: u64,
    insn_count: u64,
    hit_count: u64,
    miss_count: u64,
}

impl StatsReceiver {
    pub fn new(bus_rx: BusReader<Entry>, runtime_cfg: DecoderRuntimeCfg, file_size: u64) -> Self {
        Self {
            writer: BufWriter::new(File::create("trace.stats.txt").unwrap()),
            receiver: BusReceiver {
                name: "stats".to_string(),
                bus_rx: bus_rx,
                checksum: 0,
            },
            packet_count: 0,
            insn_count: 0,
            hit_count: 0,
            miss_count: 0,
            runtime_cfg: runtime_cfg,
            file_size: file_size,
        }
    }
}

impl AbstractReceiver for StatsReceiver {
    fn bus_rx(&mut self) -> &mut BusReader<Entry> {
        &mut self.receiver.bus_rx
    }

    fn _bump_checksum(&mut self) {
        self.receiver.checksum += 1;
    }

    fn _receive_entry(&mut self, entry: Entry) {
        match entry {
            Entry::Event { kind, .. } => {
                self.packet_count += 1;
                match kind {
                    EventKind::BPHit { hit_count } => {
                        if self.runtime_cfg.br_mode == br_mode::BrMode::BrPredict {
                            self.hit_count += hit_count;
                        }
                    }
                    EventKind::BPMiss => {
                        if self.runtime_cfg.br_mode == br_mode::BrMode::BrPredict {
                            self.miss_count += 1;
                        }
                    }
                    _ => {}
                }
            }
            Entry::Instruction { insn: _, pc: _ } => {
                self.insn_count += 1;
            }
        }
    }

    fn _flush(&mut self) {
        self.writer
            .write_all(format!("instruction count: {}\n", self.insn_count).as_bytes())
            .unwrap();
        self.writer
            .write_all(format!("packet count: {}\n", self.packet_count).as_bytes())
            .unwrap();
        if self.runtime_cfg.br_mode == br_mode::BrMode::BrPredict {
            self.writer
                .write_all(
                    format!(
                        "hit rate: {:.2}%\n",
                        self.hit_count as f64 / (self.hit_count + self.miss_count) as f64 * 100.0
                    )
                    .as_bytes(),
                )
                .unwrap();
        }
        let bpi = self.file_size as f64 * 8.0 / self.insn_count as f64; //convert bytes to bits
        self.writer
            .write_all(format!("bits per instruction: {:.4}\n", bpi).as_bytes())
            .unwrap();
        self.writer
            .write_all(
                format!(
                    "trace payload size: {:.2}KiB\n",
                    self.file_size as f64 / 1024.0
                )
                .as_bytes(),
            )
            .unwrap();
        let bpp = self.file_size as f64 * 8.0 / self.packet_count as f64;
        self.writer
            .write_all(format!("bits per packet: {:.4}\n", bpp).as_bytes())
            .unwrap();
        self.writer.flush().unwrap();
    }
}
