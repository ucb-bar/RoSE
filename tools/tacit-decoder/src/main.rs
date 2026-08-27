extern crate bus;
extern crate clap;
extern crate env_logger;
extern crate gcno_reader;
extern crate indicatif;
extern crate log;
extern crate object;
extern crate rvdasm;
extern crate rustc_data_structures;

mod frontend {
    pub mod bp_double_saturating_counter;
    pub mod br_mode;
    pub mod c_header;
    pub mod ctx_mode;
    pub mod decoder;
    pub mod decoder_cache;
    pub mod f_header;
    pub mod packet;
    pub mod runtime_cfg;
    pub mod sync_type;
    pub mod trap_type;
}

mod backend {
    pub mod abstract_receiver;
    pub mod afdo_receiver;
    pub mod atomic_receiver;
    pub mod event;
    pub mod gcda_receiver;
    pub mod path_profile_receiver;
    pub mod perfetto_receiver;
    pub mod speedscope_receiver;
    pub mod stack_txt_receiver;
    pub mod stack_unwinder;
    pub mod stats_receiver;
    pub mod txt_receiver;
    pub mod vbb_receiver;
}

mod common {
    pub mod insn_index;
    pub mod prv;
    pub mod source_location;
    pub mod static_cfg;
    pub mod symbol_index;
}

// file IO
use std::fs::File;
use std::io::{BufReader, Read};
// argparse dependency
use clap::Parser;
use object::Object;
// path dependency
use std::path::Path;
use std::io::Write;
// bus dependency
use bus::Bus;
use std::thread;
// backend dependency
use backend::abstract_receiver::AbstractReceiver;
use backend::afdo_receiver::AfdoReceiver;
use backend::atomic_receiver::AtomicReceiver;
use backend::event::Entry;
use backend::gcda_receiver::GcdaReceiver;
use backend::path_profile_receiver::PathProfileReceiver;
use backend::perfetto_receiver::PerfettoReceiver;
use backend::speedscope_receiver::SpeedscopeReceiver;
use backend::stack_txt_receiver::StackTxtReceiver;
use backend::stats_receiver::StatsReceiver;
use backend::txt_receiver::TxtReceiver;
use backend::vbb_receiver::VBBReceiver;
use common::static_cfg::{load_file_config, DecoderStaticCfg};
// error handling
use anyhow::Result;

const BUS_SIZE: usize = 1024;

#[derive(Clone, Parser)]
#[command(
    name = "trace-decoder",
    version = "0.1.0",
    about = "Decode trace files"
)]
struct Args {
    // optional JSON config file to control receivers
    #[arg(long)]
    config: Option<String>,
    // path to the encoded trace file
    #[arg(short, long)]
    encoded_trace: Option<String>,
    // path to the binary file
    #[arg(long)]
    application_binary: Option<String>,
    // path to the sbi binary file
    #[arg(long)]
    sbi_binary: Option<String>,
    // path to the kernel binary file
    #[arg(long)]
    kernel_binary: Option<String>,
    // optionally write the final receiver config to JSON
    #[arg(long)]
    dump_effective_config: Option<String>,
    // dump the symbol index to a JSON file
    #[arg(long)]
    dump_symbol_index: Option<String>,
    // print the header configuration and exit
    #[arg(long)]
    header_only: Option<bool>,
    // output the decoded trace in stats format
    #[arg(long)]
    to_stats: Option<bool>,
    // output the decoded trace in text format
    #[arg(long)]
    to_txt: Option<bool>,
    // output the tracked callstack in text format
    #[arg(long)]
    to_stack_txt: Option<bool>,
    // output a trace of atomic operations in text format
    #[arg(long)]
    to_atomics: Option<bool>,
    // output the decoded trace in afdo format
    #[arg(long)]
    to_afdo: Option<bool>,
    // path to the gcno file, must be provided if to_afdo is true
    #[arg(long)]
    gcno: Option<String>,
    // output the decoded trace in gcda format
    #[arg(long)]
    to_gcda: Option<bool>,
    // output the decoded trace in speedscope format
    #[arg(long)]
    to_speedscope: Option<bool>,
    // output the decoded trace in perfetto format
    #[arg(long)]
    to_perfetto: Option<bool>,
    // output the decoded trace in foc format
    #[arg(long)]
    to_foc: Option<bool>,
    // output the decoded trace in vbb format
    #[arg(long)]
    to_vbb: Option<bool>,
    // output the decoded trace in path profile format
    #[arg(long)]
    to_path_profile: Option<bool>,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    // If a config file is supplied, use it for receiver toggles (CLI still supplies paths).
    let file_cfg = if let Some(path) = &args.config {
        load_file_config(path)?
    } else {
        DecoderStaticCfg::default()
    };

    fn pick_arg<T: Clone>(cli: Option<T>, file: T) -> T {
        cli.unwrap_or(file)
    }

    // Resolve toggles: config file takes precedence if provided; otherwise use CLI flags
    let encoded_trace = pick_arg(args.encoded_trace, file_cfg.encoded_trace);
    let application_binary_asid_tuples = file_cfg.application_binary_asid_tuples;
    let kernel_binary = pick_arg(args.kernel_binary, file_cfg.kernel_binary);
    let kernel_jump_label_patch_log = file_cfg.kernel_jump_label_patch_log;
    let driver_binary_entry_tuples = file_cfg.driver_binary_entry_tuples;
    let sbi_binary = pick_arg(args.sbi_binary, file_cfg.sbi_binary);
    let header_only = pick_arg(args.header_only, file_cfg.header_only);
    let to_stats = pick_arg(args.to_stats, file_cfg.to_stats);
    let to_txt = pick_arg(args.to_txt, file_cfg.to_txt);
    let to_stack_txt = pick_arg(args.to_stack_txt, file_cfg.to_stack_txt);
    let to_atomics = pick_arg(args.to_atomics, file_cfg.to_atomics);
    let to_afdo = pick_arg(args.to_afdo, file_cfg.to_afdo);
    let gcno_path = pick_arg(args.gcno, file_cfg.gcno);
    let to_gcda = pick_arg(args.to_gcda, file_cfg.to_gcda);
    let to_speedscope = pick_arg(args.to_speedscope, file_cfg.to_speedscope);
    let to_perfetto = pick_arg(args.to_perfetto, file_cfg.to_perfetto);
    let to_vbb = pick_arg(args.to_vbb, file_cfg.to_vbb);
    let to_path_profile = pick_arg(args.to_path_profile, file_cfg.to_path_profile);
    let static_cfg = DecoderStaticCfg {
        encoded_trace,
        application_binary_asid_tuples,
        kernel_binary,
        kernel_jump_label_patch_log,
        driver_binary_entry_tuples,
        sbi_binary,
        header_only,
        to_stats,
        to_txt,
        to_stack_txt,
        to_atomics,
        to_afdo,
        gcno: gcno_path.clone(),
        to_gcda,
        to_speedscope,
        to_perfetto,
        to_vbb,
        to_path_profile,
    };

    // verify the binary exists and is a file
    for (binary, _) in static_cfg.application_binary_asid_tuples.clone() {
        if !Path::new(&binary).exists() || !Path::new(&binary).is_file() {
            return Err(anyhow::anyhow!(
                "Application binary file is not valid: {}",
                binary
            ));
        }
    }
    if static_cfg.sbi_binary != ""
        && (!Path::new(&static_cfg.sbi_binary).exists()
            || !Path::new(&static_cfg.sbi_binary).is_file())
    {
        return Err(anyhow::anyhow!(
            "SBI binary file is not valid: {}",
            static_cfg.sbi_binary
        ));
    }
    if static_cfg.kernel_binary != ""
        && (!Path::new(&static_cfg.kernel_binary).exists()
            || !Path::new(&static_cfg.kernel_binary).is_file())
    {
        return Err(anyhow::anyhow!(
            "Kernel binary file is not valid: {}",
            static_cfg.kernel_binary
        ));
    }

    // verify the encoded trace exists and is a file
    if !Path::new(&static_cfg.encoded_trace).exists()
        || !Path::new(&static_cfg.encoded_trace).is_file()
    {
        return Err(anyhow::anyhow!(
            "Encoded trace file is not valid: {}",
            static_cfg.encoded_trace
        ));
    }

    if let Some(path) = &args.dump_effective_config {
        let mut f = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(&mut f, &static_cfg)?;
    }

    let (first_packet, runtime_cfg, trace_format) = {
        let trace_file = File::open(static_cfg.encoded_trace.clone())?;
        let mut trace_reader = BufReader::new(trace_file);
        let (first_packet, runtime_cfg, trace_format) =
            frontend::packet::read_first_packet(&mut trace_reader)?;
        drop(trace_reader);
        (first_packet, runtime_cfg, trace_format)
    };

    if static_cfg.header_only {
        println!("Detected trace format: {:?}", trace_format);
        println!("Printing header configuration: {:?}", runtime_cfg);
        println!("Printing first packet: {:?}", first_packet);
        println!(
            "Printing starting address: 0x{:x}",
            (first_packet.target_address << 1)
        );
        println!("Printing starting prv: {:?}", first_packet.target_prv);
        std::process::exit(0);
    }

    // Build instruction index once
    // Notice that this operation is very slow for large binaries
    let insn_index = common::insn_index::build_instruction_index(static_cfg.clone())?;
    let insn_index = std::sync::Arc::new(insn_index);

    // Build symbol index once
    let symbol_index = common::symbol_index::build_symbol_index(static_cfg.clone())?;
    let symbol_index = std::sync::Arc::new(symbol_index);

    if let Some(path) = &args.dump_symbol_index {
        let mut f = std::fs::File::create(path)?;
        for (asid, symbol_map) in symbol_index.get_user_symbol_map().iter() {
            for (addr, symbol) in symbol_map.iter() {
                writeln!(f, "{} 0x{:x} {}", symbol.name, addr, asid).unwrap();
            }
        }
        for (addr, symbol) in symbol_index.get_kernel_symbol_map().iter() {
            writeln!(f, "{} 0x{:x} {}", symbol.name, addr, 0).unwrap();
        }
        for (addr, symbol) in symbol_index.get_machine_symbol_map().iter() {
            writeln!(f, "{} 0x{:x} {}", symbol.name, addr, 0).unwrap();
        }
    }

    let mut bus: Bus<Entry> = Bus::new(BUS_SIZE);
    let mut receivers: Vec<Box<dyn AbstractReceiver>> = vec![];

    // add a receiver to the bus for stats output
    if to_stats {
        let encoded_trace_file = File::open(static_cfg.encoded_trace.clone())?;
        // get the file size
        let file_size = encoded_trace_file.metadata()?.len();
        // close the file
        drop(encoded_trace_file);
        let stats_bus_endpoint = bus.add_rx();
        receivers.push(Box::new(StatsReceiver::new(
            stats_bus_endpoint,
            runtime_cfg.clone(),
            file_size,
        )));
    }

    // add a receiver to the bus for txt output
    if to_txt {
        let txt_bus_endpoint = bus.add_rx();
        receivers.push(Box::new(TxtReceiver::new(txt_bus_endpoint)));
    }

    if to_stack_txt {
        let to_stack_txt_symbol_index = std::sync::Arc::clone(&symbol_index);
        let stack_txt_rx = StackTxtReceiver::new(
            bus.add_rx(),
            to_stack_txt_symbol_index,
        );
        receivers.push(Box::new(stack_txt_rx));
    }

    if to_atomics {
        let to_atomics_symbol_index = std::sync::Arc::clone(&symbol_index);
        let atomic_rx =
            AtomicReceiver::new(bus.add_rx(), to_atomics_symbol_index);
        receivers.push(Box::new(atomic_rx));
    }

    if to_afdo {
        let afdo_bus_endpoint = bus.add_rx();
        let mut elf_file = File::open(static_cfg.application_binary_asid_tuples[0].0.clone())?;
        let mut elf_buffer = Vec::new();
        elf_file.read_to_end(&mut elf_buffer)?;
        let elf = object::File::parse(&*elf_buffer)?;
        receivers.push(Box::new(AfdoReceiver::new(
            afdo_bus_endpoint,
            elf.entry().clone(),
        )));
        drop(elf_file);
    }

    if to_gcda {
        let gcda_bus_endpoint = bus.add_rx();
        receivers.push(Box::new(GcdaReceiver::new(
            gcda_bus_endpoint,
            gcno_path.clone(),
            static_cfg.application_binary_asid_tuples[0].0.clone(),
        )));
    }

    if to_speedscope {
        let speedscope_bus_endpoint = bus.add_rx();
        let speedscope_symbol_index = std::sync::Arc::clone(&symbol_index);
        receivers.push(Box::new(SpeedscopeReceiver::new(
            speedscope_bus_endpoint,
            speedscope_symbol_index,
        )));
    }

    if to_perfetto {
        let perfetto_bus_endpoint = bus.add_rx();
        let perfetto_symbol_index = std::sync::Arc::clone(&symbol_index);
        receivers.push(Box::new(PerfettoReceiver::new(
            perfetto_bus_endpoint,
            perfetto_symbol_index,
        )));
    }

    if to_vbb {
        let vbb_bus_endpoint = bus.add_rx();
        receivers.push(Box::new(VBBReceiver::new(vbb_bus_endpoint)));
    }
    if to_path_profile {
        let path_profile_bus_endpoint = bus.add_rx();
        let path_profile_symbol_index = std::sync::Arc::clone(&symbol_index);
        receivers.push(Box::new(PathProfileReceiver::new(
            path_profile_bus_endpoint,
            path_profile_symbol_index,
        )));
    }

    let encoded_trace_path = static_cfg.encoded_trace.clone();
    let frontend_insn_index = std::sync::Arc::clone(&insn_index);
    let frontend_handle = thread::spawn(move || {
        frontend::decoder::decode_trace(
            encoded_trace_path,
            static_cfg.clone(),
            runtime_cfg.clone(),
            frontend_insn_index,
            bus,
        )
    });
    let receiver_handles: Vec<_> = receivers
        .into_iter()
        .map(|mut receiver| thread::spawn(move || receiver.try_receive_loop()))
        .collect();

    // Handle frontend thread
    match frontend_handle.join() {
        Ok(result) => result?,
        Err(e) => {
            // still join the receivers
            for handle in receiver_handles {
                handle.join().unwrap();
            }
            println!("frontend thread panicked: {:?}", e);
            return Err(anyhow::anyhow!("Frontend thread panicked: {:?}", e));
        }
    }

    // Handle receiver threads
    for (i, handle) in receiver_handles.into_iter().enumerate() {
        if let Err(e) = handle.join() {
            return Err(anyhow::anyhow!("Receiver thread {} panicked: {:?}", i, e));
        }
    }

    Ok(())
}
