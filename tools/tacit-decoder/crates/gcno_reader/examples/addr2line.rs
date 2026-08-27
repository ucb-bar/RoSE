use addr2line::Loader;
use std::env;

fn main() {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <hex_address>", args[0]);
        std::process::exit(1);
    }

    // Parse hex address from argument
    let addr = u64::from_str_radix(&args[1].trim_start_matches("0x"), 16)
        .expect("Invalid hexadecimal address");

    let loader = Loader::new(
        "/scratch/iansseijelly/ltrace_decoder/crates/gcno_reader/tests/data/x86-gcc13/sort.bin",
    )
    .unwrap();
    let symbol: Option<addr2line::Location> = loader.find_location(addr).unwrap();
    if let Some(loc) = symbol {
        if let Some(file) = loc.file {
            println!("File: {}, Line: {}", file, loc.line.unwrap_or(0));
        }
    }
}
