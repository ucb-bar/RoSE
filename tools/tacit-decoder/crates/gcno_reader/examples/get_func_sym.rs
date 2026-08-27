use object::{Object, ObjectSymbol};
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <binary>", args[0]);
        std::process::exit(1);
    }

    let binary_path = &args[1];

    // Read the binary file
    let binary_data = match fs::read(binary_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading binary file: {}", e);
            std::process::exit(1);
        }
    };

    // Parse the binary file
    let obj_file = match object::File::parse(&*binary_data) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error parsing binary: {}", e);
            std::process::exit(1);
        }
    };

    // Iterate through symbols and print function addresses
    for symbol in obj_file.symbols() {
        if symbol.kind() == object::SymbolKind::Text {
            println!(
                "{:#x}: {}",
                symbol.address(),
                symbol.name().unwrap_or("<unknown>")
            );
        }
    }
}
