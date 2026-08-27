use anyhow::Result;
use env_logger;
use gcno_reader::cfg::ControlFlowGraph;
use gcno_reader::gcno::str_trim;
use gcno_reader::reader::GCNOReader;

#[test]
fn test_gcno_read_sort() -> Result<()> {
    env_logger::init();
    // You might want to add a sample .gcno file in a tests/data directory
    let mut reader = GCNOReader::new(String::from("tests/data/x86-gcc13/sort.bin-sort.gcno"))?;
    println!("version: {:x}", reader.version());
    println!("stamp: {:x}", reader.stamp());
    println!("cwd: {:?}", reader.cwd());
    let gcno = reader.parse()?;
    println!("{:?}", gcno);
    let cfg = ControlFlowGraph::from(gcno);
    println!("{:?}", cfg);
    println!("--- instrumented edges ---");
    let edges = cfg.report_instrumented_edges();
    println!("{:?}", edges);
    Ok(())
}

#[test]
fn test_invalid_file() {
    let result = GCNOReader::new(String::from("nonexistent.gcno"));
    assert!(result.is_err());
}

#[test]
fn test_cut_string() {
    let s = String::from("hello\0");
    let cut = str_trim(s);
    assert_eq!(cut, "hello");
}
