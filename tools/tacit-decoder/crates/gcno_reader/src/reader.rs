use crate::gcno::*;
use crate::tag::*;
use anyhow::Result;
use log::trace;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
pub struct GCNOReader {
    reader: BufReader<File>,
    version: u32,
    stamp: u32,
    cwd: String,
}

/*
  int32:  byte3 byte2 byte1 byte0 | byte0 byte1 byte2 byte3
    int64:  int32:low int32:high
    string: int32:0 | int32:length char* char:0
    item: int32 | int64 | string
*/

fn read_u32(reader: &mut BufReader<File>) -> Result<u32> {
    let pos = reader.stream_position()?;
    trace!("{:08x}:", pos);
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let val = u32::from_le_bytes(buf);
    trace!("  {:08x}", val);
    Ok(val)
}

fn read_u32_from_slice(reader: &mut BufReader<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

// string: int32:0 | int32:length char* char:0
fn read_string(reader: &mut BufReader<File>) -> Result<String> {
    let pos = reader.stream_position()?;
    trace!("{:08x}: reading string", pos);
    let string_magic = read_u32(reader)?;
    assert!(
        string_magic == 0,
        "string magic should be 0 but got {:x}",
        string_magic
    );
    let length = read_u32(reader)?;
    let mut buf = vec![0u8; length as usize];
    reader.read_exact(&mut buf)?;
    read_u32(reader)?; // I don't why but this is there somehow
    Ok(str_trim(String::from_utf8(buf)?))
}

fn read_string_from_slice(reader: &mut BufReader<&[u8]>) -> Result<String> {
    let length = read_u32_from_slice(reader)?;
    let mut buf = vec![0u8; length as usize];
    reader.read_exact(&mut buf)?;
    // read_u32_from_slice(reader)?; // I don't why but this is there somehow
    Ok(str_trim(String::from_utf8(buf)?))
}

/*
  record: header data
    header: int32:tag int32:length
    data: item*
*/
fn read_record(reader: &mut BufReader<File>) -> Result<(u32, u32, Vec<u8>)> {
    let pos = reader.stream_position()?;
    trace!("{:08x}: reading record", pos);
    // let magic = read_u32(reader)?;
    // assert!(magic == 1, "Invalid record magic: {:x}", magic);
    let tag = read_u32(reader)?;
    let length = read_u32(reader)?;
    trace!("         tag={:08x}, length={:08x}", tag, length);
    match tag {
        FUNCTION_TAG => trace!("[RECORD]function"),
        BLOCKS_TAG => trace!("[RECORD]blocks"),
        ARCS_TAG => trace!("[RECORD]arcs"),
        LINES_TAG => trace!("[RECORD]lines"),
        COUNTER_BASE_TAG => trace!("[RECORD]counter base"),
        OBJECT_SUMMARY_TAG => trace!("[RECORD]object summary"),
        PROGRAM_SUMMARY_TAG => trace!("[RECORD]program summary"),
        _ => panic!("unknown tag: {:?}", tag),
    }
    let mut data = vec![0u8; length as usize];
    reader.read_exact(&mut data)?;
    Ok((tag, length, data))
}

fn parse_function(data: Vec<u8>, _length: u32) -> Result<Function> {
    let mut reader = BufReader::new(data.as_slice());
    let identifier = read_u32_from_slice(&mut reader)?;
    let lineno_checksum = read_u32_from_slice(&mut reader)?;
    let cfg_checksum = read_u32_from_slice(&mut reader)?;
    let name = read_string_from_slice(&mut reader)?;
    read_u32_from_slice(&mut reader)?; // I don't why but this is there somehow
    let source = read_string_from_slice(&mut reader)?;
    let start_lineno = read_u32_from_slice(&mut reader)?;
    let start_colno = read_u32_from_slice(&mut reader)?;
    let end_lineno = read_u32_from_slice(&mut reader)?;
    let end_colno = read_u32_from_slice(&mut reader)?;
    Ok(Function::new(
        identifier,
        lineno_checksum,
        cfg_checksum,
        name,
        source,
        start_lineno,
        start_colno,
        end_lineno,
        end_colno,
    ))
}

fn parse_blocks(data: Vec<u8>, length: u32) -> Result<Blocks> {
    assert!(
        length == 4,
        "blocks length should be 4 but got {:x}",
        length
    );
    let mut reader = BufReader::new(data.as_slice());
    let num_blocks = read_u32_from_slice(&mut reader)?;
    // assert there's no more data
    Ok(Blocks::new(num_blocks))
}

fn parse_arcs(data: Vec<u8>, length: u32) -> Result<Arcs> {
    assert!(
        length % 8 == 4,
        "arcs length should be 4 + multiple of 8 but got {:x}",
        length
    );
    let mut reader = BufReader::new(data.as_slice());
    let src_block = read_u32_from_slice(&mut reader)?;
    let num_arcs = (length - 4) / 8;
    let mut arcs = Vec::with_capacity(num_arcs as usize);
    for _ in 0..num_arcs {
        let dst_block = read_u32_from_slice(&mut reader)?;
        let flags = read_u32_from_slice(&mut reader)?;
        arcs.push(Arc {
            src_block,
            dst_block,
            flags,
        });
    }
    Ok(Arcs::new(num_arcs, src_block, arcs))
}

fn parse_lines(data: Vec<u8>, _length: u32) -> Result<Lines> {
    let mut reader = BufReader::new(data.as_slice());
    let block_id = read_u32_from_slice(&mut reader)?;
    let mut sources = Vec::new();
    let mut current_source_file = String::new(); // Track current source file
    let mut source_lineno = Vec::new();

    loop {
        match read_u32_from_slice(&mut reader) {
            Ok(lineno) => {
                if lineno == 0 {
                    let new_source_file = read_string_from_slice(&mut reader)?;
                    if !source_lineno.is_empty() {
                        sources.push(Source {
                            file_name: current_source_file.clone(),
                            lineno: source_lineno,
                        });
                    }
                    current_source_file = new_source_file;
                    source_lineno = Vec::new();
                } else {
                    source_lineno.push(lineno);
                }
            }
            Err(e) => {
                if e.downcast_ref::<std::io::Error>()
                    .map_or(false, |e| e.kind() == std::io::ErrorKind::UnexpectedEof)
                {
                    break;
                }
                return Err(e.into());
            }
        }
    }
    Ok(Lines { block_id, sources })
}

impl GCNOReader {
    pub fn new(path: String) -> Result<Self> {
        trace!("opening path: {:?}", path);
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let magic = read_u32(&mut reader)?;
        assert!(magic == 0x67636e6f); // GCNO magic number, only support little-endian
        let version = read_u32(&mut reader)?;
        let stamp = read_u32(&mut reader)?;
        trace!("version: {:x}, stamp: { }", version, stamp);
        let cwd = read_string(&mut reader)?;
        trace!("cwd: {:?}", cwd);
        Ok(Self {
            reader,
            version,
            stamp,
            cwd,
        })
    }

    pub fn parse(&mut self) -> Result<Gcno> {
        let mut gcno = Gcno::new(self.version, self.stamp, self.cwd.clone());
        let mut current_function: Option<Function> = None;
        loop {
            match read_record(&mut self.reader) {
                Ok((tag, length, data)) => match tag {
                    FUNCTION_TAG => {
                        let function = parse_function(data, length)?;
                        if let Some(cf) = current_function {
                            gcno.add_function(cf.clone());
                        }
                        current_function = Some(function);
                    }
                    BLOCKS_TAG => {
                        let blocks = parse_blocks(data, length)?;
                        current_function.as_mut().unwrap().set_blocks(blocks);
                    }
                    ARCS_TAG => {
                        let arcs = parse_arcs(data, length)?;
                        current_function
                            .as_mut()
                            .unwrap()
                            .blocks
                            .as_mut()
                            .unwrap()
                            .add_arcs(arcs);
                    }
                    LINES_TAG => {
                        let lines = parse_lines(data, length)?;
                        current_function
                            .as_mut()
                            .unwrap()
                            .blocks
                            .as_mut()
                            .unwrap()
                            .add_line(lines);
                    }
                    _ => (),
                },
                Err(e) => {
                    if e.downcast_ref::<std::io::Error>()
                        .map_or(false, |e| e.kind() == std::io::ErrorKind::UnexpectedEof)
                    {
                        if let Some(cf) = current_function {
                            gcno.add_function(cf.clone());
                        }
                        break;
                    }
                    return Err(e.into());
                }
            }
        }
        Ok(gcno)
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn stamp(&self) -> u32 {
        self.stamp
    }

    pub fn cwd(&self) -> &str {
        &self.cwd
    }
}
