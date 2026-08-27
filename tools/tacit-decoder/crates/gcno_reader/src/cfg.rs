use crate::{gcno::Gcno, tag};
use indexmap::IndexMap as OrderedMap;
use log::debug;
use std::fs::File;
use std::io::{BufWriter, Write};
// New CFG data structures

const GCDA_MAGIC: u32 = 0x67636461; // "gcda"

#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    pub cwd: String,
    pub version: u32,
    pub stamp: u32,
    pub functions: Vec<FunctionCFG>,
}

#[derive(Debug, Clone)]
pub struct FunctionCFG {
    // Function metadata
    pub identifier: u32,
    pub lineno_checksum: u32,
    pub cfg_checksum: u32,
    pub name: String,
    pub source_file: String,
    pub span: SourceSpan,
    // CFG structure
    pub basic_blocks: OrderedMap<u32, BasicBlock>,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Clone)]
pub struct SourceSpan {
    pub start_line: u32,
    pub start_column: u32,
    pub end_line: u32,
    pub end_column: u32,
}

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: u32,
    pub locations: Vec<SourceLocation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    pub file: String,
    pub lines: u32,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: u32,
    pub to: u32,
    pub edge_flags: u32,
}

// Conversion implementation
impl From<Gcno> for ControlFlowGraph {
    fn from(gcno: Gcno) -> Self {
        let functions = gcno
            .functions
            .into_iter()
            .filter_map(|func| {
                // Only convert functions that have blocks
                func.blocks.map(|blocks| {
                    let mut basic_blocks = OrderedMap::new();
                    let mut edges = Vec::new();

                    // Convert blocks and their lines
                    for arcs in blocks.arcs {
                        basic_blocks.insert(
                            arcs.src_block,
                            BasicBlock {
                                id: arcs.src_block,
                                locations: Vec::new(),
                            },
                        );

                        // Convert arcs to edges
                        for arc in arcs.arcs {
                            edges.push(Edge {
                                from: arc.src_block,
                                to: arc.dst_block,
                                edge_flags: arc.flags,
                            });
                        }
                    }

                    // insert a special bb with id = 1
                    basic_blocks.insert(
                        1,
                        BasicBlock {
                            id: 1,
                            locations: Vec::new(),
                        },
                    );

                    for line in blocks.lines {
                        for source in line.sources {
                            for lineno in source.lineno {
                                basic_blocks
                                    .get_mut(&line.block_id)
                                    .unwrap()
                                    .locations
                                    .push(SourceLocation {
                                        file: source.file_name.clone(),
                                        lines: lineno,
                                    });
                            }
                        }
                    }

                    // add special location annotation for the special bb
                    // 0 is always the entry point
                    basic_blocks
                        .get_mut(&0)
                        .unwrap()
                        .locations
                        .push(SourceLocation {
                            file: func.source.clone(),
                            lines: func.start_lineno,
                        });

                    // 1 is always the exit point
                    basic_blocks
                        .get_mut(&1)
                        .unwrap()
                        .locations
                        .push(SourceLocation {
                            file: func.source.clone(),
                            lines: func.end_lineno,
                        });

                    FunctionCFG {
                        identifier: func.identifier,
                        lineno_checksum: func.lineno_checksum,
                        cfg_checksum: func.cfg_checksum,
                        name: func.name,
                        source_file: func.source,
                        span: SourceSpan {
                            start_line: func.start_lineno,
                            start_column: func.start_colno,
                            end_line: func.end_lineno,
                            end_column: func.end_colno,
                        },
                        basic_blocks,
                        edges,
                    }
                })
            })
            .collect();

        ControlFlowGraph {
            cwd: gcno.cwd,
            version: gcno.version,
            stamp: gcno.stamp,
            functions,
        }
    }
}

// report source line information instead of block id
#[derive(Debug, Clone)]
pub struct ReportedEdge {
    pub from: Vec<SourceLocation>,
    pub to: Vec<SourceLocation>,
    pub count: u64,
    pub entry: bool, // whether this is the entry edge
    pub func_name: String,
}

impl ReportedEdge {
    fn new(edge: Edge, func: &FunctionCFG) -> Self {
        Self {
            from: func.basic_blocks.get(&edge.from).unwrap().locations.clone(),
            to: func.basic_blocks.get(&edge.to).unwrap().locations.clone(),
            count: 0,
            entry: edge.from == 0 && edge.to == 2,
            func_name: func.name.clone(),
        }
    }

    pub fn increment_count(&mut self) {
        // debug!("incremented count for edge: from {:?}", self.count);
        self.count += 1;
        // debug!("incremented count for edge: to {:?}", self.count);
    }
}

impl ControlFlowGraph {
    pub fn report_instrumented_edges(&self) -> OrderedMap<String, Vec<ReportedEdge>> {
        let mut edge_map: OrderedMap<String, Vec<ReportedEdge>> = OrderedMap::new();
        for func in self.functions.iter() {
            let edges = func.report_instrumented_edges();
            edge_map.insert(func.name.clone(), edges);
        }
        edge_map
    }

    pub fn write_gcda(
        &self,
        edge_map: &OrderedMap<String, Vec<ReportedEdge>>,
        writer: &mut BufWriter<File>,
    ) {
        writer.write_all(&GCDA_MAGIC.to_le_bytes()).unwrap();
        writer.write_all(&self.version.to_le_bytes()).unwrap();
        writer.write_all(&self.stamp.to_le_bytes()).unwrap();
        writer.write_all(&0u32.to_le_bytes()).unwrap(); // checksum
                                                        // write the function summary tag
        for func in self.functions.iter() {
            writer.write_all(&tag::FUNCTION_TAG.to_le_bytes()).unwrap();
            writer.write_all(&12u32.to_le_bytes()).unwrap();
            writer
                .write_all(&(func.identifier as u32).to_le_bytes())
                .unwrap();
            writer
                .write_all(&(func.lineno_checksum as u32).to_le_bytes())
                .unwrap();
            writer
                .write_all(&(func.cfg_checksum as u32).to_le_bytes())
                .unwrap();
            writer
                .write_all(&tag::COUNTER_BASE_TAG.to_le_bytes())
                .unwrap();
            let size = edge_map.get(&func.name).unwrap().len() as u32;
            writer.write_all(&(size * 8).to_le_bytes()).unwrap();
            for edge in edge_map.get(&func.name).unwrap() {
                debug!("writing edge count: {:?}", edge.count);
                writer.write_all(&edge.count.to_le_bytes()).unwrap();
            }
        }
    }
}

impl FunctionCFG {
    pub fn report_instrumented_edges(&self) -> Vec<ReportedEdge> {
        let mut edges = Vec::new();
        for edge in self.edges.iter() {
            if edge.edge_flags & tag::FLAG_TREE == 0 {
                edges.push(ReportedEdge::new(edge.clone(), &self));
            }
        }
        edges
    }
}

impl SourceLocation {
    pub fn from_addr2line(loc: Option<addr2line::Location>) -> Self {
        if let Some(loc) = loc {
            if let Some(file) = loc.file {
                SourceLocation {
                    file: file.to_string(),
                    lines: loc.line.unwrap_or(0),
                }
            } else {
                SourceLocation {
                    file: String::new(),
                    lines: 0,
                }
            }
        } else {
            SourceLocation {
                file: String::new(),
                lines: 0,
            }
        }
    }
}
