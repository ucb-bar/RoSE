use std::fmt;
// remove the null terminator from the end of a string
pub fn str_trim(s: String) -> String {
    s.split('\0').next().unwrap_or("").to_string()
}

pub fn str_term(s: String) -> String {
    let mut s = s.clone();
    s.push('\0');
    s
}

#[derive(Debug, Clone)]
pub struct Gcno {
    pub version: u32,
    pub stamp: u32,
    pub cwd: String,
    pub functions: Vec<Function>,
}

impl Gcno {
    pub fn new(version: u32, stamp: u32, cwd: String) -> Self {
        Self {
            version,
            stamp,
            cwd,
            functions: Vec::new(),
        }
    }

    pub fn add_function(&mut self, function: Function) {
        self.functions.push(function);
    }
}

#[derive(Clone)]
pub struct Function {
    pub identifier: u32,
    pub lineno_checksum: u32,
    pub cfg_checksum: u32,
    pub name: String,
    pub source: String,
    pub start_lineno: u32,
    pub start_colno: u32,
    pub end_lineno: u32,
    pub end_colno: u32,
    pub blocks: Option<Blocks>,
}
impl fmt::Debug for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Function {{")?;
        writeln!(f, "    identifier: {:?},", self.identifier)?;
        writeln!(f, "    lineno_checksum: {:#x},", self.lineno_checksum)?;
        writeln!(f, "    cfg_checksum: {:#x},", self.cfg_checksum)?;
        writeln!(f, "    name: {:?},", self.name)?;
        writeln!(f, "    source: {:?},", self.source)?;
        writeln!(f, "    start: {}:{},", self.start_lineno, self.start_colno)?;
        writeln!(f, "    end: {}:{},", self.end_lineno, self.end_colno)?;
        writeln!(f, "    blocks: {:?}", self.blocks)?;
        write!(f, "}}")
    }
}

impl Function {
    pub fn new(
        identifier: u32,
        lineno_checksum: u32,
        cfg_checksum: u32,
        name: String,
        source: String,
        start_lineno: u32,
        start_colno: u32,
        end_lineno: u32,
        end_colno: u32,
    ) -> Self {
        Self {
            identifier,
            lineno_checksum,
            cfg_checksum,
            name,
            source,
            start_lineno,
            start_colno,
            end_lineno,
            end_colno,
            blocks: None,
        }
    }

    pub fn set_blocks(&mut self, blocks: Blocks) {
        self.blocks = Some(blocks);
    }
}

#[derive(Debug, Clone)]
pub struct Blocks {
    pub num_blocks: u32,
    pub arcs: Vec<Arcs>,
    pub lines: Vec<Lines>,
}

impl Blocks {
    pub fn new(num_blocks: u32) -> Self {
        Self {
            num_blocks,
            arcs: Vec::new(),
            lines: Vec::new(),
        }
    }

    pub fn add_arcs(&mut self, arcs: Arcs) {
        self.arcs.push(arcs);
    }

    pub fn add_line(&mut self, line: Lines) {
        self.lines.push(line);
    }
}

#[derive(Clone)]
pub struct Arcs {
    pub num_arcs: u32,
    pub src_block: u32,
    pub arcs: Vec<Arc>,
}

impl Arcs {
    pub fn new(num_arcs: u32, src_block: u32, arcs: Vec<Arc>) -> Self {
        Self {
            num_arcs,
            src_block,
            arcs,
        }
    }
}

impl fmt::Debug for Arcs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Arcs {{")?;
        writeln!(f, "    src_block: {},", self.src_block)?;
        writeln!(f, "    arcs: [")?;
        for arc in &self.arcs {
            writeln!(
                f,
                "        ({} -> {}, flags: {}),",
                arc.src_block, arc.dst_block, arc.flags
            )?;
        }
        writeln!(f, "    ]")?;
        write!(f, "}}")
    }
}

#[derive(Debug, Clone)]
pub struct Arc {
    pub src_block: u32,
    pub dst_block: u32,
    pub flags: u32,
}

#[derive(Clone)]
pub struct Lines {
    pub block_id: u32,
    pub sources: Vec<Source>,
}

impl fmt::Debug for Lines {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Lines {{")?;
        writeln!(f, "    block_id: {},", self.block_id)?;
        writeln!(f, "    sources: [")?;
        for source in &self.sources {
            writeln!(f, "        {:?},", source)?;
        }
        write!(f, "}}")
    }
}

#[derive(Debug, Clone)]
pub struct Source {
    pub file_name: String,
    pub lineno: Vec<u32>,
}
