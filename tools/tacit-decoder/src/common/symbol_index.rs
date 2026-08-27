use addr2line::Loader;
use anyhow::Result;
use log::{debug, warn};
use object::elf::SHF_EXECINSTR;
use object::{Object, ObjectSection, ObjectSymbol, SectionFlags};
use std::collections::{BTreeMap, HashMap};
use std::fs;

use crate::common::prv::Prv;
use crate::common::source_location::SourceLocation;
use crate::common::static_cfg::DecoderStaticCfg;

// everything you need to know about a symbol
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub name: String,
    pub src: SourceLocation,
    pub prv: Prv,
    pub ctx: u64,
}

pub struct SymbolIndex {
    // use BTreeMap as symbol maps need to be ordered
    u_symbol_map: HashMap<u64, BTreeMap<u64, SymbolInfo>>,
    k_symbol_map: BTreeMap<u64, SymbolInfo>,
    m_symbol_map: BTreeMap<u64, SymbolInfo>,
}

impl SymbolIndex {
    pub fn get(&self, prv: Prv, ctx: u64) -> &BTreeMap<u64, SymbolInfo> {
        match prv {
            Prv::PrvUser => &self.u_symbol_map[&ctx],
            Prv::PrvSupervisor => &self.k_symbol_map,
            Prv::PrvMachine => &self.m_symbol_map,
            _ => panic!("Unsupported privilege level: {:?}", prv),
        }
    }

    pub fn get_user_symbol_map(&self) -> &HashMap<u64, BTreeMap<u64, SymbolInfo>> {
        &self.u_symbol_map
    }

    pub fn get_kernel_symbol_map(&self) -> &BTreeMap<u64, SymbolInfo> {
        &self.k_symbol_map
    }

    pub fn get_machine_symbol_map(&self) -> &BTreeMap<u64, SymbolInfo> {
        &self.m_symbol_map
    }

    /// Return the half-open address range `[start, end)` for the function that
    /// begins at `addr`. The end is the start of the next symbol in that
    /// privilege space, or `u64::MAX` if this is the last symbol.
    pub fn range(&self, prv: Prv, ctx: u64, addr: u64) -> Option<(u64, u64)> {
        let map = self.get(prv, ctx);

        if !map.contains_key(&addr) {
            return None;
        }

        // Find the symbol that starts at `addr` (or the nearest one <= addr)
        let (&start, _) = map.range(..=addr).next_back()?;

        // End is the next symbol start, if any
        let end = map
            .range((start + 1)..)
            .next()
            .map(|(&next_start, _)| next_start)
            .unwrap_or(u64::MAX);

        Some((start, end))
    }
}
pub fn build_single_symbol_index(
    elf_path: String,
    prv: Prv,
    offset: u64,
    ctx: u64,
) -> Result<BTreeMap<u64, SymbolInfo>> {
    // open application elf for symbol processing
    let elf_data = fs::read(&elf_path)?;
    let obj_file = object::File::parse(&*elf_data)?;
    debug!("elf_path: {}", elf_path);
    let loader = Loader::new(&elf_path)
        .map_err(|e| anyhow::Error::msg("loader error: ".to_string() + &e.to_string()))?;

    // Gather indices of all executable sections
    let exec_secs: std::collections::HashSet<_> = obj_file
        .sections()
        .filter_map(|sec| {
            if let SectionFlags::Elf { sh_flags } = sec.flags() {
                if sh_flags & (SHF_EXECINSTR as u64) != 0 {
                    return Some(sec.index());
                }
            }
            None
        })
        .collect();
    // Build func_symbol_map from _all_ symbols in executable sections
    let mut func_symbol_map: BTreeMap<u64, SymbolInfo> = BTreeMap::new();
    for symbol in obj_file.symbols() {
        // only symbols tied to an exec section
        if let Some(sec_idx) = symbol.section_index() {
            if exec_secs.contains(&sec_idx) {
                if let Ok(name) = symbol.name() {
                    // filter out ghost symbols
                    if !name.starts_with("$x") && !name.starts_with("$d") && !name.starts_with(".L") {
                        let addr = symbol.address();
                        // lookup source location (may return None)
                        let loc = loader.find_location(addr);
                        let mut info: SymbolInfo = SymbolInfo {
                            name: name.to_string(),
                            src: SourceLocation {
                                file: String::new(),
                                lines: 0,
                                prv: prv,
                            },
                            prv: prv,
                            ctx: ctx,
                        };
                        if let Ok(Some(loc)) = loc {
                            let src: SourceLocation = SourceLocation::from_addr2line(loc, prv);
                            info.src = src;
                        }
                        // dedupe aliases: prefer non‑empty over empty
                        if let Some(existing) = func_symbol_map.get_mut(&addr) {
                            if existing.name.trim().is_empty() && !info.name.trim().is_empty() {
                                *existing = info;
                            } else {
                                // warn!(
                                //     "func_addr 0x{:x} already in map as `{}`, ignoring alias `{}`",
                                //     addr, existing.name, info.name
                                // );
                            }
                        } else {
                            func_symbol_map.insert(addr + offset, info);
                        }
                    }
                }
            }
        }
    }
    debug!("func_symbol_map size: {}", func_symbol_map.len());
    Ok(func_symbol_map)
}

pub fn build_symbol_index(cfg: DecoderStaticCfg) -> Result<SymbolIndex> {
    let mut u_symbol_maps = HashMap::new();
    for (binary, asid) in cfg.application_binary_asid_tuples.clone() {
        let u_func_symbol_map =
            build_single_symbol_index(binary.clone(), Prv::PrvUser, 0, asid.parse::<u64>()?)?;
        u_symbol_maps.insert(asid.parse::<u64>()?, u_func_symbol_map);
        debug!("u_symbol_maps size: {}", u_symbol_maps.len());
    }
    let mut k_func_symbol_map = BTreeMap::new();
    if cfg.kernel_binary != "" {
        k_func_symbol_map =
            build_single_symbol_index(cfg.kernel_binary.clone(), Prv::PrvSupervisor, 0, 0)?;
        debug!("k_func_symbol_map size: {}", k_func_symbol_map.len());
        for (binary, entry) in cfg.driver_binary_entry_tuples {
            let driver_entry_point = u64::from_str_radix(entry.trim_start_matches("0x"), 16)?;
            let func_symbol_map = build_single_symbol_index(
                binary.clone(),
                Prv::PrvSupervisor,
                driver_entry_point,
                0,
            )?;
            k_func_symbol_map.extend(func_symbol_map);
            debug!("k_func_symbol_map size: {}", k_func_symbol_map.len());
        }
    }
    let m_func_symbol_map =
        build_single_symbol_index(cfg.sbi_binary.clone(), Prv::PrvMachine, 0, 0)?;
    Ok(SymbolIndex {
        u_symbol_map: u_symbol_maps,
        k_symbol_map: k_func_symbol_map,
        m_symbol_map: m_func_symbol_map,
    })
}
