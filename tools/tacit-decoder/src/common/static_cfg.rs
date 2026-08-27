use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct DecoderStaticCfg {
    pub encoded_trace: String,
    pub application_binary_asid_tuples: Vec<(String, String)>,
    pub sbi_binary: String,
    pub kernel_binary: String,
    pub kernel_jump_label_patch_log: String,
    pub driver_binary_entry_tuples: Vec<(String, String)>,
    pub header_only: bool,
    pub to_stats: bool,
    pub to_txt: bool,
    pub to_stack_txt: bool,
    pub to_atomics: bool,
    pub to_afdo: bool,
    pub gcno: String,
    pub to_gcda: bool,
    pub to_speedscope: bool,
    pub to_perfetto: bool,
    pub to_vbb: bool,
    pub to_path_profile: bool,
}

pub fn load_file_config(path: &str) -> Result<DecoderStaticCfg> {
    let f = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(f);
    let cfg: DecoderStaticCfg = serde_json::from_reader(reader)?;
    Ok(cfg)
}
