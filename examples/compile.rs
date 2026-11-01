use std::{ffi::CString, path::Path};

use worse_shader_preprocess as shader;

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets");
    shader::preprocess(string_to_cstring_lossy(&dir.to_string_lossy()), "pre_depth.slang");
}

fn string_to_cstring_lossy(s: &str) -> CString {
    CString::new(s.replace('\0', "")).unwrap()
}

