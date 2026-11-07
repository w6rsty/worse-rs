use std::{env, fs, path::PathBuf};

fn main() {
    install_slang_dll();
}

fn install_slang_dll() {
    println!("cargo:rerun-if-env-changed=SLANG_DIR");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let Some(slang_dir) = env::var_os("SLANG_DIR") else {
        return;
    };
    let slang_bin = PathBuf::from(slang_dir).join("bin");
    let entries =
        fs::read_dir(&slang_bin).unwrap_or_else(|error| panic!("failed to read {}: {error}", slang_bin.display()));

    let mut dlls = Vec::new();
    for entry in entries {
        let entry = entry.unwrap_or_else(|error| panic!("failed to read Slang DLL entry: {error}"));
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("dll"))
            && path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("slang"))
        {
            println!("cargo:rerun-if-changed={}", path.display());
            dlls.push(path);
        }
    }

    if dlls.is_empty() {
        panic!("no slang*.dll files found in {}", slang_bin.display());
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("could not determine Cargo profile output directory");
    let examples_dir = profile_dir.join("examples");
    fs::create_dir_all(&examples_dir)
        .unwrap_or_else(|error| panic!("failed to create {}: {error}", examples_dir.display()));

    for source in dlls {
        let destination = examples_dir.join(source.file_name().expect("DLL path has no filename"));
        fs::copy(&source, &destination).unwrap_or_else(|error| {
            panic!(
                "failed to copy {} to {}: {error}",
                source.display(),
                destination.display()
            )
        });
    }
}
