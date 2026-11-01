use std::ffi::CString;

use slang_rs as slang;

pub fn preprocess(search_dir: CString, module_name: &str) {
    let global_session = slang::GlobalSession::new().unwrap();
    println!("{}", global_session.build_tag_string());

    let session_options = slang::CompilerOptions::default()
        .optimization(slang::OptimizationLevel::None)
        .matrix_layout_column(true);

    let targets = [slang::TargetDesc::default()
        .format(slang::CompileTarget::SpirvAsm)
        .profile(global_session.find_profile("spirv_1_5"))];
    let search_paths = [search_dir.as_ptr()];

    let session_desc = slang::SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_paths)
        .options(&session_options);

    let session = global_session.create_session(&session_desc).unwrap();
    let module = session.load_module(module_name).unwrap();
    let entry_point = module.find_entry_point_by_name("vs_main").unwrap();

    let program = session
        .create_composite_component_type(&[module.into(), entry_point.into()])
        .unwrap();

    let linked_program = program.link().unwrap();

    let reflection = linked_program.layout(0).unwrap();

    let shader = linked_program.entry_point_code(0, 0).unwrap();
    // println!("{}", shader.as_str().unwrap());
}
