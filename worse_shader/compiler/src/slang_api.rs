//! The generated API escape hatch for options and separate debug artifacts.
//! Keep raw descriptors and their backing storage together at the FFI boundary.

use arrayvec::ArrayVec;
use slang_auto::{self as slang, sys};

use crate::{CompileDesc, DebugFlags, MatrixLayout, log_diagnostics};

pub(super) fn create_session(global: &slang::GlobalSession, desc: &CompileDesc) -> slang::Result<slang::Session> {
    // SAFETY: the global session is live and the name is a static C string.
    let capability = unsafe {
        sys::slang_rs_slang_IGlobalSession_findCapability(global.as_raw().as_ptr(), c"spvDescriptorHeapEXT".as_ptr())
    };
    if capability == sys::SlangCapabilityID_SLANG_CAPABILITY_UNKNOWN {
        return Err(slang::Error::InvalidArgument(
            "Slang SDK does not support spvDescriptorHeapEXT",
        ));
    }
    let profile = global.find_profile("spirv_1_5")?;
    let (debug_level, debug_flags) = desc.debug_info.map_or(
        (
            sys::SlangDebugInfoLevel_SLANG_DEBUG_INFO_LEVEL_NONE,
            DebugFlags::empty(),
        ),
        |debug| (debug.level.into(), debug.flags),
    );
    let options = [
        int_option(sys::slang_CompilerOptionName_Capability, capability),
        int_option(
            sys::slang_CompilerOptionName_Optimization,
            sys::SlangOptimizationLevel::from(desc.optimization) as i32,
        ),
        int_option(sys::slang_CompilerOptionName_DebugInformation, debug_level as i32),
        int_option(sys::slang_CompilerOptionName_EmitSpirvDirectly, 1),
        int_option(
            sys::slang_CompilerOptionName_EmitSeparateDebug,
            i32::from(debug_flags.contains(DebugFlags::SEPARATE)),
        ),
        int_option(sys::slang_CompilerOptionName_VulkanUseEntryPointName, 1),
    ];
    let mut targets = ArrayVec::<_, 2>::new();
    targets.push(sys::slang_TargetDesc {
        format: sys::SlangCompileTarget_SLANG_SPIRV,
        profile,
        ..Default::default()
    });
    if debug_flags.contains(DebugFlags::EMIT_ASM) {
        targets.push(sys::slang_TargetDesc {
            format: sys::SlangCompileTarget_SLANG_SPIRV_ASM,
            profile,
            ..Default::default()
        });
    }
    let session_desc = sys::slang_SessionDesc {
        targets: targets.as_ptr(),
        targetCount: targets.len() as sys::SlangInt,
        defaultMatrixLayoutMode: match desc.matrix_layout {
            MatrixLayout::RowMajor => sys::SlangMatrixLayoutMode_SLANG_MATRIX_LAYOUT_ROW_MAJOR,
            MatrixLayout::ColumnMajor => sys::SlangMatrixLayoutMode_SLANG_MATRIX_LAYOUT_COLUMN_MAJOR,
        },
        compilerOptionEntries: options.as_ptr(),
        compilerOptionEntryCount: options.len() as u32,
        ..Default::default()
    };
    // SAFETY: the two fixed-size arrays outlive this call, counts match their
    // lengths, and options contain only integers. Slang copies the descriptors
    // during createSession; no callbacks or borrowed interfaces are supplied.
    unsafe { global.create_session_raw(&session_desc, targets.len()) }
}

fn int_option(name: sys::slang_CompilerOptionName, value: i32) -> sys::slang_CompilerOptionEntry {
    sys::slang_CompilerOptionEntry {
        name,
        value: sys::slang_CompilerOptionValue {
            kind: sys::slang_CompilerOptionValueKind_Int,
            intValue0: value,
            ..Default::default()
        },
    }
}

pub(super) fn separate_debug_data(program: &slang::Component) -> slang::Result<Option<slang::Blob>> {
    let extension = program.as_raw().query::<sys::slang_IComponentType2>()?;
    // SAFETY: our composite has one entry point and target zero is SPIR-V.
    // Slang transfers owned result/diagnostic references; call_owned retains
    // the component and releases all outputs, including on failure.
    let result = log_diagnostics(unsafe {
        extension.call_owned(|ptr, out, diag| {
            sys::slang_rs_slang_IComponentType2_getEntryPointCompileResult(ptr, 0, 0, out, diag)
        })
    }?);
    // SAFETY: the compile-result interface is live for this call.
    if unsafe { sys::slang_rs_slang_ICompileResult_getItemCount(result.as_ptr()) } < 2 {
        return Ok(None);
    }
    // Slang's separate-debug result stores base SPIR-V at 0 and debug SPIR-V at 1.
    // SAFETY: index 1 exists and getItemData transfers an owned immutable blob.
    let debug = log_diagnostics(unsafe {
        result.call_owned(|ptr, out, _| sys::slang_rs_slang_ICompileResult_getItemData(ptr, 1, out))
    }?);
    // SAFETY: the owned blob and its retained compile result keep its data alive.
    Ok(Some(unsafe { slang::Blob::from_com(debug) }))
}
