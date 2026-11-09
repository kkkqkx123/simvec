//! Opaque C handle types.

/// Engine handle as seen from C: an incomplete type with real size.
///
/// The `_dummy` member keeps the declaration valid C11 the same way the
/// linkrs handle types do; the field is never read.
#[repr(C)]
pub struct simvec_engine_t {
    _dummy: u8,
}
