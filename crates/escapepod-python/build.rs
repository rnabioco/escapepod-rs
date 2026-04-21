// On macOS, pyo3's `extension-module` feature intentionally does not link
// libpython — the interpreter supplies those symbols at dlopen time. Apple's
// linker needs `-undefined dynamic_lookup` to tolerate that. Maturin sets
// these flags automatically; plain `cargo build` does not, so we emit them
// here. Equivalent to `pyo3_build_config::add_extension_module_link_args()`.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
