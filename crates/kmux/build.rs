use std::fs;
use std::path::Path;

fn main() {
    generate_tunnel_presets();
    embed_windows_manifest();
}

// Folded from the former `rmux-server` build script. The shipped tunnel presets
// (formerly `crates/rmux-server/tunnels/*.toml`, now `tunnels/*.toml` at this
// crate root) are baked into a generated `tunnel_presets.rs` that
// `src/server/web/tunnel/preset.rs` pulls in via
// `include!(concat!(env!("OUT_DIR"), "/tunnel_presets.rs"))`. Generated
// unconditionally (as upstream did): the include site is `web`-gated, so on a
// no-web build the file is simply never included.
fn generate_tunnel_presets() {
    let presets_dir = Path::new("tunnels");
    println!("cargo:rerun-if-changed=tunnels");

    let mut presets = Vec::new();
    if presets_dir.is_dir() {
        for entry in fs::read_dir(presets_dir).expect("read tunnel preset directory") {
            let path = entry.expect("read tunnel preset entry").path();
            if path
                .extension()
                .is_some_and(|extension| extension == "toml")
            {
                println!("cargo:rerun-if-changed={}", path.display());
                let name = path
                    .file_stem()
                    .expect("tunnel preset file has a stem")
                    .to_string_lossy()
                    .into_owned();
                let content = fs::read_to_string(&path).expect("read tunnel preset");
                presets.push((name, content));
            }
        }
    }
    presets.sort_by(|left, right| left.0.cmp(&right.0));

    let mut generated =
        String::from("pub(super) const SHIPPED_TUNNEL_PRESETS: &[(&str, &str)] = &[\n");
    for (name, content) in presets {
        generated.push_str(&format!("    ({name:?}, {content:?}),\n"));
    }
    generated.push_str("];\n");

    let out_path =
        Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR is set")).join("tunnel_presets.rs");
    fs::write(out_path, generated).expect("write tunnel preset include");
}

// The Windows LNK1327 fix (unchanged from the pre-collapse kmux build script).
fn embed_windows_manifest() {
    // Path is relative to this crate's root (CARGO_MANIFEST_DIR), which is also
    // the working directory Cargo sets for build scripts. Keeping it relative
    // (never an absolute literal) is what lets the tree be moved or cloned to a
    // different machine / Termux checkout without baking a host path into source.
    const WINDOWS_MANIFEST: &str = "resources/windows/kmux.exe.manifest";

    if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        // `embed-manifest` (MSVC path) emits `/MANIFESTINPUT:<canonicalized
        // absolute path>` as a cached `rustc-link-arg`. That absolute path is
        // recomputed from the *current* location every time this script runs —
        // correct as long as the script actually re-runs. The trap: Cargo caches
        // build-script output and only re-runs on the `rerun-if-*` triggers
        // below. A plain directory MOVE changes neither `build.rs` nor the
        // manifest file's contents, so without an explicit trigger Cargo happily
        // replays a stale `/MANIFESTINPUT:` pointing at the OLD location, and the
        // linker dies with LNK1327 (mt.exe: "cannot find the path specified").
        //
        // `rerun-if-env-changed=CARGO_MANIFEST_DIR` closes that gap: the crate's
        // absolute root changes on any move/clone, forcing a re-run that
        // regenerates the link arg against the new location.
        println!("cargo:rerun-if-env-changed=CARGO_MANIFEST_DIR");

        embed_manifest::embed_manifest_file(WINDOWS_MANIFEST)
            .expect("unable to embed Windows application manifest");
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={WINDOWS_MANIFEST}");
}
