fn main() {
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
