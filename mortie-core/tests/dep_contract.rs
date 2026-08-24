//! The issue #48 dependency-minimality contract, made mechanical.
//!
//! `mortie-core` is the crate external consumers (healpix-geo) depend on, so it
//! carries no non-std dependencies and never a moc-crate dependency. This test
//! line-scans the crate's own manifest and fails if any dependency table
//! appears, so adding one breaks `cargo test` instead of only breaking prose.
//!
//! Deliberately std-only: pulling in a TOML parser to guard a zero-dependency
//! contract would be the very thing it forbids.

use std::fs;
use std::path::Path;

/// The table names a dependency can only be declared under. Matched per
/// dot-separated segment, so `[dependencies.foo]` and
/// `[target.'cfg(unix)'.dependencies]` are caught along with `[dependencies]`.
const DEPENDENCY_TABLES: [&str; 3] = ["dependencies", "dev-dependencies", "build-dependencies"];

#[test]
fn manifest_declares_no_dependencies() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let text = fs::read_to_string(&manifest)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", manifest.display()));

    let offenders: Vec<&str> = text
        .lines()
        .filter(|line| is_dependency_table(line))
        .collect();

    assert!(
        offenders.is_empty(),
        "mortie-core must stay dependency-free (issue #48), but {} declares: {:?}",
        manifest.display(),
        offenders,
    );
}

/// True when `line` is a TOML table header whose path ends in a dependency table.
fn is_dependency_table(line: &str) -> bool {
    let header = line.split('#').next().unwrap_or("").trim();
    let Some(path) = header.strip_prefix('[').and_then(|h| h.strip_suffix(']')) else {
        return false;
    };
    path.split('.')
        .any(|segment| DEPENDENCY_TABLES.contains(&segment.trim().trim_matches('"')))
}

#[test]
fn the_scan_recognizes_the_forms_a_dependency_can_take() {
    // Guards the guard: these are the shapes that must trip it.
    for header in [
        "[dependencies]",
        "[dev-dependencies]",
        "[build-dependencies]",
        "[dependencies.moc]",
        "[target.'cfg(unix)'.dependencies]",
        "  [dependencies]  # with a trailing comment",
    ] {
        assert!(
            is_dependency_table(header),
            "should have tripped on {header:?}"
        );
    }

    for header in [
        "[package]",
        "# [dependencies]",
        "name = \"mortie-core\"",
        "",
    ] {
        assert!(
            !is_dependency_table(header),
            "should not have tripped on {header:?}"
        );
    }
}
