//! Regression: `vendor/ggml/VERSION` broke every macOS build. ggml puts `vendor/ggml` on its
//! include path, and on a case-insensitive filesystem `#include <version>` found our VERSION
//! file instead of the C++ standard header. Catch that on Linux CI too.
use std::path::Path;

/// Extension-less C++ standard headers a stray file could shadow.
const STD_HEADERS: &[&str] = &[
    "algorithm", "any", "array", "atomic", "bitset", "chrono", "cmath", "complex", "condition_variable", "cstdint",
    "cstdio", "cstdlib", "cstring", "deque", "exception", "filesystem", "format", "fstream", "functional", "future",
    "iomanip", "ios", "iostream", "iterator", "limits", "list", "map", "memory", "mutex", "new", "numeric", "optional",
    "queue", "random", "ranges", "regex", "set", "span", "sstream", "stack", "stdexcept", "string", "string_view",
    "thread", "tuple", "type_traits", "unordered_map", "unordered_set", "utility", "variant", "vector", "version",
];

#[test]
fn no_file_in_ggml_include_roots_shadows_a_std_header() {
    let ggml = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/ggml");
    let mut bad = Vec::new();
    for dir in [ggml.clone(), ggml.join("include"), ggml.join("src")] {
        for e in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
            let name = e.unwrap().file_name().to_string_lossy().to_lowercase();
            if STD_HEADERS.contains(&name.as_str()) {
                bad.push(dir.join(&name).display().to_string());
            }
        }
    }
    assert!(bad.is_empty(), "these shadow C++ standard headers on case-insensitive filesystems: {bad:?}");
}
