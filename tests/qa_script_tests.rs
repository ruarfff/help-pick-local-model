use std::fs;
use std::os::unix::fs::PermissionsExt;

#[test]
fn qa_script_exists_and_checks_expected_cli_outputs() {
    let metadata = fs::metadata("scripts/qa.sh").expect("scripts/qa.sh should exist");
    assert!(
        metadata.permissions().mode() & 0o111 != 0,
        "scripts/qa.sh should be executable"
    );

    let script = fs::read_to_string("scripts/qa.sh").expect("scripts/qa.sh should be readable");
    for expected in [
        "cargo build",
        "--help",
        "--json",
        "--family gemma --top 5",
        "--explain mlx-community/gemma-4-12B-it-OptiQ-4bit",
        "gemma4_unified",
        "mlx_vlm.server",
    ] {
        assert!(
            script.contains(expected),
            "scripts/qa.sh should check {expected}"
        );
    }
}
