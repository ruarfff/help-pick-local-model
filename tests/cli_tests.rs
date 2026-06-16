use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn cli_rejects_unknown_use_case() {
    let mut cmd = Command::cargo_bin("mlx-model-picker").unwrap();

    cmd.args(["--use-case", "spreadsheet"])
        .assert()
        .failure()
        .stderr(contains("unknown use case 'spreadsheet'"));
}

#[test]
fn help_lists_use_case_option() {
    let mut cmd = Command::cargo_bin("mlx-model-picker").unwrap();

    cmd.arg("--help")
        .assert()
        .success()
        .stdout(contains("--use-case <USE_CASE>"));
}
