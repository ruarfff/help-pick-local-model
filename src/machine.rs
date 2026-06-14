use anyhow::Result;
use serde::{Deserialize, Serialize};
use sysinfo::Disks;
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineInfo {
    pub os: Option<String>,
    pub arch: String,
    pub chip: Option<String>,
    pub total_memory_gb: f64,
    pub available_memory_gb: Option<f64>,
    pub estimated_llm_budget_gb: f64,
    pub free_disk_gb: Option<f64>,
    pub is_apple_silicon: bool,
}

pub async fn detect_machine() -> Result<MachineInfo> {
    let arch = match std::env::consts::ARCH {
        "aarch64" => "arm64",
        value => value,
    }
    .to_string();
    let total_memory_gb = sysctl_u64("hw.memsize")
        .await
        .map(bytes_to_gb)
        .unwrap_or_else(|| {
            let mut system = sysinfo::System::new_all();
            system.refresh_memory();
            bytes_to_gb(system.total_memory())
        });
    let mut system = sysinfo::System::new_all();
    system.refresh_memory();
    let available_memory_gb = Some(bytes_to_gb(system.available_memory()));
    let os = macos_version().await;
    let chip = sysctl_string("machdep.cpu.brand_string").await;
    let is_apple_silicon = arch == "aarch64"
        || arch == "arm64"
        || sysctl_string("hw.optional.arm64").await.as_deref() == Some("1");
    let free_disk_gb = Disks::new_with_refreshed_list()
        .iter()
        .find(|disk| disk.mount_point().to_string_lossy() == "/")
        .map(|disk| bytes_to_gb(disk.available_space()));
    let os_reserve_gb = 8.0_f64.max(total_memory_gb * 0.25);
    let estimated_llm_budget_gb = (total_memory_gb - os_reserve_gb).max(0.0);

    Ok(MachineInfo {
        os,
        arch,
        chip,
        total_memory_gb,
        available_memory_gb,
        estimated_llm_budget_gb,
        free_disk_gb,
        is_apple_silicon,
    })
}

fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

async fn sysctl_u64(name: &str) -> Option<u64> {
    sysctl_string(name).await?.trim().parse().ok()
}

async fn sysctl_string(name: &str) -> Option<String> {
    let output = Command::new("sysctl")
        .arg("-n")
        .arg(name)
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!value.is_empty()).then_some(value)
}

async fn macos_version() -> Option<String> {
    let output = Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Some(format!("macOS {version}"))
}
