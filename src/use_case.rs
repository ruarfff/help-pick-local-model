use serde::{Deserialize, Serialize};

use crate::{parser::ParsedModel, runtime::RuntimeDecision, runtime::RuntimeKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UseCaseId {
    #[serde(rename = "coding")]
    Coding,
    #[serde(rename = "openclaw")]
    OpenClaw,
    #[serde(rename = "chat")]
    Chat,
    #[serde(rename = "vision")]
    Vision,
}

impl UseCaseId {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "coding" | "coding-agent" | "code" => Ok(Self::Coding),
            "openclaw" | "open-claw" => Ok(Self::OpenClaw),
            "chat" | "assistant" | "general" => Ok(Self::Chat),
            "vision" | "vlm" | "multimodal" => Ok(Self::Vision),
            unknown => Err(format!(
                "unknown use case '{unknown}'. Expected one of: coding, openclaw, chat, vision"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Coding => "coding",
            Self::OpenClaw => "openclaw",
            Self::Chat => "chat",
            Self::Vision => "vision",
        }
    }
}

impl std::fmt::Display for UseCaseId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UseCaseProfile {
    id: UseCaseId,
    description: &'static str,
}

impl UseCaseProfile {
    pub fn for_id(id: UseCaseId) -> Self {
        let description = match id {
            UseCaseId::Coding => "local coding assistant / coding agent",
            UseCaseId::OpenClaw => "local OpenClaw-style agent runtime",
            UseCaseId::Chat => "general local assistant",
            UseCaseId::Vision => "multimodal and image workflows",
        };
        Self { id, description }
    }

    pub fn id(self) -> UseCaseId {
        self.id
    }

    pub fn description(self) -> &'static str {
        self.description
    }

    pub fn output_note(self) -> &'static str {
        match self.id {
            UseCaseId::OpenClaw => {
                "OpenClaw note: Start the local server, then configure OpenClaw to use http://localhost:8080/v1."
            }
            UseCaseId::Vision => {
                "Vision note: Multimodal models generally need mlx-vlm and may not load with mlx-lm."
            }
            _ => "",
        }
    }

    pub fn score_adjustments(
        self,
        parsed: &ParsedModel,
        runtime: &RuntimeDecision,
    ) -> ProfileScore {
        let mut score = ProfileScore::default();
        match self.id {
            UseCaseId::Coding => {
                if parsed.is_coding {
                    score.add(14.0, "coding profile: coder marker");
                    score.reason("matches coding profile");
                }
                if parsed.is_instruction_tuned {
                    score.add(6.0, "coding profile: instruction tuned");
                }
            }
            UseCaseId::OpenClaw => {
                if parsed.is_coding {
                    score.add(12.0, "openclaw profile: coding-capable model");
                    score.reason("matches openclaw profile");
                }
                if parsed.is_instruction_tuned {
                    score.add(8.0, "openclaw profile: instruction tuned");
                }
                match runtime.inferred_runtime.or(runtime.runtime) {
                    Some(RuntimeKind::MlxLm) => {
                        score.add(10.0, "openclaw profile: text runtime compatible");
                    }
                    Some(RuntimeKind::MlxVlm) => {
                        score.add(
                            -25.0,
                            "openclaw profile: VLM-only model needs mlx-vlm, not mlx-lm",
                        );
                        score.warning(
                            "OpenClaw works best with a text model served through mlx-lm; this model needs mlx-vlm.",
                        );
                    }
                    None => {
                        score.add(-10.0, "openclaw profile: unknown serving runtime");
                    }
                }
            }
            UseCaseId::Chat => {
                if parsed.is_instruction_tuned {
                    score.add(12.0, "chat profile: instruction/chat tuned");
                    score.reason("matches chat profile");
                }
                if parsed.is_coding {
                    score.add(-2.0, "chat profile: coding specialization is not required");
                }
            }
            UseCaseId::Vision => match runtime.inferred_runtime {
                Some(RuntimeKind::MlxVlm) => {
                    score.add(36.0, "vision profile: mlx-vlm compatible");
                    score.reason("matches vision profile");
                }
                Some(RuntimeKind::MlxLm) => {
                    score.add(-24.0, "vision profile: text-only runtime");
                    score.warning("Vision use cases usually need mlx-vlm-compatible models.");
                }
                None => {
                    score.add(-10.0, "vision profile: unknown multimodal support");
                    if runtime.runtime == Some(RuntimeKind::MlxVlm) {
                        score.warning(
                            "Forced mlx-vlm runtime does not prove this model is multimodal-compatible.",
                        );
                    }
                }
            },
        }
        score
    }
}

#[derive(Debug, Default)]
pub struct ProfileScore {
    pub items: Vec<(&'static str, f64)>,
    pub reasons: Vec<&'static str>,
    pub warnings: Vec<&'static str>,
}

impl ProfileScore {
    fn add(&mut self, value: f64, label: &'static str) {
        self.items.push((label, value));
    }

    fn reason(&mut self, reason: &'static str) {
        if !self.reasons.contains(&reason) {
            self.reasons.push(reason);
        }
    }

    fn warning(&mut self, warning: &'static str) {
        if !self.warnings.contains(&warning) {
            self.warnings.push(warning);
        }
    }
}
