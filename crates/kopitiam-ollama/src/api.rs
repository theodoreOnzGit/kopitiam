//! The shared vocabulary -- messages, tools, tool calls, capabilities.
//!
//! **Upstream:** `api/types.go` and `types/model/{capability,config}.go`
//! (ollama, MIT).
//!
//! ## Why this module is small and boring on purpose
//!
//! Everything else in the port talks in these types. A renderer turns
//! [`Message`]s into a prompt; a parser turns model output back into
//! [`Message`]s and [`ToolCall`]s; the scheduler passes them around. So this is
//! the **frozen contract** -- if it churns, every consumer churns with it.
//! [`crate::options`] holds the *knobs*; this holds the *nouns*.
//!
//! ## The one subtlety worth reading before you touch anything: map ordering
//!
//! Upstream stores tool-call arguments and tool properties in an **insertion-
//! ordered** map (`orderedmap.Map`), not Go's built-in map. That is deliberate,
//! and the reason is fidelity to the model: when a model emits
//! `{"city": "Singapore", "unit": "celsius"}`, re-serialising it as
//! `{"unit": ..., "city": ...}` produces a different string, and anything that
//! hashes, caches, diffs, or replays that call now disagrees with the model's
//! own output. So [`ToolCallArguments`] preserves insertion order, via
//! `IndexMap`.
//!
//! **But template rendering deliberately drops that order.** Upstream's
//! `convertToolsForTemplate` calls `.ToMap()`, handing the template a plain Go
//! map -- and Go's `text/template` then ranges it in **sorted key order** (see
//! [`crate::gotmpl`]). So the same data is ordered on the wire and sorted in a
//! prompt. Both are correct, for different reasons: the wire must match the
//! model, the prompt must be reproducible. [`ToolCallArguments::to_template_value`]
//! is where the one becomes the other, and it is the only place that conversion
//! should happen.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::gotmpl::Value;

/// One message in a chat sequence. **Upstream:** `api.Message`.
///
/// Note [`Message::role`] is **lowercased on deserialise**, matching upstream's
/// custom `UnmarshalJSON`. A client sending `"User"` and one sending `"user"`
/// must not produce two different prompts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Message {
    #[serde(deserialize_with = "lowercase_role")]
    pub role: String,

    pub content: String,

    /// What was inside the model's thinking tags, once
    /// [`crate::thinking`] has separated it out. Only populated when the
    /// request asked for thinking.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub thinking: String,

    /// Base64/raw image payloads. Carried but not yet used by the port's
    /// renderers -- see the note in [`crate::template`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,

    /// Which tool this message is the *result* of (`role == "tool"`).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tool_name: String,

    /// Which specific call this message answers. This is why `tool` messages
    /// are never merged by `collate` -- merging would destroy the pairing.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tool_call_id: String,
}

fn lowercase_role<'de, D>(d: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(d)?;
    Ok(s.to_lowercase())
}

impl Message {
    /// The common two-field case.
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_lowercase(),
            content: content.to_string(),
            ..Default::default()
        }
    }

    /// Shape this message the way a template sees it.
    ///
    /// **Upstream:** `convertMessagesForTemplate` -> `templateMessage`. The
    /// field names are the **Go exported** names (`Role`, `Content`, ...)
    /// because that is what a chat template writes -- `{{ .Content }}`, not
    /// `{{ .content }}`. Renaming them to snake_case would break every real
    /// template on earth.
    pub fn to_template_value(&self) -> Value {
        let mut m = std::collections::BTreeMap::new();
        m.insert("Role".into(), Value::Str(self.role.clone()));
        m.insert("Content".into(), Value::Str(self.content.clone()));
        m.insert("Thinking".into(), Value::Str(self.thinking.clone()));
        m.insert("ToolName".into(), Value::Str(self.tool_name.clone()));
        m.insert("ToolCallID".into(), Value::Str(self.tool_call_id.clone()));
        m.insert(
            "ToolCalls".into(),
            Value::List(self.tool_calls.iter().map(ToolCall::to_template_value).collect()),
        );
        Value::Map(m)
    }
}

/// A tool invocation the model asked for. **Upstream:** `api.ToolCall`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub id: String,
    pub function: ToolCallFunction,
}

impl ToolCall {
    pub fn to_template_value(&self) -> Value {
        let mut m = std::collections::BTreeMap::new();
        m.insert("ID".into(), Value::Str(self.id.clone()));
        m.insert("Function".into(), self.function.to_template_value());
        Value::Map(m)
    }
}

/// **Upstream:** `api.ToolCallFunction`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolCallFunction {
    /// Position in a parallel-tool-call batch. Streamed deltas use it to say
    /// which call a fragment belongs to.
    #[serde(default)]
    pub index: usize,
    pub name: String,
    #[serde(default)]
    pub arguments: ToolCallArguments,
}

impl ToolCallFunction {
    pub fn to_template_value(&self) -> Value {
        let mut m = std::collections::BTreeMap::new();
        m.insert("Index".into(), Value::Int(self.index as i64));
        m.insert("Name".into(), Value::Str(self.name.clone()));
        m.insert("Arguments".into(), self.arguments.to_template_value());
        Value::Map(m)
    }
}

/// Tool-call arguments, **in the order the model emitted them**.
///
/// **Upstream:** `api.ToolCallFunctionArguments`, backed by
/// `orderedmap.Map[string, any]`. See the module docs for why the order is
/// load-bearing rather than fussy.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolCallArguments(pub IndexMap<String, serde_json::Value>);

impl ToolCallArguments {
    pub fn new() -> Self {
        Self(IndexMap::new())
    }

    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.0.get(key)
    }

    /// Insert, preserving insertion order. Re-setting an existing key keeps its
    /// **original** position, which is `IndexMap`'s behaviour and upstream's.
    pub fn set(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.0.insert(key.into(), value);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// **Upstream:** `(*ToolCallFunctionArguments).String()` -- and note it
    /// returns `"{}"` for the nil case, never `"null"`. A template printing
    /// `{{ .Function.Arguments }}` must emit valid JSON object syntax even when
    /// there are no arguments, or the model sees a malformed call.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(&self.0).unwrap_or_else(|_| "{}".to_string())
    }

    /// Hand these arguments to a template -- **order is dropped here, on
    /// purpose.** See the module docs.
    pub fn to_template_value(&self) -> Value {
        Value::Map(
            self.0
                .iter()
                .map(|(k, v)| (k.clone(), Value::from_json(v)))
                .collect(),
        )
    }
}

/// A tool the caller is offering the model. **Upstream:** `api.Tool`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<serde_json::Value>,
    pub function: ToolFunction,
}

impl Tool {
    /// **Upstream:** `convertToolsForTemplate` -> `templateTool`.
    pub fn to_template_value(&self) -> Value {
        let mut m = std::collections::BTreeMap::new();
        m.insert("Type".into(), Value::Str(self.tool_type.clone()));
        m.insert(
            "Items".into(),
            self.items.as_ref().map(Value::from_json).unwrap_or(Value::Nil),
        );
        m.insert("Function".into(), self.function.to_template_value());
        Value::Map(m)
    }
}

/// **Upstream:** `api.ToolFunction`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default)]
    pub parameters: ToolFunctionParameters,
}

impl ToolFunction {
    pub fn to_template_value(&self) -> Value {
        let mut m = std::collections::BTreeMap::new();
        m.insert("Name".into(), Value::Str(self.name.clone()));
        m.insert("Description".into(), Value::Str(self.description.clone()));
        m.insert("Parameters".into(), self.parameters.to_template_value());
        Value::Map(m)
    }
}

/// A tool's JSON-Schema parameter block. **Upstream:** `api.ToolFunctionParameters`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolFunctionParameters {
    #[serde(rename = "type", default)]
    pub param_type: String,
    #[serde(rename = "$defs", default, skip_serializing_if = "Option::is_none")]
    pub defs: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,

    /// Insertion-ordered, same reasoning as [`ToolCallArguments`].
    ///
    /// **`Option` because Go's is a POINTER, and the three states differ on the
    /// wire.** Upstream's field is `*ToolPropertiesMap`, so:
    ///
    /// | Go state | marshals as | means |
    /// |---|---|---|
    /// | nil pointer | `"properties":null` | nobody said what this tool takes |
    /// | set, empty | `"properties":{}` | this tool takes **no** arguments |
    /// | set, populated | `{"city":{...}}` | these arguments |
    ///
    /// Collapsing the first two loses a real distinction. A no-argument tool
    /// emitted as `null` reads to a model as *"I don't know this tool's
    /// arguments"* rather than *"it has none"*, and some families are sensitive
    /// to that. This was `bd-iut`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub properties: Option<IndexMap<String, ToolProperty>>,
}

impl ToolFunctionParameters {
    /// Walk the declared properties, treating **absent and empty alike**.
    ///
    /// Most callers want this: the nil-vs-empty distinction only matters when
    /// *emitting* JSON (see [`ToolFunctionParameters::properties`]); when you
    /// are iterating, "none declared" and "none present" are the same job.
    pub fn properties_iter(&self) -> impl Iterator<Item = (&String, &ToolProperty)> {
        self.properties.iter().flatten()
    }

    /// Is there at least one declared property? Absent and empty both say no.
    pub fn has_properties(&self) -> bool {
        self.properties.as_ref().is_some_and(|p| !p.is_empty())
    }

    /// Look one property up by name.
    pub fn property(&self, name: &str) -> Option<&ToolProperty> {
        self.properties.as_ref()?.get(name)
    }

    pub fn to_template_value(&self) -> Value {
        let mut m = std::collections::BTreeMap::new();
        m.insert("Type".into(), Value::Str(self.param_type.clone()));
        m.insert(
            "Required".into(),
            Value::List(self.required.iter().map(|s| Value::Str(s.clone())).collect()),
        );
        // Sorted for the template -- see the module docs. An absent map and an
        // empty one both render as an empty (falsy) map here, which is right:
        // `{{ if .Properties }}` must not fire for either, and a template has no
        // way to act on the difference anyway. The distinction only matters on
        // the wire, which is where it is preserved.
        m.insert(
            "Properties".into(),
            Value::Map(
                self.properties
                    .iter()
                    .flatten()
                    .map(|(k, v)| (k.clone(), v.to_template_value()))
                    .collect(),
            ),
        );
        Value::Map(m)
    }
}

/// One parameter's schema. **Upstream:** `api.ToolProperty`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolProperty {
    #[serde(rename = "anyOf", default, skip_serializing_if = "Vec::is_empty")]
    pub any_of: Vec<ToolProperty>,
    #[serde(rename = "type", default, skip_serializing_if = "PropertyType::is_empty")]
    pub prop_type: PropertyType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(rename = "enum", default, skip_serializing_if = "Vec::is_empty")]
    pub enum_values: Vec<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub properties: Option<IndexMap<String, ToolProperty>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,
}

impl ToolProperty {
    /// Render this property as a TypeScript type.
    ///
    /// **Upstream:** `(ToolProperty).ToTypeScriptType()`, exposed to templates as
    /// the `toTypeScriptType` function.
    ///
    /// Why TypeScript, of all things: several model families (Qwen3-Coder among
    /// them) were fine-tuned with tool schemas written as TS signatures rather
    /// than raw JSON Schema, because it is far more compact in tokens. So this
    /// is not a convenience -- it is part of the prompt those models expect.
    pub fn to_typescript_type(&self) -> String {
        if !self.any_of.is_empty() {
            return self
                .any_of
                .iter()
                .map(ToolProperty::to_typescript_type)
                .collect::<Vec<_>>()
                .join(" | ");
        }
        if self.prop_type.0.is_empty() {
            return "any".to_string();
        }
        self.prop_type
            .0
            .iter()
            .map(|t| map_to_typescript_type(t))
            .collect::<Vec<_>>()
            .join(" | ")
    }

    pub fn to_template_value(&self) -> Value {
        let mut m = std::collections::BTreeMap::new();
        m.insert("Description".into(), Value::Str(self.description.clone()));
        m.insert(
            "Type".into(),
            Value::Str(self.prop_type.0.join(" | ")),
        );
        m.insert(
            "Enum".into(),
            Value::List(self.enum_values.iter().map(Value::from_json).collect()),
        );
        Value::Map(m)
    }
}

/// **Upstream:** `mapToTypeScriptType`. JSON Schema type name -> TS type name.
fn map_to_typescript_type(json_type: &str) -> &'static str {
    match json_type {
        "string" => "string",
        "number" | "integer" => "number",
        "boolean" => "boolean",
        "array" => "any[]",
        "object" => "Record<string, any>",
        "null" => "null",
        _ => "any",
    }
}

/// A JSON-Schema `type`, which may be one string or a list of them.
///
/// **Upstream:** `api.PropertyType` with its custom `UnmarshalJSON` -- it
/// accepts `"string"` and `["string","null"]` alike, because both spellings
/// appear in schemas in the wild.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(transparent)]
pub struct PropertyType(pub Vec<String>);

impl PropertyType {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<'de> Deserialize<'de> for PropertyType {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum OneOrMany {
            One(String),
            Many(Vec<String>),
        }
        Ok(match OneOrMany::deserialize(d)? {
            OneOrMany::One(s) => PropertyType(vec![s]),
            OneOrMany::Many(v) => PropertyType(v),
        })
    }
}

/// Whether -- and how hard -- the model should think before answering.
///
/// **Upstream:** `api.ThinkValue`, whose `Value` is `any` holding either a bool
/// or one of four level strings.
///
/// Modelled as a real enum here rather than a dynamic value, because Rust can:
/// the upstream type exists only because Go has no sum type. The **three-state**
/// nature survives, and it matters -- `None` ("caller never mentioned
/// thinking") is genuinely distinct from `Some(Bool(false))` ("caller explicitly
/// turned it off"), and templates can see the difference via `IsThinkSet`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ThinkValue {
    Bool(bool),
    Level(ThinkLevel),
}

/// **Upstream:** the four accepted strings in `ThinkValue.IsValid()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkLevel {
    Low,
    Medium,
    High,
    Max,
}

impl ThinkLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            ThinkLevel::Low => "low",
            ThinkLevel::Medium => "medium",
            ThinkLevel::High => "high",
            ThinkLevel::Max => "max",
        }
    }
}

impl ThinkValue {
    /// Is thinking enabled at all? **Upstream:** `(*ThinkValue).Bool()`.
    ///
    /// **Any** level string counts as enabled — upstream's own comment.
    pub fn enabled(&self) -> bool {
        match self {
            ThinkValue::Bool(b) => *b,
            ThinkValue::Level(_) => true,
        }
    }

    /// The level as a string. **Upstream:** `(*ThinkValue).String()`.
    ///
    /// Watch this one: a bare `true` reports **`"medium"`**, not `"true"` and
    /// not `""` — upstream's *"Default level when just true"*. A template
    /// branching on `.ThinkLevel` therefore sees a real level even when the
    /// caller only said `true`. Getting this wrong sends a reasoning model an
    /// empty level and it falls back to its own default, silently.
    pub fn level(&self) -> &'static str {
        match self {
            ThinkValue::Bool(true) => "medium",
            ThinkValue::Bool(false) => "",
            ThinkValue::Level(l) => l.as_str(),
        }
    }
}

/// What a model can do. **Upstream:** `types/model/capability.go`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Capability {
    Completion,
    Tools,
    /// Fill-in-the-middle: the model accepts a prompt **and** a suffix.
    Insert,
    Vision,
    Embedding,
    Thinking,
    Image,
    Audio,
}

impl Capability {
    pub fn as_str(&self) -> &'static str {
        match self {
            Capability::Completion => "completion",
            Capability::Tools => "tools",
            Capability::Insert => "insert",
            Capability::Vision => "vision",
            Capability::Embedding => "embedding",
            Capability::Thinking => "thinking",
            Capability::Image => "image",
            Capability::Audio => "audio",
        }
    }
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A model's config blob, as stored in its manifest.
///
/// **Upstream:** `types/model/config.go` `ConfigV2`.
///
/// This is the JSON that the manifest's `config` layer points at. `renderer`
/// and `parser` are the fields that select a model-family-specific chat
/// renderer / response parser -- i.e. this struct is how a pulled model tells
/// the runtime "do not use the generic template path for me".
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ConfigV2 {
    #[serde(default)]
    pub model_format: String,
    #[serde(default)]
    pub model_family: String,
    #[serde(default)]
    pub model_families: Vec<String>,
    /// Shown as "Parameter Size" in `show`.
    #[serde(default)]
    pub model_type: String,
    /// Shown as "Quantization Level" in `show`.
    #[serde(default)]
    pub file_type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub renderer: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub parser: String,
    /// Minimum runtime version, from a Modelfile `REQUIRES`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub requires: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub remote_host: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub remote_model: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub context_length: i64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub embedding_length: i64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub base_name: String,
    /// Required by the OCI image spec.
    #[serde(default)]
    pub architecture: String,
    #[serde(default)]
    pub os: String,
    #[serde(default)]
    pub rootfs: RootFs,
}

fn is_zero(v: &i64) -> bool {
    *v == 0
}

/// **Upstream:** `types/model/config.go` `RootFS`. Required by the OCI spec even
/// though ollama does not use it for anything.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RootFs {
    #[serde(rename = "type", default)]
    pub fs_type: String,
    #[serde(default)]
    pub diff_ids: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_role_is_lowercased_on_deserialise() {
        let m: Message = serde_json::from_value(json!({"role": "User", "content": "hi"})).unwrap();
        assert_eq!(m.role, "user");
    }

    /// The ordering property, asserted rather than trusted. Re-serialising a
    /// tool call in a different key order would break anything that hashes,
    /// caches, or replays what the model actually said.
    #[test]
    fn tool_call_arguments_keep_the_models_own_key_order() {
        let raw = r#"{"city":"Singapore","unit":"celsius","days":3}"#;
        let args: ToolCallArguments = serde_json::from_str(raw).unwrap();
        assert_eq!(args.to_json_string(), raw, "order must survive a round trip");
        assert_eq!(
            args.0.keys().collect::<Vec<_>>(),
            vec!["city", "unit", "days"]
        );
    }

    /// ...and the deliberate opposite: a template sees them SORTED, because a
    /// prompt must be reproducible.
    #[test]
    fn arguments_are_sorted_once_they_reach_a_template() {
        let args: ToolCallArguments =
            serde_json::from_str(r#"{"zeta":1,"alpha":2}"#).unwrap();
        let Value::Map(m) = args.to_template_value() else {
            panic!("expected a map")
        };
        assert_eq!(m.keys().collect::<Vec<_>>(), vec!["alpha", "zeta"]);
    }

    #[test]
    fn empty_arguments_render_as_an_object_not_null() {
        assert_eq!(ToolCallArguments::new().to_json_string(), "{}");
    }

    /// **`bd-iut`.** Go's `Properties` is a pointer, so three states are
    /// distinguishable on the wire, and the middle one carries real meaning: a
    /// tool that takes NO arguments must not be indistinguishable from a tool
    /// whose arguments nobody described.
    #[test]
    fn properties_keeps_gos_three_way_nil_empty_populated_distinction() {
        let mut p = ToolFunctionParameters {
            param_type: "object".into(),
            ..Default::default()
        };

        // Absent -> null.
        assert_eq!(p.properties, None);
        let v = serde_json::to_value(&p).unwrap();
        assert!(
            !v.as_object().unwrap().contains_key("properties"),
            "absent must not emit a properties key at all"
        );
        assert!(!p.has_properties());

        // Set but empty -> a real, present, empty map.
        p.properties = Some(IndexMap::new());
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["properties"], json!({}), "empty must survive as {{}}");
        assert!(!p.has_properties(), "empty still means nothing to iterate");
        assert_eq!(p.properties_iter().count(), 0);

        // Populated.
        let mut m = IndexMap::new();
        m.insert("city".to_string(), ToolProperty::default());
        p.properties = Some(m);
        assert!(p.has_properties());
        assert_eq!(p.properties_iter().count(), 1);
        assert!(p.property("city").is_some());
        assert!(p.property("nope").is_none());
    }

    /// The round trip both ways, since a client sending `null` and one omitting
    /// the key must land in the same state.
    #[test]
    fn absent_and_explicit_null_properties_both_deserialise_to_none() {
        let omitted: ToolFunctionParameters =
            serde_json::from_value(json!({"type": "object"})).unwrap();
        let explicit: ToolFunctionParameters =
            serde_json::from_value(json!({"type": "object", "properties": null})).unwrap();
        assert_eq!(omitted.properties, None);
        assert_eq!(explicit.properties, None);

        let empty: ToolFunctionParameters =
            serde_json::from_value(json!({"type": "object", "properties": {}})).unwrap();
        assert_eq!(
            empty.properties,
            Some(IndexMap::new()),
            "an explicit {{}} is NOT the same as absent"
        );
    }

    #[test]
    fn property_type_accepts_a_string_or_a_list() {
        let one: ToolProperty = serde_json::from_value(json!({"type": "string"})).unwrap();
        assert_eq!(one.prop_type.0, vec!["string"]);
        let many: ToolProperty =
            serde_json::from_value(json!({"type": ["string", "null"]})).unwrap();
        assert_eq!(many.prop_type.0, vec!["string", "null"]);
    }

    #[test]
    fn typescript_types_match_the_upstream_mapping() {
        let p = |t: serde_json::Value| -> ToolProperty {
            serde_json::from_value(json!({"type": t})).unwrap()
        };
        assert_eq!(p(json!("string")).to_typescript_type(), "string");
        assert_eq!(p(json!("integer")).to_typescript_type(), "number");
        assert_eq!(p(json!("number")).to_typescript_type(), "number");
        assert_eq!(p(json!("boolean")).to_typescript_type(), "boolean");
        assert_eq!(p(json!("array")).to_typescript_type(), "any[]");
        assert_eq!(p(json!("object")).to_typescript_type(), "Record<string, any>");
        assert_eq!(p(json!("null")).to_typescript_type(), "null");
        assert_eq!(p(json!("weird")).to_typescript_type(), "any");
        assert_eq!(
            p(json!(["string", "null"])).to_typescript_type(),
            "string | null"
        );
        // No type at all -> any.
        assert_eq!(ToolProperty::default().to_typescript_type(), "any");
    }

    #[test]
    fn any_of_unions_its_branches() {
        let p: ToolProperty = serde_json::from_value(json!({
            "anyOf": [{"type": "string"}, {"type": "integer"}]
        }))
        .unwrap();
        assert_eq!(p.to_typescript_type(), "string | number");
    }

    /// The trap: a bare `true` must report level "medium", not "" and not
    /// "true". A reasoning model given an empty level silently falls back to
    /// its own default.
    #[test]
    fn a_bare_true_think_value_reports_medium() {
        assert_eq!(ThinkValue::Bool(true).level(), "medium");
        assert_eq!(ThinkValue::Bool(false).level(), "");
        assert_eq!(ThinkValue::Level(ThinkLevel::High).level(), "high");
    }

    #[test]
    fn any_think_level_counts_as_enabled() {
        assert!(ThinkValue::Bool(true).enabled());
        assert!(!ThinkValue::Bool(false).enabled());
        for l in [ThinkLevel::Low, ThinkLevel::Medium, ThinkLevel::High, ThinkLevel::Max] {
            assert!(ThinkValue::Level(l).enabled(), "{l:?}");
        }
    }

    #[test]
    fn think_value_parses_both_json_shapes() {
        assert_eq!(
            serde_json::from_value::<ThinkValue>(json!(true)).unwrap(),
            ThinkValue::Bool(true)
        );
        assert_eq!(
            serde_json::from_value::<ThinkValue>(json!("high")).unwrap(),
            ThinkValue::Level(ThinkLevel::High)
        );
        // An unknown level is rejected, matching upstream's validation.
        assert!(serde_json::from_value::<ThinkValue>(json!("blazing")).is_err());
    }

    #[test]
    fn capabilities_round_trip_through_their_wire_names() {
        for (c, s) in [
            (Capability::Completion, "completion"),
            (Capability::Tools, "tools"),
            (Capability::Insert, "insert"),
            (Capability::Vision, "vision"),
            (Capability::Embedding, "embedding"),
            (Capability::Thinking, "thinking"),
            (Capability::Image, "image"),
            (Capability::Audio, "audio"),
        ] {
            assert_eq!(c.as_str(), s);
            assert_eq!(serde_json::to_value(c).unwrap(), json!(s));
        }
    }

    #[test]
    fn a_message_renders_its_go_field_names_for_templates() {
        let mut m = Message::new("assistant", "hi");
        m.thinking = "hmm".into();
        let Value::Map(v) = m.to_template_value() else {
            panic!()
        };
        // Go-exported names, because that is what templates write.
        for k in ["Role", "Content", "Thinking", "ToolCalls", "ToolName", "ToolCallID"] {
            assert!(v.contains_key(k), "missing {k}");
        }
    }

    #[test]
    fn a_config_blob_round_trips() {
        let raw = json!({
            "model_format": "gguf",
            "model_family": "qwen3",
            "model_families": ["qwen3"],
            "model_type": "0.6B",
            "file_type": "Q4_K_M",
            "renderer": "qwen3-coder",
            "parser": "qwen3-coder",
            "architecture": "amd64",
            "os": "linux",
            "rootfs": {"type": "layers", "diff_ids": []}
        });
        let c: ConfigV2 = serde_json::from_value(raw).unwrap();
        assert_eq!(c.renderer, "qwen3-coder");
        assert_eq!(c.model_type, "0.6B");
        assert_eq!(c.file_type, "Q4_K_M");
    }
}
