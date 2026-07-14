#[derive(Debug, Clone, PartialEq)]
pub struct List {
    pub ordered: bool,
    pub items: Vec<String>,
}
