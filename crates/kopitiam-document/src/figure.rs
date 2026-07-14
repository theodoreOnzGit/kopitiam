use std::path::PathBuf;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Figure {
    pub caption: Option<String>,
    pub image_path: Option<PathBuf>,
}
