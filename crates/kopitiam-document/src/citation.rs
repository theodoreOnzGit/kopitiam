/// A citation detected inside paragraph text, kept for provenance reporting.
/// Citations are never extracted out of or altered within their surrounding
/// paragraph text -- this only records that one was seen, and where.
#[derive(Debug, Clone, PartialEq)]
pub struct Citation {
    pub text: String,
}
