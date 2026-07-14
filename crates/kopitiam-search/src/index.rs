use std::path::Path;

use anyhow::Result;
use kopitiam_ontology::{Entity, EntityId, EntityKind};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, STORED, STRING, Schema, TEXT, Value};
use tantivy::{Index, IndexReader, IndexWriter, TantivyDocument, doc};

/// One result from [`SearchIndex::search`], reconstructed from a
/// [`kopitiam_ontology::Entity`] previously indexed with
/// [`SearchIndex::add_entity`].
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub id: EntityId,
    pub name: String,
    pub kind: EntityKind,
    pub source: String,
    pub score: f32,
}

/// The tantivy field handles for [`SearchIndex`]'s fixed schema, resolved
/// once so every call site avoids repeated `schema.get_field` lookups.
struct Fields {
    id: Field,
    name: Field,
    kind: Field,
    source: Field,
}

fn build_schema() -> (Schema, Fields) {
    let mut builder = Schema::builder();
    let id = builder.add_text_field("id", STRING | STORED);
    let name = builder.add_text_field("name", TEXT | STORED);
    let kind = builder.add_text_field("kind", STRING | STORED);
    let source = builder.add_text_field("source", STRING | STORED);
    (builder.build(), Fields { id, name, kind, source })
}

/// A tantivy-backed search index over [`kopitiam_ontology::Entity`] records.
///
/// This is deliberately storage-agnostic in the same sense as
/// `kopitiam_knowledge::SemanticGraph`: it does not know about
/// `kopitiam-knowledge` or `kopitiam-index` and is fed entities one at a
/// time by whatever caller (today: tests; eventually: `kopitiam-workflow`'s
/// context builder) owns the graph. Only `name` is tokenized for full-text
/// matching — `id`, `kind`, and `source` are indexed but not stemmed/split,
/// since they are exact-match identifiers, not prose.
pub struct SearchIndex {
    index: Index,
    fields: Fields,
    writer: IndexWriter,
    reader: IndexReader,
}

impl SearchIndex {
    /// Creates a brand-new index on disk at `path`, which must already
    /// exist and be empty.
    pub fn create_in_dir(path: &Path) -> Result<Self> {
        let (schema, fields) = build_schema();
        let index = Index::create_in_dir(path, schema)?;
        Self::from_index(index, fields)
    }

    /// Opens a previously-created index at `path`.
    pub fn open_in_dir(path: &Path) -> Result<Self> {
        let index = Index::open_in_dir(path)?;
        let schema = index.schema();
        let fields = Fields {
            id: schema.get_field("id")?,
            name: schema.get_field("name")?,
            kind: schema.get_field("kind")?,
            source: schema.get_field("source")?,
        };
        Self::from_index(index, fields)
    }

    fn from_index(index: Index, fields: Fields) -> Result<Self> {
        let writer = index.writer(50_000_000)?;
        let reader = index.reader()?;
        Ok(Self { index, fields, writer, reader })
    }

    /// Queues `entity` for indexing. Call [`Self::commit`] to make it
    /// visible to [`Self::search`].
    pub fn add_entity(&mut self, entity: &Entity) -> Result<()> {
        self.writer.add_document(doc!(
            self.fields.id => entity.id.to_string(),
            self.fields.name => entity.name.clone(),
            self.fields.kind => kind_to_str(entity.kind),
            self.fields.source => entity.source.clone(),
        ))?;
        Ok(())
    }

    /// Flushes queued documents and makes them visible to subsequent
    /// searches.
    pub fn commit(&mut self) -> Result<()> {
        self.writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    /// Runs a full-text query against indexed entity names, returning up
    /// to `limit` hits ordered by relevance score (highest first).
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.fields.name]);
        let parsed_query = query_parser.parse_query(query)?;
        let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(limit).order_by_score())?;

        top_docs
            .into_iter()
            .map(|(score, address)| {
                let doc: TantivyDocument = searcher.doc(address)?;
                self.hit_from_doc(&doc, score)
            })
            .collect()
    }

    fn hit_from_doc(&self, doc: &TantivyDocument, score: f32) -> Result<SearchHit> {
        let text = |field: Field| -> Result<String> {
            doc.get_first(field)
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .ok_or_else(|| anyhow::anyhow!("indexed document missing a required field"))
        };

        let id: EntityId = text(self.fields.id)?.parse::<uuid::Uuid>()?.into();
        let kind = kind_from_str(&text(self.fields.kind)?)?;

        Ok(SearchHit { id, name: text(self.fields.name)?, kind, source: text(self.fields.source)?, score })
    }
}

/// Round-trips an [`EntityKind`] through its `serde` representation rather
/// than hand-matching variant names here, so this stays correct if
/// `kopitiam-ontology` adds a kind without anyone remembering to update a
/// parallel match arm in this crate.
fn kind_to_str(kind: EntityKind) -> String {
    match serde_json::to_value(kind).expect("EntityKind serializes to a JSON string") {
        serde_json::Value::String(s) => s,
        other => unreachable!("EntityKind must serialize to a string, got {other:?}"),
    }
}

fn kind_from_str(s: &str) -> Result<EntityKind> {
    Ok(serde_json::from_value(serde_json::Value::String(s.to_string()))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_and_finds_entities_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let mut index = SearchIndex::create_in_dir(dir.path()).unwrap();

        let a = Entity::new(EntityKind::Symbol, "parse_pdf", "rust-analyzer");
        let b = Entity::new(EntityKind::Symbol, "render_markdown", "rust-analyzer");
        let a_id = a.id;
        index.add_entity(&a).unwrap();
        index.add_entity(&b).unwrap();
        index.commit().unwrap();

        let hits = index.search("parse_pdf", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, a_id);
        assert_eq!(hits[0].name, "parse_pdf");
        assert_eq!(hits[0].kind, EntityKind::Symbol);
        assert_eq!(hits[0].source, "rust-analyzer");
    }

    #[test]
    fn respects_the_search_limit() {
        let dir = tempfile::tempdir().unwrap();
        let mut index = SearchIndex::create_in_dir(dir.path()).unwrap();
        for i in 0..5 {
            index.add_entity(&Entity::new(EntityKind::Symbol, format!("widget_{i}"), "test")).unwrap();
        }
        index.commit().unwrap();

        let hits = index.search("widget*", 2).unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn reopens_a_previously_created_index() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut index = SearchIndex::create_in_dir(dir.path()).unwrap();
            index.add_entity(&Entity::new(EntityKind::Artifact, "kopitiam-search", "test")).unwrap();
            index.commit().unwrap();
        }

        let index = SearchIndex::open_in_dir(dir.path()).unwrap();
        let hits = index.search("kopitiam-search", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, EntityKind::Artifact);
    }
}
