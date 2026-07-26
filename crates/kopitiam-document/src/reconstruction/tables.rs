use super::Line;
use crate::Table;

#[cfg(test)]
use super::Cell;

const MIN_TABLE_ROWS: usize = 2;
const MIN_TABLE_COLUMNS: usize = 2;
const COLUMN_X_TOLERANCE: f32 = 8.0;

/// Detects a table as a run of consecutive lines that split into the same
/// number of geometric cells at matching x-positions. Falls through to
/// paragraph handling on anything ambiguous, rather than guessing.
///
/// # Longest valid prefix, not all-or-nothing
///
/// The header (row 0) fixes the column count and the reference x-positions.
/// A following line joins the table only if it has *exactly* that many cells,
/// each starting within `COLUMN_X_TOLERANCE` of the header's cell in the same
/// column. This walks the run and stops at the **first** line that breaks
/// uniformity, emitting the leading uniform rows as the table and reporting
/// how many lines it consumed so `build_blocks` resumes on the remainder.
///
/// This used to require *every* line in the candidate run to be uniform: a
/// single ragged row -- a merged cell, a wrapped cell, a subtotal line --
/// made the whole run return `None`, so the entire table fell through to
/// `consume_paragraph` and came out as one `Paragraph` per cell joined by
/// `\n\n` (`kopitiam_token_max.md` §6 card I-E; the 97-line one-cell-per-line
/// table in §3). Cutting the table at the last uniform row instead keeps the
/// aligned rows as a real table and lets the ragged remainder continue
/// through normal block handling, where it becomes whatever it actually is.
///
/// The ragged row is deliberately **not** padded to header width: padding
/// would have to emit extra `|` scaffolding for cells that do not exist, and
/// inventing empty cells is exactly the kind of fabricated content that
/// inflates the recovery ratio past 1.0 (`kopitiam_token_max.md` §2.1). This
/// change only *reformats* cell text that was already present, and the pipes
/// and separator row it emits are the same scaffolding
/// `strip_rendered_markdown_syntax` already discounts, so the ratio stays in
/// `[0.98, 1.0]`.
pub(super) fn try_table(lines: &[Line]) -> Option<(Table, usize)> {
    // The header sets the shape every body row must match. An empty first line
    // (no cells) or one with too few columns is not a table header.
    let column_count = lines.first()?.cells.len();
    if column_count < MIN_TABLE_COLUMNS {
        return None;
    }

    let header_cells = &lines[0].cells;
    let mut run_end = 0;
    for line in lines {
        if line.cells.len() != column_count {
            break;
        }
        let aligned = line
            .cells
            .iter()
            .zip(header_cells)
            .all(|(cell, header_cell)| (cell.x - header_cell.x).abs() <= COLUMN_X_TOLERANCE);
        if !aligned {
            break;
        }
        run_end += 1;
    }

    if run_end < MIN_TABLE_ROWS {
        return None;
    }

    let headers = header_cells.iter().map(|cell| cell.text.clone()).collect();
    let rows = lines[1..run_end]
        .iter()
        .map(|line| line.cells.iter().map(|cell| cell.text.clone()).collect())
        .collect();

    Some((Table { headers, rows }, run_end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(cells: &[(&str, f32)]) -> Line {
        Line {
            text: cells.iter().map(|(t, _)| *t).collect::<Vec<_>>().join(" "),
            y: 0.0,
            font_size: 10.0,
            cells: cells
                .iter()
                .map(|(t, x)| Cell {
                    text: t.to_string(),
                    x: *x,
                    x_end: *x + 20.0,
                })
                .collect(),
        }
    }

    #[test]
    fn aligned_two_column_rows_become_a_table() {
        let lines = vec![
            line(&[("Metric", 0.0), ("Value", 60.0)]),
            line(&[("Commits", 0.0), ("282", 60.0)]),
            line(&[("Outside", 0.0), ("81", 60.0)]),
        ];
        let (table, consumed) = try_table(&lines).unwrap();
        assert_eq!(table.headers, vec!["Metric", "Value"]);
        assert_eq!(
            table.rows,
            vec![vec!["Commits", "282"], vec!["Outside", "81"]]
        );
        assert_eq!(consumed, 3);
    }

    #[test]
    fn misaligned_columns_are_not_a_table() {
        let lines = vec![
            line(&[("Metric", 0.0), ("Value", 60.0)]),
            line(&[("Commits", 0.0), ("282", 90.0)]),
        ];
        assert!(try_table(&lines).is_none());
    }

    #[test]
    fn single_row_is_not_a_table() {
        let lines = vec![line(&[("Metric", 0.0), ("Value", 60.0)])];
        assert!(try_table(&lines).is_none());
    }

    #[test]
    fn ragged_row_truncates_the_table_to_its_uniform_prefix() {
        // A 5-line run whose 4th line is ragged (a merged/subtotal row with
        // only one cell). The leading three uniform rows must still come out
        // as a table; the ragged row and everything after it are left for
        // normal block handling (reported via the consumed count), NOT
        // swallowed into the table and NOT collapsed with it into paragraphs.
        let lines = vec![
            line(&[("Metric", 0.0), ("Value", 60.0)]),
            line(&[("Commits", 0.0), ("282", 60.0)]),
            line(&[("Outside", 0.0), ("81", 60.0)]),
            line(&[("Subtotal across both", 0.0)]), // ragged: one cell
            line(&[("Notes", 0.0), ("later", 60.0)]),
        ];

        let (table, consumed) = try_table(&lines).unwrap();
        assert_eq!(consumed, 3, "table must stop at the last uniform row");
        assert_eq!(table.headers, vec!["Metric", "Value"]);
        assert_eq!(
            table.rows,
            vec![vec!["Commits", "282"], vec!["Outside", "81"]],
            "only the uniform prefix rows belong to the table"
        );
        // The remainder (the ragged row onward) is untouched by try_table, so
        // build_blocks resumes at index `consumed` and handles it normally.
        assert!(consumed < lines.len(), "remainder must be left for the caller");
    }

    #[test]
    fn a_misaligned_row_partway_down_cuts_the_table_there() {
        // Same idea via x-misalignment rather than cell count: row 3's second
        // cell drifts past COLUMN_X_TOLERANCE, so the table is the first two
        // rows and the drifting row continues as a normal block.
        let lines = vec![
            line(&[("Metric", 0.0), ("Value", 60.0)]),
            line(&[("Commits", 0.0), ("282", 60.0)]),
            line(&[("Outside", 0.0), ("81", 90.0)]), // misaligned column 2
            line(&[("Reviews", 0.0), ("7", 60.0)]),
        ];

        let (table, consumed) = try_table(&lines).unwrap();
        assert_eq!(consumed, 2);
        assert_eq!(table.headers, vec!["Metric", "Value"]);
        assert_eq!(table.rows, vec![vec!["Commits", "282"]]);
    }

    #[test]
    fn a_ragged_row_too_early_leaves_no_valid_prefix() {
        // If uniformity breaks before MIN_TABLE_ROWS is reached, there is no
        // table at all -- the prefix is too short to be one.
        let lines = vec![
            line(&[("Metric", 0.0), ("Value", 60.0)]),
            line(&[("Only one cell here", 0.0)]),
            line(&[("Commits", 0.0), ("282", 60.0)]),
        ];
        assert!(try_table(&lines).is_none());
    }

    #[test]
    fn too_few_columns_is_not_a_table() {
        // A run of single-cell lines never has MIN_TABLE_COLUMNS columns.
        let lines = vec![
            line(&[("Just one", 0.0)]),
            line(&[("Another", 0.0)]),
            line(&[("And another", 0.0)]),
        ];
        assert!(try_table(&lines).is_none());
    }
}
