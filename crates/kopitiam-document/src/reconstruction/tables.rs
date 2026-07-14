use super::Line;
use crate::Table;

#[cfg(test)]
use super::Cell;

const MIN_TABLE_ROWS: usize = 2;
const MIN_TABLE_COLUMNS: usize = 2;
const COLUMN_X_TOLERANCE: f32 = 8.0;

/// Detects a table as a run of consecutive lines that all split into the
/// same number of geometric cells at matching x-positions. Falls through to
/// paragraph handling on anything ambiguous, rather than guessing.
pub(super) fn try_table(lines: &[Line]) -> Option<(Table, usize)> {
    let mut run_end = 0;
    for line in lines {
        if line.cells.len() >= MIN_TABLE_COLUMNS {
            run_end += 1;
        } else {
            break;
        }
    }

    if run_end < MIN_TABLE_ROWS {
        return None;
    }

    let column_count = lines[0].cells.len();
    for line in &lines[..run_end] {
        if line.cells.len() != column_count {
            return None;
        }
        for (cell, first_row_cell) in line.cells.iter().zip(&lines[0].cells) {
            if (cell.x - first_row_cell.x).abs() > COLUMN_X_TOLERANCE {
                return None;
            }
        }
    }

    let headers = lines[0]
        .cells
        .iter()
        .map(|cell| cell.text.clone())
        .collect();
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
}
