use std::collections::HashMap;

/// Scalar field value.
#[derive(Clone, Debug)]
pub enum ScalarValue {
    Int64(i64),
    Float64(f64),
    Bool(bool),
    Str(String),
}

/// Filter expression tree.
#[derive(Clone, Debug)]
pub enum FilterExpr {
    Eq(String, ScalarValue),
    Ne(String, ScalarValue),
    Lt(String, ScalarValue),
    Le(String, ScalarValue),
    Gt(String, ScalarValue),
    Ge(String, ScalarValue),
    And(Box<FilterExpr>, Box<FilterExpr>),
    Or(Box<FilterExpr>, Box<FilterExpr>),
    Not(Box<FilterExpr>),
    In(String, Vec<ScalarValue>),
}

/// Scalar fields for one row/document.
pub type ScalarRow = HashMap<String, ScalarValue>;

/// Materialized scalar view for pre-filtering document ids.
pub struct MaterializedView {
    /// doc_id -> scalar row
    rows: Vec<Option<ScalarRow>>,
}

impl MaterializedView {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    /// Insert or update row for a given doc id.
    pub fn upsert(&mut self, doc_id: usize, row: ScalarRow) {
        if doc_id >= self.rows.len() {
            self.rows.resize_with(doc_id + 1, || None);
        }
        self.rows[doc_id] = Some(row);
    }

    /// Delete row for a given doc id.
    pub fn delete(&mut self, doc_id: usize) {
        if doc_id < self.rows.len() {
            self.rows[doc_id] = None;
        }
    }

    /// Current capacity (includes empty slots).
    pub fn capacity(&self) -> usize {
        self.rows.len()
    }

    /// Filter a given doc id list, preserving input order.
    pub fn filter(&self, doc_ids: &[usize], expr: &FilterExpr) -> Vec<usize> {
        doc_ids
            .iter()
            .copied()
            .filter(|&doc_id| {
                self.rows
                    .get(doc_id)
                    .and_then(|r| r.as_ref())
                    .is_some_and(|row| eval_expr(expr, row))
            })
            .collect()
    }

    /// Scan all doc ids (0..capacity) and return ids passing filter.
    pub fn scan(&self, expr: &FilterExpr) -> Vec<usize> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            return self
                .rows
                .par_iter()
                .enumerate()
                .filter_map(|(doc_id, row)| {
                    row.as_ref()
                        .filter(|r| eval_expr(expr, r))
                        .map(|_| doc_id)
                })
                .collect();
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.rows
                .iter()
                .enumerate()
                .filter_map(|(doc_id, row)| {
                    row.as_ref()
                        .filter(|r| eval_expr(expr, r))
                        .map(|_| doc_id)
                })
                .collect()
        }
    }
}

fn eval_expr(expr: &FilterExpr, row: &ScalarRow) -> bool {
    match expr {
        FilterExpr::Eq(field, rhs) => row
            .get(field)
            .is_some_and(|lhs| cmp_eq(lhs, rhs)),
        FilterExpr::Ne(field, rhs) => row
            .get(field)
            .is_some_and(|lhs| !cmp_eq(lhs, rhs)),
        FilterExpr::Lt(field, rhs) => row
            .get(field)
            .is_some_and(|lhs| cmp_lt(lhs, rhs)),
        FilterExpr::Le(field, rhs) => row
            .get(field)
            .is_some_and(|lhs| cmp_le(lhs, rhs)),
        FilterExpr::Gt(field, rhs) => row
            .get(field)
            .is_some_and(|lhs| cmp_gt(lhs, rhs)),
        FilterExpr::Ge(field, rhs) => row
            .get(field)
            .is_some_and(|lhs| cmp_ge(lhs, rhs)),
        FilterExpr::And(a, b) => eval_expr(a, row) && eval_expr(b, row),
        FilterExpr::Or(a, b) => eval_expr(a, row) || eval_expr(b, row),
        FilterExpr::Not(inner) => !eval_expr(inner, row),
        FilterExpr::In(field, list) => row
            .get(field)
            .is_some_and(|lhs| list.iter().any(|rhs| cmp_eq(lhs, rhs))),
    }
}

fn cmp_eq(a: &ScalarValue, b: &ScalarValue) -> bool {
    match (a, b) {
        (ScalarValue::Int64(x), ScalarValue::Int64(y)) => x == y,
        (ScalarValue::Float64(x), ScalarValue::Float64(y)) => x == y,
        (ScalarValue::Bool(x), ScalarValue::Bool(y)) => x == y,
        (ScalarValue::Str(x), ScalarValue::Str(y)) => x == y,
        _ => false,
    }
}

fn cmp_lt(a: &ScalarValue, b: &ScalarValue) -> bool {
    match (a, b) {
        (ScalarValue::Int64(x), ScalarValue::Int64(y)) => x < y,
        (ScalarValue::Float64(x), ScalarValue::Float64(y)) => x < y,
        _ => false,
    }
}

fn cmp_le(a: &ScalarValue, b: &ScalarValue) -> bool {
    match (a, b) {
        (ScalarValue::Int64(x), ScalarValue::Int64(y)) => x <= y,
        (ScalarValue::Float64(x), ScalarValue::Float64(y)) => x <= y,
        _ => false,
    }
}

fn cmp_gt(a: &ScalarValue, b: &ScalarValue) -> bool {
    match (a, b) {
        (ScalarValue::Int64(x), ScalarValue::Int64(y)) => x > y,
        (ScalarValue::Float64(x), ScalarValue::Float64(y)) => x > y,
        _ => false,
    }
}

fn cmp_ge(a: &ScalarValue, b: &ScalarValue) -> bool {
    match (a, b) {
        (ScalarValue::Int64(x), ScalarValue::Int64(y)) => x >= y,
        (ScalarValue::Float64(x), ScalarValue::Float64(y)) => x >= y,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(age: i64, city: &str, active: bool) -> ScalarRow {
        let mut r = ScalarRow::new();
        r.insert("age".to_string(), ScalarValue::Int64(age));
        r.insert("city".to_string(), ScalarValue::Str(city.to_string()));
        r.insert("active".to_string(), ScalarValue::Bool(active));
        r
    }

    fn build_mv() -> MaterializedView {
        let mut mv = MaterializedView::new();
        mv.upsert(0, row(20, "Beijing", true));
        mv.upsert(1, row(25, "Shanghai", true));
        mv.upsert(2, row(30, "Beijing", false));
        mv.upsert(3, row(30, "Shenzhen", true));
        mv.upsert(4, row(40, "Beijing", true));
        mv
    }

    #[test]
    fn test_mv_filter_eq() {
        let mv = build_mv();
        let expr = FilterExpr::Eq("age".to_string(), ScalarValue::Int64(30));
        let out = mv.filter(&[0, 1, 2, 3, 4], &expr);
        assert_eq!(out, vec![2, 3]);
    }

    #[test]
    fn test_mv_filter_and() {
        let mv = build_mv();
        let expr = FilterExpr::And(
            Box::new(FilterExpr::Ge("age".to_string(), ScalarValue::Int64(25))),
            Box::new(FilterExpr::Eq(
                "city".to_string(),
                ScalarValue::Str("Beijing".to_string()),
            )),
        );
        let out = mv.filter(&[0, 1, 2, 3, 4], &expr);
        assert_eq!(out, vec![2, 4]);
    }

    #[test]
    fn test_mv_filter_scan() {
        let mv = build_mv();
        let expr = FilterExpr::Eq("city".to_string(), ScalarValue::Str("Beijing".to_string()));
        let out = mv.scan(&expr);
        assert_eq!(out.len(), 3);
        assert_eq!(out, vec![0, 2, 4]);
    }

    #[test]
    fn test_mv_delete() {
        let mut mv = build_mv();
        mv.delete(2);
        let expr = FilterExpr::Eq("age".to_string(), ScalarValue::Int64(30));
        let out = mv.scan(&expr);
        assert_eq!(out, vec![3]);
    }
}
