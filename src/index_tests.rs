#[cfg(test)]
mod index_tests {
    use crate::index::SearchResult;
    
    #[test]
    fn test_search_result_compatible() {
        let result = SearchResult::new(
            vec![1, 2, 3],
            vec![0.1, 0.2, 0.3],
            1.5,
        );
        assert_eq!(result.len(), 3);
    }
}
