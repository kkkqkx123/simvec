use simvec_cli::lexer::{split, Split};

#[test]
fn keeps_json_argument_intact() {
    let result = split("search docs '{\"vector\":[1,0],\"limit\":2}'").unwrap();
    assert_eq!(
        result,
        Split::Ready(vec![
            "search".to_string(),
            "docs".to_string(),
            "{\"vector\":[1,0],\"limit\":2}".to_string(),
        ])
    );
}

#[test]
fn continues_unterminated_quotes() {
    assert_eq!(split("upsert docs '[{"), Ok(Split::Continuation));
}
