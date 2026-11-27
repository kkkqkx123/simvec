/// Result of splitting one logical command line.
#[derive(Debug, PartialEq, Eq)]
pub enum Split {
    Ready(Vec<String>),
    Continuation,
}

/// Split a command line into arguments, honoring quotes and escapes.
///
/// Double quotes suppress word splitting and treat a backslash before a
/// quote or another backslash as an escape. Other backslash sequences are
/// preserved so JSON escapes such as `\n` reach the JSON parser unchanged.
/// Single quotes suppress word splitting and keep every backslash literal.
pub fn split(line: &str) -> Result<Split, String> {
    let chars: Vec<char> = line.chars().collect();
    let mut args = Vec::new();
    let mut current = String::new();
    let mut started = false;
    let mut quote: Option<char> = None;
    let mut index = 0;

    while index < chars.len() {
        let ch = chars[index];
        match quote {
            Some('\'') => {
                if ch == '\'' {
                    quote = None;
                } else {
                    current.push(ch);
                }
                index += 1;
            }
            Some('"') => {
                if ch == '"' {
                    quote = None;
                    index += 1;
                } else if ch == '\\' && index + 1 < chars.len() {
                    let next = chars[index + 1];
                    if next == '"' || next == '\\' {
                        current.push(next);
                    } else {
                        current.push(ch);
                        current.push(next);
                    }
                    index += 2;
                } else if ch == '\\' {
                    return Ok(Split::Continuation);
                } else {
                    current.push(ch);
                    index += 1;
                }
            }
            _ => {
                if ch.is_whitespace() {
                    if started {
                        args.push(std::mem::take(&mut current));
                        started = false;
                    }
                    index += 1;
                } else if ch == '\'' || ch == '"' {
                    quote = Some(ch);
                    started = true;
                    index += 1;
                } else if ch == '\\' {
                    if index + 1 >= chars.len() {
                        return Ok(Split::Continuation);
                    }
                    current.push(chars[index + 1]);
                    started = true;
                    index += 2;
                } else {
                    current.push(ch);
                    started = true;
                    index += 1;
                }
            }
        }
    }

    if quote.is_some() {
        return Ok(Split::Continuation);
    }
    if started {
        args.push(current);
    }
    Ok(Split::Ready(args))
}

#[cfg(test)]
mod tests {
    use super::{split, Split};

    #[test]
    fn splits_plain_arguments() {
        assert_eq!(
            split("create docs 3 Cosine"),
            Ok(Split::Ready(vec![
                "create".into(),
                "docs".into(),
                "3".into(),
                "Cosine".into(),
            ]))
        );
    }

    #[test]
    fn keeps_json_in_single_quotes() {
        assert_eq!(
            split("search docs '{\"vector\":[1,0],\"limit\":2}'"),
            Ok(Split::Ready(vec![
                "search".into(),
                "docs".into(),
                "{\"vector\":[1,0],\"limit\":2}".into(),
            ]))
        );
    }

    #[test]
    fn preserves_json_escape_sequences() {
        assert_eq!(
            split("set_field docs a title \"line\\nnext\""),
            Ok(Split::Ready(vec![
                "set_field".into(),
                "docs".into(),
                "a".into(),
                "title".into(),
                "line\\nnext".into(),
            ]))
        );
    }

    #[test]
    fn reports_unclosed_quotes_as_continuation() {
        assert_eq!(split("upsert docs '[{"), Ok(Split::Continuation));
    }

    #[test]
    fn keeps_hash_inside_quotes() {
        assert_eq!(
            split("set_field docs a tag '#rust'"),
            Ok(Split::Ready(vec![
                "set_field".into(),
                "docs".into(),
                "a".into(),
                "tag".into(),
                "#rust".into(),
            ]))
        );
    }
}
