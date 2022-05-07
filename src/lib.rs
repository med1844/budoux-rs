//! # Overview
//!
//! BudouX-rs is a rust port of [BudouX](https://github.com/google/budoux) (machine learning powered line break organizer tool).
//!
//! Note:
//! This project contains the deliverables of the [BudouX](https://github.com/google/budoux) project.
//!
//! Note:
//! BudouX-rs supported plain text only, not supports html inputs.

use std::collections::HashMap;

mod unicode_blocks;

/// models provides trained machine learning model.
pub mod models;

/// DEFAULT_THRESHOLD is default threshold for splitting a sentences.
pub const DEFAULT_THRESHOLD: i32 = 1000;

/// Model is type of trained machine learning model.
/// key (String) is feature of character, value (i32) is score of feature.
pub type Model = HashMap<String, i32>;

/// INVALID_FEATURE is indicate for invalid feature.
const INVALID_FEATURE: &str = "▔";

/// parse returns splitted string slice from input.
/// It is shorthand for budoux::parse_with_threshold(model, input, budoux::DEFAULT_THRESHOLD).
///
/// * `model` - trained machine learning model.
/// * `input` - input sentences.
///
/// # Examples
///
/// Split sentences with internal model.
///
/// ```
/// let model = budoux::models::default_japanese_model();
/// let words = budoux::parse(model, "これはテストです。");
///
/// assert_eq!(words, vec!["これは", "テストです。"]);
/// ```
///
/// Load model from json file and split sentences using the loaded model.
///
/// ```ignore
/// let file = File::open(path_to_json).unwrap();
/// let reader = BufReader::new(file);
/// let model: budoux::Model = serde_json::from_reader(reader).unwrap();
/// let words = budoux::parse(&model, "これはテストです。");
///
/// assert_eq!(words, vec!["これは", "テストです。"]);
/// ```
pub fn parse(model: &Model, input: &str) -> Vec<String> {
    parse_with_threshold(model, input, DEFAULT_THRESHOLD)
}

/// parse_with_threshold returns splitted string slice from input.
///
/// * `model` - trained machine learning model.
/// * `input` - input sentences.
/// * `threshold` - threshold for splitting a sentences.
///
/// # Examples
///
/// Split sentences with internal model.
///
/// ```
/// let model = budoux::models::default_japanese_model();
/// let words = budoux::parse_with_threshold(model, "これはテストです。", budoux::DEFAULT_THRESHOLD);
///
/// assert_eq!(words, vec!["これは", "テストです。"]);
/// ```
///
/// If you use a large threshold, will not be split.
///
/// ```
/// let model = budoux::models::default_japanese_model();
/// let words = budoux::parse_with_threshold(model, "これはテストです。", 100000000);
///
/// assert_eq!(words, vec!["これはテストです。"]);
/// ```
pub fn parse_with_threshold(model: &Model, input: &str, threshold: i32) -> Vec<String> {
    let chars: Vec<char> = input.chars().collect();

    if chars.len() <= 1 {
        return vec![input.to_string()];
    }

    let mut out: Vec<String> = Vec::new();
    let mut buf: String = chars[0].to_string();

    let mut p1 = "U"; // unknown
    let mut p2 = "U"; // unknown
    let mut p3 = "U"; // unknown

    let (mut w1, mut b1) = (String::from(""), String::from(INVALID_FEATURE)); // i - 3
    let (mut w2, mut b2) = (String::from(""), String::from(INVALID_FEATURE)); // i - 2
    let (mut w3, mut b3) = get_unicode_block_and_feature(&chars, 0); // i - 1
    let (mut w4, mut b4) = get_unicode_block_and_feature(&chars, 1); // i
    let (mut w5, mut b5) = get_unicode_block_and_feature(&chars, 2); // i + 1

    let mut wb = String::with_capacity(20); // working buffer

    for i in 1..chars.len() {
        let (w6, b6) = get_unicode_block_and_feature(&chars, i + 2);

        let score = get_feature(
            model, &mut wb, &w1, &w2, &w3, &w4, &w5, &w6, &b1, &b2, &b3, &b4, &b5, &b6, p1, p2, p3,
        );

        if score > threshold {
            out.push(buf);
            buf = w4.to_string();
        } else {
            buf += &w4;
        }

        p1 = p2;
        p2 = p3;

        if score > 0 {
            p3 = "B"; // positive
        } else {
            p3 = "O"; // negative
        }

        w1 = w2;
        w2 = w3;
        w3 = w4;
        w4 = w5;
        w5 = w6;

        b1 = b2;
        b2 = b3;
        b3 = b4;
        b4 = b5;
        b5 = b6;
    }

    if !buf.is_empty() {
        out.push(buf);
    }

    out
}

/// get_unicode_block_and_feature returns unicode character and block feature from char slice.
fn get_unicode_block_and_feature(chars: &[char], index: usize) -> (String, String) {
    if chars.len() <= index {
        return (String::from(""), String::from(INVALID_FEATURE)); // out of index.
    }

    let v = chars[index];
    let c = v as u32;

    let pos = match unicode_blocks::UNICODE_BLOCKS.binary_search(&c) {
        Ok(v) => v + 1,
        Err(e) => e,
    };

    return (v.to_string(), format!("{:>03}", pos));
}

/// get_feature returns feature list.
#[allow(clippy::too_many_arguments)]
fn get_feature(
    model: &Model,
    buf: &mut String, // working buffer
    w1: &str,
    w2: &str,
    w3: &str,
    w4: &str,
    w5: &str,
    w6: &str,
    b1: &str,
    b2: &str,
    b3: &str,
    b4: &str,
    b5: &str,
    b6: &str,
    p1: &str,
    p2: &str,
    p3: &str,
) -> i32 {
    let mut score: i32 = 0;

    // UP is means unigram of previous results.
    score += model.get(key(buf, &["UP1:", p1])).unwrap_or(&0);
    score += model.get(key(buf, &["UP2:", p2])).unwrap_or(&0);
    score += model.get(key(buf, &["UP3:", p3])).unwrap_or(&0);
    // BP is means bigram of previous results.
    score += model.get(key(buf, &["BP1:", p1, p2])).unwrap_or(&0);
    score += model.get(key(buf, &["BP2:", p2, p3])).unwrap_or(&0);
    // UW is means unigram of words.
    score += model.get(key(buf, &["UW1:", w1])).unwrap_or(&0);
    score += model.get(key(buf, &["UW2:", w2])).unwrap_or(&0);
    score += model.get(key(buf, &["UW3:", w3])).unwrap_or(&0);
    score += model.get(key(buf, &["UW4:", w4])).unwrap_or(&0);
    score += model.get(key(buf, &["UW5:", w5])).unwrap_or(&0);
    score += model.get(key(buf, &["UW6:", w6])).unwrap_or(&0);
    // BW is means bigram of words.
    score += model.get(key(buf, &["BW1:", w2, w3])).unwrap_or(&0);
    score += model.get(key(buf, &["BW2:", w3, w4])).unwrap_or(&0);
    score += model.get(key(buf, &["BW3:", w4, w5])).unwrap_or(&0);
    // TW is means trigram of words.
    score += model.get(key(buf, &["TW1:", w1, w2, w3])).unwrap_or(&0);
    score += model.get(key(buf, &["TW2:", w2, w3, w4])).unwrap_or(&0);
    score += model.get(key(buf, &["TW3:", w3, w4, w5])).unwrap_or(&0);
    score += model.get(key(buf, &["TW4:", w4, w5, w6])).unwrap_or(&0);
    // UB is means unigram of unicode blocks.
    score += model.get(key(buf, &["UB1:", b1])).unwrap_or(&0);
    score += model.get(key(buf, &["UB2:", b2])).unwrap_or(&0);
    score += model.get(key(buf, &["UB3:", b3])).unwrap_or(&0);
    score += model.get(key(buf, &["UB4:", b4])).unwrap_or(&0);
    score += model.get(key(buf, &["UB5:", b5])).unwrap_or(&0);
    score += model.get(key(buf, &["UB6:", b6])).unwrap_or(&0);
    // BB is means bigram of unicode blocks.
    score += model.get(key(buf, &["BB1:", b2, b3])).unwrap_or(&0);
    score += model.get(key(buf, &["BB2:", b3, b4])).unwrap_or(&0);
    score += model.get(key(buf, &["BB3:", b4, b5])).unwrap_or(&0);
    // TB is means trigram of unicode blocks.
    score += model.get(key(buf, &["TB1:", b1, b2, b3])).unwrap_or(&0);
    score += model.get(key(buf, &["TB2:", b2, b3, b4])).unwrap_or(&0);
    score += model.get(key(buf, &["TB3:", b3, b4, b5])).unwrap_or(&0);
    score += model.get(key(buf, &["TB4:", b4, b5, b6])).unwrap_or(&0);
    // UQ is combination of UP and UB.
    score += model.get(key(buf, &["UQ1:", p1, b1])).unwrap_or(&0);
    score += model.get(key(buf, &["UQ2:", p2, b2])).unwrap_or(&0);
    score += model.get(key(buf, &["UQ3:", p3, b3])).unwrap_or(&0);
    // BQ is combination of UP and BB.
    score += model.get(key(buf, &["BQ1:", p2, b2, b3])).unwrap_or(&0);
    score += model.get(key(buf, &["BQ2:", p2, b3, b4])).unwrap_or(&0);
    score += model.get(key(buf, &["BQ3:", p3, b2, b3])).unwrap_or(&0);
    score += model.get(key(buf, &["BQ4:", p3, b3, b4])).unwrap_or(&0);
    // TQ is combination of UP and TB.
    score += model.get(key(buf, &["TQ1:", p2, b1, b2, b3])).unwrap_or(&0);
    score += model.get(key(buf, &["TQ2:", p2, b2, b3, b4])).unwrap_or(&0);
    score += model.get(key(buf, &["TQ3:", p3, b1, b2, b3])).unwrap_or(&0);
    score += model.get(key(buf, &["TQ4:", p3, b2, b3, b4])).unwrap_or(&0);

    score
}

/// key returns feature key.
fn key<'a>(buf: &'a mut String, params: &[&str]) -> &'a str {
    buf.clear();
    for param in params {
        buf.push_str(param);
    }

    buf
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_parse() {
        let m = super::models::default_japanese_model();

        assert_eq!(super::parse(m, ""), vec![""]);
        assert_eq!(super::parse(m, "日本語"), vec!["日本語"]);
        assert_eq!(super::parse(m, "水と油"), vec!["水と", "油"]);
        assert_eq!(
            super::parse(m, "水道水とミネラルウォーター"),
            vec!["水道水と", "ミネラルウォーター"]
        );
        assert_eq!(
            super::parse(m, "PythonとJavaScriptとGolang"),
            vec!["Pythonと", "JavaScriptと", "Golang"]
        );
        assert_eq!(
            super::parse(
                m,
                "日本語の文章において語の区切りに空白を挟んで記述すること"
            ),
            vec![
                "日本語の",
                "文章に",
                "おいて",
                "語の",
                "区切りに",
                "空白を",
                "挟んで",
                "記述する",
                "こと"
            ]
        );
        assert_eq!(
            super::parse(m, "これはテストです。"),
            vec!["これは", "テストです。"]
        );
        assert_eq!(
            super::parse(m, "これは美しいペンです。"),
            vec!["これは", "美しい", "ペンです。"]
        );
        assert_eq!(
            super::parse(m, "今日は天気です。"),
            vec!["今日は", "天気です。"]
        );
        assert_eq!(
            super::parse(m, "今日はとても天気です。"),
            vec!["今日は", "とても", "天気です。"]
        );
        assert_eq!(
            super::parse(m, "あなたに寄り添う最先端のテクノロジー。"),
            vec!["あなたに", "寄り添う", "最先端の", "テクノロジー。"]
        );
        assert_eq!(
            super::parse(m, "これはテストです。今日は晴天です。"),
            vec!["これは", "テストです。", "今日は", "晴天です。"]
        );
        assert_eq!(
            super::parse(m, "これはテストです。\n今日は晴天です。"),
            vec!["これは", "テストです。", "\n今日は", "晴天です。"]
        );
    }

    #[test]
    fn test_parse_zh_hans() {
        let m = super::models::default_simplified_chinese_model();

        assert_eq!(super::parse(m, ""), vec![""]);
        assert_eq!(
            super::parse(m, "今天是晴天。"),
            vec!["今天", "是", "晴天。"]
        );
    }

    #[test]
    fn test_get_unicode_block_and_feature() {
        let to_chars = |x: &str| {
            let chars: Vec<char> = x.chars().collect();
            return chars;
        };

        assert_eq!(
            super::get_unicode_block_and_feature(&to_chars("abc"), 0),
            (String::from("a"), String::from("001"),)
        );
        assert_eq!(
            super::get_unicode_block_and_feature(&to_chars("xyz"), 2),
            (String::from("z"), String::from("001"),)
        );
        assert_eq!(
            super::get_unicode_block_and_feature(&to_chars("abc"), 0),
            (String::from("a"), String::from("001"),)
        );
        assert_eq!(
            super::get_unicode_block_and_feature(&to_chars("out of index"), 12),
            (String::from(""), String::from(super::INVALID_FEATURE),)
        );
        assert_eq!(
            super::get_unicode_block_and_feature(&to_chars("あいうえお"), 0),
            (String::from("あ"), String::from("108"),)
        );
        assert_eq!(
            super::get_unicode_block_and_feature(&to_chars("わをん"), 2),
            (String::from("ん"), String::from("108"),)
        );
        assert_eq!(
            super::get_unicode_block_and_feature(&to_chars("安"), 0),
            (String::from("安"), String::from("120"),)
        );
        assert_eq!(
            super::get_unicode_block_and_feature(&to_chars("範囲外アクセス"), 7),
            (String::from(""), String::from(super::INVALID_FEATURE),)
        );
    }

    #[test]
    fn test_key() {
        let mut wb = String::with_capacity(20);

        assert_eq!(super::key(&mut wb, &[""]), "");
        assert_eq!(super::key(&mut wb, &["AAA", "BBB"]), "AAABBB");
        assert_eq!(super::key(&mut wb, &["AAA", "BBB", "CCC"]), "AAABBBCCC");
        assert_eq!(
            super::key(&mut wb, &["TW4:", "日", "本", "語"]),
            "TW4:日本語"
        );
        assert_eq!(
            super::key(&mut wb, &["TQ4:", "O", "120", "120", "120"]),
            "TQ4:O120120120"
        );

        assert_eq!(wb.capacity(), 20);
    }
}
