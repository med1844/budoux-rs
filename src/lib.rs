//! # Overview
//!
//! BudouX-rs is a rust port of [BudouX](https://github.com/google/budoux) (machine learning powered line break organizer tool).
//!
//! Note:
//! This project contains the deliverables of the [BudouX](https://github.com/google/budoux) project.
//!
//! Note:
//! BudouX-rs supported plain text only, not supports html inputs.

use std::{collections::HashMap, ops::Range};

mod unicode_blocks;

/// models provides trained machine learning model.
pub mod models;

/// DEFAULT_THRESHOLD is default threshold for splitting a sentences.
pub const DEFAULT_THRESHOLD: i32 = 1000;

/// Model is type of trained machine learning model.
#[derive(Debug, PartialEq, Eq)]
pub struct Model {
    map: HashMap<String, HashMap<String, i32>>,
    base_score: i32,
}

impl Model {
    /// Creates a new `Model` instance from a nested `HashMap` and calculates the base score.
    /// The input `HashMap` must have the same structure as BudouX model json files.
    pub fn new(map: HashMap<String, HashMap<String, i32>>) -> Self {
        let base_score = -map
            .values()
            .into_iter()
            .map(|v| v.values().copied().sum::<i32>())
            .sum::<i32>();
        Self { map, base_score }
    }

    pub fn from_reader<R: std::io::Read>(reader: R) -> serde_json::Result<Self> {
        Ok(Self::new(serde_json::from_reader(reader)?))
    }

    pub fn from_str(s: &str) -> serde_json::Result<Self> {
        Ok(Self::new(serde_json::from_str(s)?))
    }

    pub fn as_inner(&self) -> &HashMap<String, HashMap<String, i32>> {
        &self.map
    }

    fn get(&self, key: &str) -> Option<&HashMap<String, i32>> {
        self.map.get(key)
    }

    /// parse returns splitted string slice from input.
    ///
    /// * `input` - input sentences.
    ///
    /// # Examples
    ///
    /// Split sentences with internal model.
    ///
    /// ```rust
    /// let model = budoux::models::default_japanese_model();
    /// let words = model.parse("これはテストです。");
    ///
    /// assert_eq!(words, vec!["これは", "テストです。"]);
    /// ```
    ///
    /// Load model from json file and split sentences using the loaded model.
    ///
    /// ```ignore
    /// let file = File::open(path_to_json).unwrap();
    /// let reader = BufReader::new(file);
    /// let model = budoux::Model::from_reader(reader).unwrap();
    /// let words = model.parse("これはテストです。");
    ///
    /// assert_eq!(words, vec!["これは", "テストです。"]);
    /// ```
    pub fn parse<'i>(&self, input: &'i str) -> Vec<&'i str> {
        if input.is_empty() {
            return vec![];
        }
        let chars = input.char_indices().collect::<Vec<_>>();
        fn to_range(i: (usize, char)) -> Range<usize> {
            let (start, char) = i;
            start..start + char.len_utf8()
        }
        fn merge_range(a: Range<usize>, b: Range<usize>) -> Range<usize> {
            a.start.min(b.start)..a.end.max(b.end)
        }
        fn get_merged_range(start: usize, end: usize, chars: &[(usize, char)]) -> Range<usize> {
            merge_range(to_range(chars[start]), to_range(chars[end - 1]))
        }
        fn get_score(model: &Model, key: &str, range: Range<usize>, text: &str) -> i32 {
            model
                .get(key)
                .and_then(|v| v.get(&text[range]))
                .copied()
                .unwrap_or(0)
                * 2
        }
        assert!(chars.len() > 0);
        let mut chunks = vec![to_range(chars[0])];
        for i in 1..chars.len() {
            let mut score = self.base_score;
            if i > 2 {
                score += get_score(self, "UW1", to_range(chars[i - 3]), input);
            }
            if i > 1 {
                score += get_score(self, "UW2", to_range(chars[i - 2]), input);
            }
            score += get_score(self, "UW3", to_range(chars[i - 1]), input);
            score += get_score(self, "UW4", to_range(chars[i]), input);
            if i + 1 < chars.len() {
                score += get_score(self, "UW5", to_range(chars[i + 1]), input);
            }
            if i + 2 < chars.len() {
                score += get_score(self, "UW6", to_range(chars[i + 2]), input);
            }

            if i > 1 {
                score += get_score(self, "BW1", get_merged_range(i - 2, i, &chars), input);
            }
            score += get_score(self, "BW2", get_merged_range(i - 1, i, &chars), input);
            if i + 1 < chars.len() {
                score += get_score(self, "BW3", get_merged_range(i, i + 2, &chars), input);
            }

            if i > 2 {
                score += get_score(self, "TW1", get_merged_range(i - 3, i, &chars), input);
            }
            if i > 1 {
                score += get_score(self, "TW2", get_merged_range(i - 2, i + 1, &chars), input);
            }
            if i + 1 < chars.len() {
                score += get_score(self, "TW3", get_merged_range(i - 1, i + 2, &chars), input);
            }
            if i + 2 < chars.len() {
                score += get_score(self, "TW4", get_merged_range(i, i + 3, &chars), input);
            }
            if score > 0 {
                chunks.push(to_range(chars[i]));
            } else {
                if let Some(last_range) = chunks.last_mut() {
                    let cur_range = to_range(chars[i]);
                    *last_range = merge_range(last_range.clone(), cur_range);
                }
            }
        }
        chunks.into_iter().map(|r| &input[r]).collect()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_parse() {
        let m = super::models::default_japanese_model();

        assert_eq!(m.parse(""), Vec::<&str>::new());
        assert_eq!(m.parse("日本語"), vec!["日本語"]);
        assert_eq!(m.parse("水と油"), vec!["水と", "油"]);
        assert_eq!(
            m.parse("水道水とミネラルウォーター"),
            vec!["水道水と", "ミネラルウォーター"]
        );
        assert_eq!(
            m.parse("PythonとJavaScriptとGolang"),
            vec!["Pythonと", "JavaScriptと", "Golang"]
        );
        assert_eq!(
            m.parse("日本語の文章において語の区切りに空白を挟んで記述すること"),
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
            m.parse("これはテストです。"),
            vec!["これは", "テストです。"]
        );
        assert_eq!(
            m.parse("これは美しいペンです。"),
            vec!["これは", "美しい", "ペンです。"]
        );
        assert_eq!(m.parse("今日は天気です。"), vec!["今日は", "天気です。"]);
        assert_eq!(
            m.parse("今日はとても天気です。"),
            vec!["今日は", "とても", "天気です。"]
        );
        assert_eq!(
            m.parse("あなたに寄り添う最先端のテクノロジー。"),
            vec!["あなたに", "寄り添う", "最先端の", "テクノロジー。"]
        );
        assert_eq!(
            m.parse("これはテストです。今日は晴天です。"),
            vec!["これは", "テストです。", "今日は", "晴天です。"]
        );
        assert_eq!(
            m.parse("これはテストです。\n今日は晴天です。"),
            vec!["これは", "テストです。", "\n今日は", "晴天です。"]
        );
    }

    #[test]
    fn test_parse_zh_hans() {
        let m = super::models::default_simplified_chinese_model();

        assert_eq!(m.parse(""), Vec::<&str>::new());
        assert_eq!(m.parse("今天是晴天。"), vec!["今天", "是", "晴天。"]);
    }

    #[test]
    fn test_parse_zh_hans_on_mixed() {
        let m = super::models::default_simplified_chinese_model();
        assert_eq!(
            m.parse("你喜欢看アニメ吗"),
            vec!["你", "喜欢", "看", "アニメ", "吗"]
        )
    }
}
