#[path = "ja.rs"]
mod ja;

#[path = "th.rs"]
mod th;

#[path = "zh_hans.rs"]
mod zh_hans;

#[path = "zh_hant.rs"]
mod zh_hant;

/// default_japanese_model returns trained machine learning model for japanese.
pub fn default_japanese_model() -> &'static crate::Model {
    &ja::MODEL
}

/// default_thai_model returns trained machine learning model for thai.
pub fn default_thai_model() -> &'static crate::Model {
    &th::MODEL
}

/// default_simplified_chinese_model returns trained machine learning model for simplified chinese.
pub fn default_simplified_chinese_model() -> &'static crate::Model {
    &zh_hans::MODEL
}

/// default_traditional_chinese_model returns trained machine learning model for traditional chinese.
pub fn default_traditional_chinese_model() -> &'static crate::Model {
    &zh_hant::MODEL
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_multiple_ref() {
        let m1 = super::default_japanese_model();
        let m2 = super::default_japanese_model();

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_multiple_ref_zh_hans() {
        let m1 = super::default_simplified_chinese_model();
        let m2 = super::default_simplified_chinese_model();

        assert_eq!(m1, m2);
    }
}
