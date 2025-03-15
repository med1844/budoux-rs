use once_cell::sync::Lazy;

/// MODEL reference to trained machine learning model.
pub static MODEL: Lazy<crate::Model> =
    Lazy::new(|| crate::Model::from_str(include_str!("../../resources/ja.json")).unwrap());
