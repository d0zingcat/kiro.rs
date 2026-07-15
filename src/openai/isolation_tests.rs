#[test]
fn openai_sources_must_not_import_anthropic() {
    let openai_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/openai");
    let forbidden = ["crate", "anthropic"].join("::");
    for entry in std::fs::read_dir(&openai_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap();
        assert!(
            !src.contains(&forbidden),
            "{} must not import {}",
            path.display(),
            forbidden
        );
    }
}
