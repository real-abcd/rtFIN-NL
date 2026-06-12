fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .compile_well_known_types(true)
        .compile(&["proto/search_list.proto"], &["proto"])?;
    Ok(())
}