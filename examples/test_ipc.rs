use podfive_core::{Reader, arrow_ipc::ArrowIpcFooter};

fn main() -> anyhow::Result<()> {
    let reader = Reader::open("data/dna/pod5/FBC74904_90b682e0_e09f0700_0.pod5")?;
    
    let signal_bytes = reader.signal_table_bytes()?;
    println!("Signal table size: {} bytes ({:.2} MB)", 
             signal_bytes.len(), 
             signal_bytes.len() as f64 / 1024.0 / 1024.0);
    
    let footer = ArrowIpcFooter::parse(signal_bytes)?;
    println!("Found {} record batches", footer.record_batches.len());
    
    for (i, batch) in footer.record_batches.iter().enumerate().take(5) {
        println!("  Batch {}: offset={}, meta={}, body={}, total={}",
                 i, batch.offset, batch.metadata_length, batch.body_length, batch.total_length());
    }
    
    if footer.record_batches.len() > 5 {
        println!("  ... and {} more", footer.record_batches.len() - 5);
    }
    
    Ok(())
}
