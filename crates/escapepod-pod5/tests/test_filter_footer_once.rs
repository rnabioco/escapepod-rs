//! `filter` resolves each input's signal geometry once (#465).
//!
//! The signal footer parse reads every signal batch's own header: one
//! scattered touch per batch, 15-24 ms each cold on BeeGFS. The CLI used to
//! open every input a first time just to ask `nonuniform_signal_batch()`, and
//! the filter then opened and walked them all again. The cost is invisible in
//! the output and in a reads/s number, so this counts the walks directly.
//!
//! One test in this file on purpose: the counter is process-global.

mod common;

use std::collections::HashSet;

use escapepod_pod5::operations::{FilterCriteria, FilterOptions, filter_files_with_criteria};
use escapepod_pod5::reader::signal_footer_parses;
use escapepod_pod5::{Reader, Uuid, Writer, WriterOptions};
use tempfile::TempDir;

use common::{make_read, make_run_info, synth_signal};

const FILES: usize = 3;
const READS_PER_FILE: usize = 23;
const SAMPLES: usize = 300;

#[test]
fn filter_opens_footer_once() {
    let tmp = TempDir::new().unwrap();
    let mut paths = Vec::new();
    let mut wanted: HashSet<Uuid> = HashSet::new();
    for f in 0..FILES {
        let path = tmp.path().join(format!("in{f}.pod5"));
        let options = WriterOptions {
            signal_batch_size: 4,
            ..Default::default()
        };
        let mut writer = Writer::create(&path, options).unwrap();
        let run = writer
            .add_run_info(make_run_info(&format!("acq{f}")))
            .unwrap();
        for i in 0..READS_PER_FILE {
            let read = make_read(run, i as u32 + 1, SAMPLES as u64);
            if i % 5 == 0 {
                wanted.insert(read.read_id);
            }
            writer
                .add_read(read, &synth_signal(SAMPLES, (f * 100 + i) as u64))
                .unwrap();
        }
        writer.finish().unwrap();
        paths.push(path);
    }

    let before = signal_footer_parses();
    let output = tmp.path().join("out.pod5");
    let criteria = FilterCriteria {
        read_ids: Some(wanted.clone()),
        ..Default::default()
    };
    let result =
        filter_files_with_criteria(&paths, &output, &criteria, FilterOptions::default(), None)
            .unwrap();
    let walked = signal_footer_parses() - before;

    assert_eq!(result.matched_reads as usize, wanted.len());
    assert_eq!(
        walked, FILES,
        "filter walked {walked} signal footers for {FILES} inputs; once each is the floor"
    );
    // The portability answer comes from that same walk, not a second open.
    assert!(result.nonuniform_inputs.is_empty());
    assert!(Reader::open(&output).is_ok());
}
