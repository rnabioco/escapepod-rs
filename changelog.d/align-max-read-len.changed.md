- **`escpod align` no longer aligns reads longer than 1,000 nt by default.**
  New `--max-read-len <N>` (default `1000`, measured on SEQ; `0` = no limit,
  which restores the previous output exactly). A longer read is not scored on
  the CPU or the GPU and not traced, but it is still written — in input order,
  unmapped (flag 4), every input tag copied, no `NM`/`MD`/`AS`/`XS`/`XA` —
  the same way a read below `--min-score` is; under `--strand both` neither
  strand is scored. The run reports how many reads were skipped. On a
  490 k-read tRNA sample 687 reads (0.14%) are over the limit, and skipping
  them takes the run from 15.1 to 13.3 s on 16 rna cores and from 7.6 to
  6.5 s with `--device gpu`. **Default output changes** for any input with
  reads over 1,000 nt: those reads were previously aligned (#415).
