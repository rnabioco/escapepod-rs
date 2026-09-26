- **`escpod align --read-ids` now accepts the same UUID spellings `escpod
  filter` does.** The ID list and each record's name are canonicalised
  through `parse_uuid_flexible` when they parse as a UUID (dashed or the
  compact 32-hex-char form), falling back to raw bytes for a non-UUID name
  (a FASTQ read name). Previously a dashless ID list matched raw bytes
  against dashed read names and silently selected nothing (#409).
