- `escpod classify`'s output BAM `@PG` record now carries a `DS` field with
  full model provenance — `model_version`, the loaded scorer's sha256,
  `basecaller`, `operating_point`, whether a calibration is shipped, and the
  abstain rule — so which model/basecaller pairing produced a BAM no longer
  depends on having kept that run's log (#370).
