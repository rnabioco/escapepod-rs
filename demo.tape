# VHS tape file for podfive-rs demo
# Run with: vhs demo.tape

Output demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 900
Set Height 600
Set Theme "Catppuccin Mocha"

Type "# podfive-rs - A fast Rust CLI for POD5 files"
Enter
Sleep 1s

Type "# Let's explore a POD5 file"
Enter
Sleep 500ms

Type "podfive summary ext/pod5-file-format/test_data/multi_fast5_zip_v3.pod5"
Enter
Sleep 3s

Type "# View reads as a table"
Enter
Sleep 500ms

Type "podfive view ext/pod5-file-format/test_data/multi_fast5_zip_v3.pod5"
Enter
Sleep 2s

Type "# Extract just read IDs"
Enter
Sleep 500ms

Type "podfive view ext/pod5-file-format/test_data/multi_fast5_zip_v3.pod5 --ids"
Enter
Sleep 2s

Type "# Inspect file metadata"
Enter
Sleep 500ms

Type "podfive inspect summary ext/pod5-file-format/test_data/multi_fast5_zip_v3.pod5"
Enter
Sleep 2s

Type "# List available commands"
Enter
Sleep 500ms

Type "podfive --help"
Enter
Sleep 3s
