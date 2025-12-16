# VHS tape file for podfive-rs demo
# Run with: vhs demo.tape

Output docs/images/readme.gif

Set Shell "bash"
Set FontFamily "FiraCode Nerd Font Mono"
Set FontSize 16
Set Width 900
Set Height 600
Set LetterSpacing 0
Set LineHeight 1.2
Set Padding 20
Set Theme "Dracula"
Set WindowBar Colorful
Set Margin 20
Set BorderRadius 10

Type "# podfive-rs - A fast Rust CLI for POD5 files"
Enter
Sleep 1s

Type "# Let's explore a POD5 file"
Enter
Sleep 500ms

Type "podfive summary data/drna/yeast_trna_reads.pod5"
Enter
Sleep 3s

Type "# View reads as a table"
Enter
Sleep 500ms

Type "podfive view data/drna/yeast_trna_reads.pod5"
Enter
Sleep 2s

Type "# Extract just read IDs"
Enter
Sleep 500ms

Type "podfive view data/drna/yeast_trna_reads.pod5 --ids"
Enter
Sleep 2s

Type "# Inspect file metadata"
Enter
Sleep 500ms

Type "podfive inspect summary data/drna/yeast_trna_reads.pod5"
Enter
Sleep 2s

Type "# List available commands"
Enter
Sleep 500ms

Type "podfive --help"
Enter
Sleep 5s
