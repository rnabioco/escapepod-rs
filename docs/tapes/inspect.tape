# VHS tape for escpod inspect command
Output docs/images/inspect.gif

Set Shell "bash"
Set FontFamily "FiraCode Nerd Font Mono"
Set FontSize 16
Set Width 900
Set Height 500
Set LetterSpacing 0
Set LineHeight 1.2
Set Padding 20
Set Theme "Dracula"
Set WindowBar Colorful
Set Margin 20
Set BorderRadius 10

Type "# List all read IDs in the file"
Enter
Sleep 500ms

Type "escpod inspect reads data/drna/yeast_trna_reads.pod5 | head -5"
Enter
Sleep 2s

Type "# Show details for a single read"
Enter
Sleep 500ms

Type "escpod inspect read data/drna/yeast_trna_reads.pod5 \"
Enter
Type "  36d401b5-ba9c-4862-b003-7cd309ce5281"
Enter
Sleep 3s
