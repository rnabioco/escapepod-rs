# VHS tape file for escpod-rs demo
# Run with: vhs readme.tape

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

Type "escpod summary data/drna/yeast_trna_reads.pod5"
Enter
Sleep 4s
Ctrl+L

Type "escpod view data/drna/yeast_trna_reads.pod5"
Enter
Sleep 3s
Ctrl+L

Type "escpod view data/drna/yeast_trna_reads.pod5 --ids"
Enter
Sleep 3s
Ctrl+L

Type "escpod inspect summary data/drna/yeast_trna_reads.pod5"
Enter
Sleep 3s
Ctrl+L

Hide
Type "escpod view data/drna/yeast_trna_reads.pod5 --ids | head -2 > /tmp/ids.txt"
Enter
Sleep 500ms
Show

Type "escpod filter data/drna/yeast_trna_reads.pod5 -i /tmp/ids.txt -o /tmp/filtered.pod5"
Enter
Sleep 4s
Ctrl+L

Type "escpod bam-filter data/drna/yeast_trna_reads.pod5 -b data/drna/yeast_trna_mappings.bam --mapped -q 10 -o /tmp/filtered.pod5"
Enter
Sleep 4s
Ctrl+L

Type "escpod --help"
Enter
Sleep 5s
