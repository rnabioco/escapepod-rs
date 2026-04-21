# VHS tape for escpod filter command
Output docs/images/filter.gif

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

Type "escpod view data/drna/yeast_trna_reads.pod5 --ids | head -2 > /tmp/ids.txt"
Enter
Sleep 1s

Type "cat /tmp/ids.txt"
Enter
Sleep 2s

Type "escpod filter data/drna/yeast_trna_reads.pod5 -i /tmp/ids.txt -o /tmp/filtered.pod5"
Enter
Sleep 5s
