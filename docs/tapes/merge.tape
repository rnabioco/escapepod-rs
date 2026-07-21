# VHS tape for escpod merge command
Output docs/images/merge.gif

Set Shell "bash"
Set FontFamily "Fira Code"
Set FontSize 16
Set Width 900
Set Height 320
Set LetterSpacing 0
Set LineHeight 1.2
Set Padding 20
Set Theme "Dracula"
Set WindowBar Colorful
Set Margin 20
Set BorderRadius 10

# Reset any output from a previous render so escpod doesn't refuse to overwrite
Hide
Type "rm -f /tmp/merged.pod5"
Enter
Ctrl+L
Show

Type "escpod merge data/drna/yeast_trna_reads.pod5 -o /tmp/merged.pod5"
Enter
Sleep 5s
