# VHS tape for escpod bam-filter command
Output docs/images/bam-filter.gif

Set Shell "bash"
Set FontFamily "Source Code Pro"
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

# Reset any output from a previous render so escpod doesn't refuse to overwrite
Hide
Type "rm -f /tmp/bam-filtered.pod5"
Enter
Show

Type 'escpod bam-filter data/drna/yeast_trna_reads.pod5 \'
Enter
Type '  -b data/drna/yeast_trna_mappings.bam \'
Enter
Type '  --mapped --quality 10 \'
Enter
Type '  -o /tmp/bam-filtered.pod5'
Enter
Sleep 4s
Ctrl+L

Type "escpod inspect summary /tmp/bam-filtered.pod5"
Enter
Sleep 4s
