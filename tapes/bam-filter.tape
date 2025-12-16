# VHS tape for podfive bam-filter command
Output docs/images/bam-filter.gif

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

Type "# Filter POD5 reads using a paired BAM file"
Enter
Sleep 500ms

Type "# Keep only mapped reads with quality >= 10"
Enter
Sleep 500ms

Type "podfive bam-filter data/drna/yeast_trna_reads.pod5 -b data/drna/yeast_trna_mappings.bam --mapped -q 10 -o /tmp/filtered.pod5"
Enter
Sleep 5s

Type "# Check the result"
Enter
Sleep 500ms

Type "podfive inspect summary /tmp/filtered.pod5"
Enter
Sleep 3s
