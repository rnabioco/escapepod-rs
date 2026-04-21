# VHS tape for escpod bam-filter command
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

Type "escpod bam-filter data/drna/yeast_trna_reads.pod5 \"
Enter
Type "  -b data/drna/yeast_trna_mappings.bam \"
Enter
Type "  --mapped --quality 10 \"
Enter
Type "  -o /tmp/filtered.pod5"
Enter
Sleep 5s

Type "# Check the result"
Enter
Sleep 500ms

Type "escpod inspect summary /tmp/filtered.pod5"
Enter
Sleep 3s
