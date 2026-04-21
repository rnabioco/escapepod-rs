# VHS tape for escpod subset command
Output docs/images/subset.gif

Set Shell "bash"
Set FontFamily "FiraCode Nerd Font Mono"
Set FontSize 16
Set Width 900
Set Height 550
Set LetterSpacing 0
Set LineHeight 1.2
Set Padding 20
Set Theme "Dracula"
Set WindowBar Colorful
Set Margin 20
Set BorderRadius 10

Type "# Create a CSV mapping file"
Enter
Sleep 500ms

Type "echo 'read_id,output' > /tmp/mapping.csv"
Enter
Sleep 300ms

Type "escpod view data/drna/yeast_trna_reads.pod5 --ids \"
Enter
Type "  | head -1 \"
Enter
Type "  | xargs -I{} echo '{},sample1.pod5' >> /tmp/mapping.csv"
Enter
Sleep 500ms

Type "cat /tmp/mapping.csv"
Enter
Sleep 1s

Type "# Subset reads based on mapping"
Enter
Sleep 500ms

Type "escpod subset data/drna/yeast_trna_reads.pod5 \"
Enter
Type "  --csv /tmp/mapping.csv \"
Enter
Type "  -o /tmp/subsets/"
Enter
Sleep 5s
