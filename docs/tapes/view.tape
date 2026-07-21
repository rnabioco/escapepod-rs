# VHS tape for escpod view command
Output docs/images/view.gif

Set Shell "bash"
Set FontFamily "Source Code Pro"
Set FontSize 16
Set Width 1100
Set Height 480
Set LetterSpacing 0
Set LineHeight 1.2
Set Padding 20
Set Theme "Dracula"
Set WindowBar Colorful
Set Margin 20
Set BorderRadius 10

# 180 reads in this file — cap the rows so the command stays on screen with its
# output (SIGPIPE is handled, so the closed `head` pipe doesn't error). A trimmed
# column set keeps each row on a single line at this width instead of wrapping.
Type "escpod view data/drna/yeast_trna_reads.pod5 --include read_id,channel,num_samples,end_reason | head -8"
Enter
Sleep 5s
