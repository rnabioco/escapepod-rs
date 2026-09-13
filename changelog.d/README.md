# changelog.d/

News fragments, one file per change, so parallel branches never collide by
all editing `CHANGELOG.md`'s `## Unreleased` section at once.

## Filename

```
changelog.d/<branch-or-issue-slug>.<category>.md
```

`category` is one of the `## Unreleased` subheadings this repo's
`CHANGELOG.md` uses, lowercased: `added`, `changed`, `fixed`, `breaking`.

## Content

Just the bullet(s) that would go under that heading — same style as the
rest of `CHANGELOG.md` (bold one-line headline for a user-facing summary,
then the cause/detail). No frontmatter, no heading line of your own.

## Landing a PR

Add your fragment file in your branch alongside your code change. Do **not**
edit `CHANGELOG.md` directly — that's the whole point of this directory.

## Rolling up

At release time (or whenever consolidating), collect every fragment's
content under its category's heading in `CHANGELOG.md`'s `## Unreleased`
section, in the order the fragments landed, then delete the fragment files
in the same commit.
