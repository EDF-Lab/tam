# Signing off your commits (DCO)

The sign-off says you wrote the change, or have the right to submit it, under the licence of the project ([DCO](../../DCO)). It is a line at the end of the commit message.

## New commits

```bash
git commit -s -m "fix: short phrase"
```

Use the name and email set in `git config user.name` and `git config user.email`; they must match your commits.

## The last commit lacks it

```bash
git commit --amend -s --no-edit
git push --force-with-lease
```

## Several commits of the branch lack it

```bash
git rebase --signoff main
git push --force-with-lease
```

Replace `main` by the branch your pull request targets. Only force-push your own feature branch.

## The check still fails

Open the `DCO` check of the pull request: it lists the commits without a sign-off and gives the commands above for them.
