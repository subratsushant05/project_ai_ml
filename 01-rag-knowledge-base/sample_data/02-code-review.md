# Code Review Guidelines

Code review keeps quality high and spreads knowledge across the team.

## Review turnaround

Reviewers are expected to respond to a pull request within one business day.
If you cannot review in time, reassign the request rather than letting it sit.
Small pull requests under 300 changed lines get reviewed fastest, so split
large changes into stacked pull requests.

## Approval rules

Every change to a production service needs approval from at least one code
owner of the affected directory. Changes touching shared libraries need two
approvals. Authors may not approve their own pull requests.

## What reviewers look for

Reviewers check correctness first, then readability, test coverage, and
operational concerns such as logging and metrics. Style nits should be
enforced by linters, not humans; if a nit is not caught by tooling, propose a
lint rule instead of blocking the review.
