# Publishing this package

1. Read `README.md`, `reports/` and `docs/MISSING_FILES.md`. The original source-stage notebook is now included; decide whether to release this package with its tested result-level scope or first validate a complete raw-data replay.
2. Confirm the author name, affiliation and ORCID. Select an explicit software license and determine the terms for derived result tables. Do not apply that license to third-party datasets.
3. Create a repository in your GitHub account, for example `paper3-feature-stability-transfer`. This is a proposed name, not an existing URL.
4. Upload the **contents of** `Paper3_GitHub_Repository/`, preserving subfolders. Do not upload only the ZIP as the repository's code.
5. If using Git locally, inspect `git status` before committing. `.gitignore` excludes generated `outputs/`, caches, model binaries and raw-data archives. The small verification summaries in `reports/` remain tracked.
6. Run the three quick-start commands again after download/clone. Review any failed numeric checks rather than replacing expected values.
7. Create a versioned release after the contents are complete. If a permanent archival DOI is desired, archive the actual release with a service such as Zenodo; this package does not claim one has been created.
8. Send the real repository/release URL to update the manuscript's code-availability statement.

A public code-availability statement must match the package scope. For a result-level release, say it provides protocols, saved results, verification and reanalysis code; do not claim complete raw-data reproduction. The manuscript draft's highlighted availability placeholder remains unresolved until a real access URL exists.
