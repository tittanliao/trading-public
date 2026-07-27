# Public export security review

Reviewed: 2026-07-14 (Asia/Taipei)

- Legacy ignored credential configuration was not read, copied, or published.
- `xauusd/pipeline/settings.py` reads credentials only from environment variables;
  `.env.example` contains names and empty values only.
- Public regular files contain no CSV, DOC/DOCX, XLS/XLSX, SQLite database, private
  key, or GitHub-token-format material.
- The AWS-key-format scanner reports one sequence in
  `tx/TX-Long-Experiments/report.html`. Manual context review confirms it is an
  accidental substring inside an embedded PNG base64 data URI, not a credential.
- Generic token-assignment matches in the pipeline are runtime variables or
  environment lookups, not literal secrets. Workflow `id-token: write` entries are
  GitHub Actions permissions.

Local CSV compatibility paths are ignored symbolic links to the sibling Private
repository and are never included in the Public commit or GitHub Pages artifact.
