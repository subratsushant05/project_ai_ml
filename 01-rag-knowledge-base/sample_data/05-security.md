# Security Practices

Security is part of everyday engineering work, not a separate phase.

## Secrets management

Secrets never live in source code or CI variables. All credentials are stored
in the central vault and injected at deploy time. If you suspect a secret has
leaked, rotate it immediately and declare a SEV2 incident.

## Dependencies

Dependency updates are automated; the update bot opens weekly pull requests.
Security patches flagged as critical must be merged within two business days.
New third-party dependencies require a short written justification in the
pull request description.

## Access control

Production access follows least privilege. Engineers get read access to logs
and dashboards by default; write access to production data requires a
time-boxed access grant approved by the service owner. Access grants expire
automatically after 24 hours.
