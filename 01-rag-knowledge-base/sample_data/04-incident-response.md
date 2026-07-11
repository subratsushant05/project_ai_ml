# Incident Response

Incidents are inevitable; chaotic responses are not.

## Declaring an incident

Anyone can declare an incident by running `/incident declare` in the
`#eng-incidents` channel. When a production incident is declared, the first
step is to assign an incident commander; the person who declared it acts as
commander until someone else explicitly takes over. The commander coordinates
communication and delegates investigation, and does not debug the issue
personally.

## Severity levels

SEV1 means customer-facing functionality is broken for many users and pages
the on-call engineer immediately. SEV2 means degraded service or a broken
internal tool with a workaround. SEV3 covers minor issues that can wait for
business hours.

## After the incident

Every SEV1 and SEV2 incident gets a blameless post-incident review within
five business days. The review focuses on contributing causes and follow-up
actions, never on individual blame. Action items are tracked in the incident
backlog and reviewed monthly.
