# Deployment Process

All services deploy through the shared delivery pipeline. Manual deploys to
production are not permitted.

## Standard rollout

Merging to the `main` branch triggers a build, the full test suite, and an
automatic deploy to staging. Production rollouts are canary-based: the new
version first serves 5 percent of traffic for 30 minutes while error rate and
latency dashboards are watched automatically. If the canary stays healthy,
the rollout proceeds to 100 percent in two steps.

## Rolling back

To roll back a bad deployment, run `helios deploy rollback <service>` which
redeploys the previous known-good release within minutes. Rollback first,
debug second: never attempt a fix-forward while users are impacted. After a
rollback, open an incident review issue and link the offending release.

## Deploy freezes

Deploys are frozen during the last week of each quarter and during declared
incidents. The release calendar in the handbook lists current freeze windows.
