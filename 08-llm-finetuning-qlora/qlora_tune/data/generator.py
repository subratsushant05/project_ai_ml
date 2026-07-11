"""Deterministic synthetic IT-helpdesk instruction dataset generator.

Generates ticket -> resolution pairs across five categories. Generation is
fully deterministic for a given seed, which makes the dataset reproducible
and the pipeline unit-testable. A configurable fraction of tickets embeds
fake contact details (email/phone) so the PII-scrubbing stage has real work
to do in the demo.
"""

from __future__ import annotations

import logging
import random

from qlora_tune.data.records import Example

logger = logging.getLogger(__name__)

CATEGORIES: tuple[str, ...] = (
    "password_reset",
    "vpn",
    "hardware",
    "software_install",
    "access_request",
)

_FIRST_NAMES = ["Alex", "Priya", "Jordan", "Mei", "Sam", "Ravi", "Dana", "Lena", "Omar", "Kai"]
_TEAMS = ["finance", "marketing", "engineering", "sales", "HR", "legal", "support", "design"]
_OS = ["Windows 11", "macOS Sonoma", "Ubuntu 22.04", "Windows 10"]
_URGENCY = [
    "This is blocking my work.",
    "I have a deadline today, please help.",
    "No rush, but I'd like this fixed this week.",
    "This has been happening since yesterday.",
]

# Per-category (ticket_template, resolution_template) pairs. Slot values are
# drawn deterministically from the pools above.
_TICKETS: dict[str, list[tuple[str, str]]] = {
    "password_reset": [
        (
            "I forgot my password for the {system} portal and I'm locked out "
            "after too many attempts. {urgency}",
            "1. Verified the requester's identity via manager approval.\n"
            "2. Unlocked the {system} account in the identity console.\n"
            "3. Issued a one-time reset link valid for 15 minutes.\n"
            "4. Confirmed the user set a new password meeting policy and can sign in.",
        ),
        (
            "My {system} password expired while I was on leave and the "
            "self-service reset page rejects my security answers. {urgency}",
            "1. Confirmed identity through a video call and employee ID check.\n"
            "2. Cleared the stale security answers on the {system} profile.\n"
            "3. Triggered a temporary password with forced change at next login.\n"
            "4. Advised enrolling in the authenticator app to avoid future lockouts.",
        ),
        (
            "I keep getting 'invalid credentials' on {system} even though I "
            "reset my password an hour ago. {urgency}",
            "1. Checked replication status; the new credential had not synced to {system}.\n"
            "2. Forced a directory sync and invalidated cached sessions.\n"
            "3. Asked the user to wait five minutes and sign in again.\n"
            "4. Verified successful login and closed the ticket.",
        ),
    ],
    "vpn": [
        (
            "The VPN client on my {os} laptop disconnects every few minutes "
            "when I work from home. {urgency}",
            "1. Collected VPN client logs and confirmed MTU-related drops.\n"
            "2. Lowered the tunnel MTU to 1350 in the client profile.\n"
            "3. Updated the client to the current supported version on {os}.\n"
            "4. Monitored a 2-hour session with no disconnects; resolved.",
        ),
        (
            "I can connect to the VPN but I cannot reach the internal "
            "{system} site; the browser times out. {urgency}",
            "1. Verified the tunnel was up and split-tunnel routes were applied.\n"
            "2. Found the {system} subnet missing from the user's VPN group policy.\n"
            "3. Added the route to the group policy and had the user reconnect.\n"
            "4. Confirmed the {system} site loads over the tunnel.",
        ),
        (
            "VPN setup fails on my new {os} machine with error 'certificate "
            "not trusted'. {urgency}",
            "1. Confirmed the corporate root CA was missing from the {os} trust store.\n"
            "2. Pushed the CA certificate via device management.\n"
            "3. Re-imported the VPN profile and re-authenticated.\n"
            "4. Verified a stable connection and access to internal resources.",
        ),
    ],
    "hardware": [
        (
            "My laptop battery drains from full to empty in about an hour "
            "and the chassis gets hot near the fan. {urgency}",
            "1. Ran hardware diagnostics; battery health reported at 41%.\n"
            "2. Cleaned fan vents and updated power-management firmware.\n"
            "3. Ordered a replacement battery under warranty.\n"
            "4. Swapped the battery and verified normal runtime and temperatures.",
        ),
        (
            "The external monitor at my desk flickers and sometimes shows "
            "'no signal' until I replug the cable. {urgency}",
            "1. Reproduced the flicker and tested with a known-good cable.\n"
            "2. Identified a worn DisplayPort cable as the cause.\n"
            "3. Replaced the cable and reseated the dock firmware update.\n"
            "4. Confirmed a stable image across sleep/wake cycles.",
        ),
        (
            "My {os} laptop will not power on; the charging LED blinks "
            "three times and nothing appears on screen. {urgency}",
            "1. Performed a hard reset by draining residual power.\n"
            "2. Reseated the RAM modules; the blink code indicated a memory fault.\n"
            "3. Replaced the faulty DIMM from spares stock.\n"
            "4. Booted successfully and ran a full memory test with no errors.",
        ),
    ],
    "software_install": [
        (
            "Please install {app} on my {os} workstation; the self-service "
            "store says 'not entitled'. {urgency}",
            "1. Checked licensing; the user's team had no {app} entitlement.\n"
            "2. Requested a license seat under the {team} cost center and got approval.\n"
            "3. Assigned the entitlement and pushed {app} via the software center.\n"
            "4. Verified the install launches and activates correctly on {os}.",
        ),
        (
            "The {app} update keeps failing at 90% with error code 0x8007 "
            "on my {os} machine. {urgency}",
            "1. Reviewed installer logs; a locked file from a stale process blocked the update.\n"
            "2. Ended the stale {app} background process and cleared the update cache.\n"
            "3. Re-ran the deployment from the software center.\n"
            "4. Confirmed {app} updated to the latest version and opens normally.",
        ),
        (
            "I need {app} for a {team} project but installation requires "
            "admin rights I don't have. {urgency}",
            "1. Confirmed {app} is on the approved software list for {team}.\n"
            "2. Packaged the installer for per-user deployment (no admin rights needed).\n"
            "3. Published it to the user's self-service portal.\n"
            "4. User installed successfully; verified version and license status.",
        ),
    ],
    "access_request": [
        (
            "I moved to the {team} team and need access to the {system} "
            "shared drive and dashboards. {urgency}",
            "1. Validated the transfer with the {team} manager per access policy.\n"
            "2. Added the user to the {team}-{system} security group.\n"
            "3. Removed group memberships from the previous role (least privilege).\n"
            "4. User confirmed the drive and dashboards are accessible.",
        ),
        (
            "Requesting read access to the {system} production logs to "
            "debug an issue for {team}. {urgency}",
            "1. Confirmed a valid business justification and manager approval.\n"
            "2. Granted time-boxed read-only access to {system} logs (14 days).\n"
            "3. Documented the grant in the access register with an expiry reminder.\n"
            "4. Verified the user can query logs; access will auto-revoke on expiry.",
        ),
        (
            "A new contractor on {team} needs an account with access to "
            "{system} starting Monday. {urgency}",
            "1. Verified the signed contractor agreement and sponsor approval.\n"
            "2. Created a contractor account with an end date matching the contract.\n"
            "3. Granted the standard {team} baseline plus {system} role only.\n"
            "4. Sent onboarding credentials via the secure channel; sponsor confirmed.",
        ),
    ],
}

_SYSTEMS = ["Jira", "Confluence", "SAP", "Salesforce", "GitLab", "Tableau", "the intranet"]
_APPS = ["Visual Studio Code", "Adobe Acrobat", "Slack", "Docker Desktop", "MATLAB", "Figma"]


def _fake_contact(rng: random.Random) -> str:
    """Return a deterministic fake contact sentence containing PII."""
    name = rng.choice(_FIRST_NAMES).lower()
    email = f"{name}.{rng.randint(10, 99)}@example-corp.com"
    phone = f"+1-555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
    return f" You can reach me at {email} or {phone}."


def generate_dataset(
    n_per_category: int = 60,
    seed: int = 13,
    pii_fraction: float = 0.2,
) -> list[Example]:
    """Generate a deterministic synthetic helpdesk instruction dataset.

    Args:
        n_per_category: Number of examples per category (5 categories total).
        seed: RNG seed; the same seed always yields the same dataset.
        pii_fraction: Fraction of tickets that embed fake email/phone PII,
            exercising the downstream scrubbing stage.

    Returns:
        List of :class:`Example`, ``5 * n_per_category`` items, ordered by
        category then index.
    """
    rng = random.Random(seed)
    examples: list[Example] = []
    counter = 0
    for category in CATEGORIES:
        pairs = _TICKETS[category]
        for i in range(n_per_category):
            ticket_tpl, resolution_tpl = pairs[i % len(pairs)]
            slots = {
                "system": rng.choice(_SYSTEMS),
                "app": rng.choice(_APPS),
                "os": rng.choice(_OS),
                "team": rng.choice(_TEAMS),
                "urgency": rng.choice(_URGENCY),
            }
            instruction = ticket_tpl.format(**slots)
            if rng.random() < pii_fraction:
                instruction += _fake_contact(rng)
            response = resolution_tpl.format(**slots)
            counter += 1
            examples.append(
                Example(
                    id=f"TCK-{counter:05d}",
                    category=category,
                    instruction=instruction,
                    response=response,
                    meta={"variant": i % len(pairs)},
                )
            )
    logger.info("Generated %d synthetic examples (seed=%d)", len(examples), seed)
    return examples
