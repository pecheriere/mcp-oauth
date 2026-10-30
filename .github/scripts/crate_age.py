"""Fail a PR whose Cargo.lock brings in a crate version published less than 7 days ago.

Compares the registry packages in Cargo.lock at a base revision with the working
tree (or a head revision). Every crates.io version that is new or changed is looked
up on crates.io, and one published less than 7 days ago fails the check unless
.github/crate-age-allow.toml lists that exact version with the RUSTSEC or GitHub
(GHSA) advisory it fixes. An allow-list entry whose version is no longer in Cargo.lock is stale and
fails the check too. With --head, Cargo.lock and the allow-list are both read at
that revision; with --no-base, every version counts as new.

Versions that did not change are never looked up, so a PR that leaves Cargo.lock
alone makes no network request. A lookup that fails fails the check.

Usage: crate_age.py (--base REV | --no-base) [--head REV]
"""

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO

ROOT = Path(__file__).resolve().parent.parent.parent
LOCKFILE = "Cargo.lock"
ALLOW_LIST = ".github/crate-age-allow.toml"
MIN_AGE = timedelta(days=7)
# Cargo records crates.io under this source whichever index protocol it used.
CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"
API = "https://crates.io/api/v1/crates"
# crates.io asks crawlers for at most one request per second.
REQUEST_INTERVAL = 1.0
TIMEOUT = 30
MAX_RESPONSE_BYTES = 1 << 20
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
# A RustSec id, or a GitHub advisory id (GHSA-xxxx-xxxx-xxxx in its own base32
# alphabet) for advisories RustSec does not carry.
ADVISORY = re.compile(r"\b(?:RUSTSEC-\d{4}-\d{4}|GHSA(?:-[23456789cfghjmpqrvwx]{4}){3})\b")
VERSION = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?")


class PublishDateError(Exception):
    """The publish date of a crate version could not be established."""


@dataclass(frozen=True, order=True)
class Crate:
    """One version of one crate."""

    name: str
    version: str

    def __str__(self) -> str:
        """Return e.g. `serde 1.0.1`."""
        return f"{self.name} {self.version}"


Lookup = Callable[[Crate], datetime]


def _crate(name: object, version: object) -> Crate:
    """Return the crate, refusing names and versions crates.io would not accept."""
    if not isinstance(name, str) or not NAME.fullmatch(name):
        msg = f"invalid crate name {name!r}"
        raise ValueError(msg)
    if not isinstance(version, str) or not VERSION.fullmatch(version):
        msg = f"invalid version {version!r} of crate {name}"
        raise ValueError(msg)
    return Crate(name, version)


def _toml(text: str, what: str) -> dict[str, object]:
    """Parse `text`, naming `what` in the error."""
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as err:
        msg = f"{what} is not valid TOML: {err}"
        raise ValueError(msg) from err


def registry_crates(lock_text: str) -> set[Crate]:
    """Return the crates.io packages of a Cargo.lock.

    Workspace members have no source and are skipped, as are git sources, which
    cargo-deny refuses. A package from any other registry is an error: its publish
    date cannot be checked.
    """
    packages = _toml(lock_text, LOCKFILE).get("package", [])
    if not isinstance(packages, list):
        msg = f"{LOCKFILE}: `package` is not an array of tables"
        raise TypeError(msg)
    crates: set[Crate] = set()
    for package in packages:
        if not isinstance(package, dict):
            msg = f"{LOCKFILE}: a `package` entry is not a table"
            raise TypeError(msg)
        source = package.get("source")
        if source is None or (isinstance(source, str) and source.startswith("git+")):
            continue
        crate = _crate(package.get("name"), package.get("version"))
        if source != CRATES_IO:
            msg = f"{crate} comes from {source!r}, which is not crates.io"
            raise ValueError(msg)
        crates.add(crate)
    return crates


def load_allow_list(text: str) -> dict[Crate, str]:
    """Return the allowed crate versions and the reason for each."""
    data = _toml(text, "allow-list")
    if set(data) - {"allow"}:
        msg = "allow-list: only [[allow]] entries are allowed"
        raise ValueError(msg)
    entries = data.get("allow", [])
    if not isinstance(entries, list):
        msg = "allow-list: `allow` is not an array of tables"
        raise TypeError(msg)
    allowed: dict[Crate, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "version", "reason"}:
            msg = "allow-list: each [[allow]] entry needs exactly name, version and reason"
            raise ValueError(msg)
        crate = _crate(entry["name"], entry["version"])
        reason = entry["reason"]
        if not isinstance(reason, str) or not ADVISORY.search(reason):
            msg = (
                f"allow-list: the reason for {crate} must name the advisory it fixes "
                "(RUSTSEC-... or GHSA-...)"
            )
            raise ValueError(msg)
        if crate in allowed:
            msg = f"allow-list: {crate} is listed twice"
            raise ValueError(msg)
        allowed[crate] = reason
    return allowed


def check(
    base_lock: str | None,
    head_lock: str,
    allow_text: str,
    lookup: Lookup,
    now: datetime,
) -> list[str]:
    """Return one error per problem; an empty list means the PR passes.

    `base_lock` is None when the base has no Cargo.lock, so every version is new.
    """
    head = registry_crates(head_lock)
    base = registry_crates(base_lock) if base_lock is not None else set()
    allowed = load_allow_list(allow_text)

    errors: list[str] = []
    for crate in sorted(head - base):
        if crate in allowed:
            continue
        try:
            published = lookup(crate)
        except PublishDateError as err:
            errors.append(
                f"{crate}: could not get its publish date from crates.io, so the check "
                f"fails rather than pass it unchecked ({err}). Re-run the job.",
            )
            continue
        if now - published < MIN_AGE:
            errors.append(
                f"{crate} was published on {published:%Y-%m-%d}, less than "
                f"{MIN_AGE.days} days ago. Wait until {published + MIN_AGE:%Y-%m-%d} or, "
                "for the oldest release fixing a security advisory, add an exception to "
                ".github/crate-age-allow.toml (see the README).",
            )
    errors.extend(
        f"allow-list entry {crate} is stale: that version is not in {LOCKFILE}. Remove it."
        for crate in sorted(set(allowed) - head)
    )
    return errors


Opener = Callable[..., IO[bytes]]


class CratesIo:
    """Look up publish dates on crates.io, at most one request per second."""

    def __init__(
        self,
        user_agent: str,
        *,
        opener: Opener = urllib.request.urlopen,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Send `user_agent`, which crates.io requires to identify the caller."""
        self._user_agent = user_agent
        self._opener = opener
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def __call__(self, crate: Crate) -> datetime:
        """Return when `crate` was published."""
        if self._last is not None:
            wait = self._last + REQUEST_INTERVAL - self._clock()
            if wait > 0:
                self._sleep(wait)
        name = urllib.parse.quote(crate.name, safe="")
        version = urllib.parse.quote(crate.version, safe="")
        # The URL is always the https crates.io API; only the escaped path varies.
        request = urllib.request.Request(  # noqa: S310
            f"{API}/{name}/{version}",
            headers={"User-Agent": self._user_agent, "Accept": "application/json"},
        )
        try:
            with self._opener(request, timeout=TIMEOUT) as reply:
                body = reply.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, http.client.HTTPException) as err:
            msg = f"request failed: {err!r}"
            raise PublishDateError(msg) from err
        finally:
            self._last = self._clock()
        if len(body) > MAX_RESPONSE_BYTES:
            msg = f"response larger than {MAX_RESPONSE_BYTES} bytes"
            raise PublishDateError(msg)
        return _published(body, crate)


def _published(body: bytes, crate: Crate) -> datetime:
    """Return `version.created_at` from a crates.io version response for `crate`."""
    try:
        data = json.loads(body)
    except (ValueError, RecursionError) as err:
        msg = f"response is not JSON: {err}"
        raise PublishDateError(msg) from err
    version = data.get("version") if isinstance(data, dict) else None
    if (
        not isinstance(version, dict)
        or version.get("crate") != crate.name
        or version.get("num") != crate.version
    ):
        msg = "response is not about this crate version"
        raise PublishDateError(msg)
    created_at = version.get("created_at")
    try:
        published = datetime.fromisoformat(created_at) if isinstance(created_at, str) else None
    except ValueError:
        published = None
    if published is None or published.tzinfo is None:
        msg = f"unexpected created_at {created_at!r}"
        raise PublishDateError(msg)
    return published


def _git(*args: str) -> str:
    """Run git in the repository and return its output."""
    try:
        return subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout
    except subprocess.CalledProcessError as err:
        msg = f"git {' '.join(args)} failed: {err.stderr.strip()}"
        raise ValueError(msg) from err


def file_at(rev: str, path: str) -> str | None:
    """Return the file at `path` as it is at git `rev`, or None if `rev` has none."""
    if rev.startswith("-"):
        msg = f"invalid revision {rev!r}"
        raise ValueError(msg)
    try:
        _git("rev-parse", "--verify", f"{rev}^{{commit}}")
    except ValueError as err:
        msg = f"unknown revision {rev!r}"
        raise ValueError(msg) from err
    if not _git("ls-tree", "--name-only", rev, "--", path).strip():
        return None
    return _git("show", f"{rev}:{path}")


def head_files(rev: str | None) -> tuple[str, str]:
    """Return Cargo.lock and the allow-list at git `rev`, or in the working tree if None.

    A revision from before the allow-list existed allows nothing.
    """
    if rev is None:
        lock = (ROOT / LOCKFILE).read_text(encoding="utf-8")
        return lock, (ROOT / ALLOW_LIST).read_text(encoding="utf-8")
    lock = file_at(rev, LOCKFILE)
    if lock is None:
        msg = f"{rev} has no {LOCKFILE}"
        raise ValueError(msg)
    return lock, file_at(rev, ALLOW_LIST) or ""


def user_agent() -> str:
    """Identify this check to crates.io, with the repository when run in Actions."""
    repository = os.environ.get("GITHUB_REPOSITORY")
    if repository is None:
        return "mcp-oauth crate-age check (local run)"
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    return f"mcp-oauth crate-age check ({server}/{repository})"


def annotation(message: str) -> str:
    """Return `message` as a GitHub error annotation on Cargo.lock."""
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return f"::error file={LOCKFILE},title=Crate age::{escaped}"


def main(argv: list[str] | None = None) -> int:
    """Run the check and print its errors; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base_group = parser.add_mutually_exclusive_group(required=True)
    base_group.add_argument("--base", help="git revision to compare against")
    base_group.add_argument("--no-base", action="store_true", help="check every version")
    parser.add_argument("--head", help="git revision to check (default: the working tree)")
    args = parser.parse_args(argv)

    try:
        base = None if args.no_base else file_at(args.base, LOCKFILE)
        head, allow_text = head_files(args.head)
        errors = check(base, head, allow_text, CratesIo(user_agent()), datetime.now(UTC))
    except (ValueError, TypeError, OSError) as err:
        errors = [str(err)]

    for error in errors:
        print(annotation(error))
    if errors:
        return 1
    print(f"Every new or changed crate version is at least {MIN_AGE.days} days old or allowed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
