"""Unit tests for crate_age.py; none touch the network, and git only in temporary repos."""

import contextlib
import http.client
import io
import json
import os
import subprocess
import tempfile
import unittest
import urllib.error
from datetime import UTC, datetime, timedelta
from email.message import Message
from pathlib import Path
from unittest import mock

import crate_age
from crate_age import Crate, CratesIo, PublishDateError

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"


def lockfile(*packages: tuple[str, str, str | None]) -> str:
    """Build a Cargo.lock with one `[[package]]` per (name, version, source)."""
    out = ["version = 4\n"]
    for name, version, source in packages:
        out.append(f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n')
        if source is not None:
            out.append(f'source = "{source}"\nchecksum = "{"0" * 64}"\n')
    return "".join(out)


def allow_list(*entries: tuple[str, str, str]) -> str:
    """Build an allow-list with one `[[allow]]` per (name, version, reason)."""
    return "".join(
        f'[[allow]]\nname = "{name}"\nversion = "{version}"\nreason = "{reason}"\n\n'
        for name, version, reason in entries
    )


class FakeLookup:
    """Publish dates by crate; records which crates were looked up."""

    def __init__(
        self, dates: dict[Crate, datetime], failing: frozenset[Crate] = frozenset()
    ) -> None:
        """Answer from `dates`; raise PublishDateError for crates in `failing`."""
        self.dates = dates
        self.failing = failing
        self.calls: list[Crate] = []

    def __call__(self, crate: Crate) -> datetime:
        """Return the publish date of `crate`."""
        self.calls.append(crate)
        if crate in self.failing:
            msg = f"lookup of {crate} failed"
            raise PublishDateError(msg)
        return self.dates[crate]


OLD = NOW - timedelta(days=30)
NEW = NOW - timedelta(days=3)


class RegistryCratesTest(unittest.TestCase):
    """Reading registry packages out of a lockfile."""

    def test_keeps_crates_io_and_skips_path_and_git_packages(self) -> None:
        """Workspace members have no source and git sources are cargo-deny's job."""
        text = lockfile(
            ("serde", "1.0.0", CRATES_IO),
            ("tokio", "1.2.3", CRATES_IO),
            ("mcp-oauth", "0.3.0", None),
            ("forked", "0.1.0", "git+https://example.com/forked#abc"),
        )
        self.assertEqual(
            crate_age.registry_crates(text),
            {Crate("serde", "1.0.0"), Crate("tokio", "1.2.3")},
        )

    def test_rejects_other_registries(self) -> None:
        """A version from a registry other than crates.io cannot be dated, so it fails."""
        text = lockfile(("serde", "1.0.0", "registry+https://example.com/index"))
        with self.assertRaisesRegex(ValueError, "not crates.io"):
            crate_age.registry_crates(text)

    def test_rejects_names_and_versions_crates_io_would_not_accept(self) -> None:
        """Names and versions go into a URL and into annotations, so they are validated."""
        for name, version in (("../admin", "1.0.0"), ("serde", "1.0.0/../x"), ("a b", "1")):
            with self.subTest(name=name, version=version):
                text = lockfile((name, version, CRATES_IO))
                with self.assertRaisesRegex(ValueError, "invalid"):
                    crate_age.registry_crates(text)

    def test_rejects_a_lockfile_that_is_not_toml(self) -> None:
        """A broken lockfile is an error, not an empty set."""
        with self.assertRaises(ValueError):
            crate_age.registry_crates("[[package]\n")


class AllowListTest(unittest.TestCase):
    """Parsing and validating the allow-list."""

    def test_reads_entries(self) -> None:
        """Each entry maps a crate version to its reason."""
        text = allow_list(
            ("webpki-root-certs", "1.0.10", "Oldest release fixing RUSTSEC-2026-0383")
        )
        self.assertEqual(
            crate_age.load_allow_list(text),
            {Crate("webpki-root-certs", "1.0.10"): "Oldest release fixing RUSTSEC-2026-0383"},
        )

    def test_an_empty_file_allows_nothing(self) -> None:
        """No entries is valid."""
        self.assertEqual(crate_age.load_allow_list(""), {})

    def test_rejects_malformed_entries(self) -> None:
        """A reason without an advisory, unknown keys, wrong types and duplicates are errors."""
        cases = {
            "missing reason": '[[allow]]\nname = "a"\nversion = "1.0.0"\n',
            "blank reason": '[[allow]]\nname = "a"\nversion = "1.0.0"\nreason = "  "\n',
            "no advisory": allow_list(("a", "1.0.0", "Fixes a security bug")),
            "advisory id malformed": allow_list(("a", "1.0.0", "Fixes RUSTSEC-26-1")),
            "advisory id too long": allow_list(("a", "1.0.0", "Fixes RUSTSEC-2026-00011")),
            "advisory id prefixed": allow_list(("a", "1.0.0", "Fixes XRUSTSEC-2026-0001")),
            "unknown key": '[[allow]]\nname = "a"\nversion = "1"\nreason = "r"\nuntil = "x"\n',
            "not a string": '[[allow]]\nname = "a"\nversion = 1\nreason = "r"\n',
            "unknown table": '[other]\nname = "a"\n',
            "allow not a list": 'allow = "a"\n',
            "invalid name": allow_list(("a/b", "1.0.0", "Fixes RUSTSEC-2026-0001")),
            "duplicate": allow_list(
                ("a", "1.0.0", "Fixes RUSTSEC-2026-0001"),
                ("a", "1.0.0", "Fixes RUSTSEC-2026-0002"),
            ),
            "not toml": "[[allow]\n",
        }
        for case, text in cases.items():
            with self.subTest(case=case), self.assertRaises((ValueError, TypeError)):
                crate_age.load_allow_list(text)


class CheckTest(unittest.TestCase):
    """The age rule applied to a base and a head lockfile."""

    def test_unchanged_versions_are_not_looked_up(self) -> None:
        """A lockfile with no new registry versions needs no network at all."""
        text = lockfile(("serde", "1.0.0", CRATES_IO), ("mcp-oauth", "0.3.0", None))
        lookup = FakeLookup({})
        self.assertEqual(crate_age.check(text, text, "", lookup, NOW), [])
        self.assertEqual(lookup.calls, [])

    def test_removed_versions_are_not_looked_up(self) -> None:
        """Dropping a dependency never needs a date."""
        base = lockfile(("serde", "1.0.0", CRATES_IO), ("tokio", "1.0.0", CRATES_IO))
        head = lockfile(("serde", "1.0.0", CRATES_IO))
        lookup = FakeLookup({})
        self.assertEqual(crate_age.check(base, head, "", lookup, NOW), [])
        self.assertEqual(lookup.calls, [])

    def test_new_and_changed_versions_old_enough_pass(self) -> None:
        """Only versions absent from the base are looked up, and old ones pass."""
        base = lockfile(("serde", "1.0.0", CRATES_IO), ("tokio", "1.0.0", CRATES_IO))
        head = lockfile(
            ("serde", "1.0.1", CRATES_IO),
            ("tokio", "1.0.0", CRATES_IO),
            ("bytes", "1.5.0", CRATES_IO),
        )
        lookup = FakeLookup({Crate("serde", "1.0.1"): OLD, Crate("bytes", "1.5.0"): OLD})
        self.assertEqual(crate_age.check(base, head, "", lookup, NOW), [])
        self.assertEqual(lookup.calls, [Crate("bytes", "1.5.0"), Crate("serde", "1.0.1")])

    def test_a_new_version_younger_than_14_days_fails(self) -> None:
        """A crate added at a version published three days ago is refused, with its dates."""
        base = lockfile(("tokio", "1.0.0", CRATES_IO))
        head = lockfile(("tokio", "1.0.0", CRATES_IO), ("serde", "1.0.1", CRATES_IO))
        lookup = FakeLookup({Crate("serde", "1.0.1"): NEW})
        errors = crate_age.check(base, head, "", lookup, NOW)
        self.assertEqual(len(errors), 1)
        self.assertIn("serde 1.0.1", errors[0])
        self.assertIn("2026-10-26", errors[0])
        self.assertIn("2026-11-09", errors[0])

    def test_a_changed_version_younger_than_14_days_fails(self) -> None:
        """Changing an existing crate to a new version is checked like adding one."""
        base = lockfile(("serde", "1.0.0", CRATES_IO))
        head = lockfile(("serde", "1.0.1", CRATES_IO))
        lookup = FakeLookup({Crate("serde", "1.0.1"): NEW})
        self.assertEqual(len(crate_age.check(base, head, "", lookup, NOW)), 1)

    def test_exactly_14_days_old_passes(self) -> None:
        """The boundary is inclusive: 14 days old is old enough."""
        head = lockfile(("serde", "1.0.1", CRATES_IO))
        lookup = FakeLookup({Crate("serde", "1.0.1"): NOW - timedelta(days=14)})
        self.assertEqual(crate_age.check(lockfile(), head, "", lookup, NOW), [])

    def test_a_publish_date_in_the_future_fails(self) -> None:
        """A clock or API oddity is treated as too new, never as old enough."""
        head = lockfile(("serde", "1.0.1", CRATES_IO))
        lookup = FakeLookup({Crate("serde", "1.0.1"): NOW + timedelta(days=1)})
        self.assertEqual(len(crate_age.check(lockfile(), head, "", lookup, NOW)), 1)

    def test_no_base_lockfile_checks_every_version(self) -> None:
        """A base without Cargo.lock means every registry version is new."""
        head = lockfile(("serde", "1.0.0", CRATES_IO))
        lookup = FakeLookup({Crate("serde", "1.0.0"): OLD})
        self.assertEqual(crate_age.check(None, head, "", lookup, NOW), [])
        self.assertEqual(lookup.calls, [Crate("serde", "1.0.0")])

    def test_an_allow_listed_young_version_passes_without_a_lookup(self) -> None:
        """An exception covers its exact version and needs no network."""
        head = lockfile(("webpki-root-certs", "1.0.10", CRATES_IO))
        allowed = allow_list(
            ("webpki-root-certs", "1.0.10", "Oldest release fixing RUSTSEC-2026-0383")
        )
        lookup = FakeLookup({})
        self.assertEqual(crate_age.check(lockfile(), head, allowed, lookup, NOW), [])
        self.assertEqual(lookup.calls, [])

    def test_an_exception_does_not_cover_another_version(self) -> None:
        """An entry for 1.0.10 does not let 1.0.11 through."""
        head = lockfile(
            ("webpki-root-certs", "1.0.10", CRATES_IO),
            ("webpki-root-certs", "1.0.11", CRATES_IO),
        )
        allowed = allow_list(
            ("webpki-root-certs", "1.0.10", "Oldest release fixing RUSTSEC-2026-0383")
        )
        lookup = FakeLookup({Crate("webpki-root-certs", "1.0.11"): NEW})
        errors = crate_age.check(lockfile(), head, allowed, lookup, NOW)
        self.assertEqual(len(errors), 1)
        self.assertIn("webpki-root-certs 1.0.11", errors[0])

    def test_an_allow_list_entry_missing_from_the_lockfile_fails(self) -> None:
        """An exception whose version left Cargo.lock is stale and must be removed."""
        base = lockfile(("webpki-root-certs", "1.0.10", CRATES_IO))
        head = lockfile(("webpki-root-certs", "1.0.12", CRATES_IO))
        allowed = allow_list(
            ("webpki-root-certs", "1.0.10", "Oldest release fixing RUSTSEC-2026-0383")
        )
        lookup = FakeLookup({Crate("webpki-root-certs", "1.0.12"): OLD})
        errors = crate_age.check(base, head, allowed, lookup, NOW)
        self.assertEqual(len(errors), 1)
        self.assertIn("stale", errors[0])
        self.assertIn("webpki-root-certs 1.0.10", errors[0])

    def test_an_allow_list_entry_still_in_an_unchanged_lockfile_passes(self) -> None:
        """An exception that has aged out stays valid while its version is locked."""
        text = lockfile(("webpki-root-certs", "1.0.10", CRATES_IO))
        allowed = allow_list(
            ("webpki-root-certs", "1.0.10", "Oldest release fixing RUSTSEC-2026-0383")
        )
        self.assertEqual(crate_age.check(text, text, allowed, FakeLookup({}), NOW), [])

    def test_a_failed_lookup_fails_and_the_others_are_still_checked(self) -> None:
        """Lookup errors fail closed, and every problem is reported in one run."""
        head = lockfile(("bytes", "1.5.0", CRATES_IO), ("serde", "1.0.1", CRATES_IO))
        lookup = FakeLookup(
            {Crate("serde", "1.0.1"): NEW},
            failing=frozenset({Crate("bytes", "1.5.0")}),
        )
        errors = crate_age.check(lockfile(), head, "", lookup, NOW)
        self.assertEqual(len(errors), 2)
        self.assertIn("bytes 1.5.0", errors[0])
        self.assertIn("lookup of", errors[0])
        self.assertIn("serde 1.0.1", errors[1])


class BrokenReply(io.BytesIO):
    """A response whose body ends before its declared length."""

    def read(self, size: int | None = -1) -> bytes:
        """Fail the way http.client does on a truncated body."""
        partial = b"{"
        raise http.client.IncompleteRead(partial, size)


def response(body: object) -> io.BytesIO:
    """Return a file-like crates.io response carrying `body` as JSON."""
    return io.BytesIO(json.dumps(body).encode())


def version_body(name: str, num: str, created_at: str) -> dict[str, object]:
    """Return the part of a crates.io version response the lookup reads."""
    return {"version": {"crate": name, "num": num, "created_at": created_at}}


class CratesIoTest(unittest.TestCase):
    """The crates.io lookup, with the HTTP opener, clock and sleep injected."""

    def setUp(self) -> None:
        """Record requests and sleeps; the clock only advances when slept."""
        self.requests: list[tuple[str, str, float]] = []
        self.sleeps: list[float] = []
        self.now = 1000.0
        self.replies: list[io.BytesIO | Exception] = []

    def opener(self, request: crate_age.urllib.request.Request, timeout: float) -> io.BytesIO:
        """Return or raise the next queued reply."""
        self.requests.append((request.full_url, request.get_header("User-agent"), timeout))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def sleep(self, seconds: float) -> None:
        """Advance the fake clock."""
        self.sleeps.append(seconds)
        self.now += seconds

    def client(self) -> CratesIo:
        """Return a lookup wired to the fakes."""
        return CratesIo(
            "test-agent (alice)", opener=self.opener, clock=lambda: self.now, sleep=self.sleep
        )

    def test_returns_the_publish_date(self) -> None:
        """The URL, User-Agent and parsed date are what crates.io documents."""
        self.replies.append(response(version_body("serde", "1.0.1", "2026-10-19T20:11:38.5Z")))
        published = self.client()(Crate("serde", "1.0.1"))
        self.assertEqual(published, datetime(2026, 10, 19, 20, 11, 38, 500000, tzinfo=UTC))
        self.assertEqual(
            self.requests,
            [("https://crates.io/api/v1/crates/serde/1.0.1", "test-agent (alice)", 30)],
        )

    def test_waits_a_second_between_requests(self) -> None:
        """crates.io asks for at most one request per second."""
        self.replies.append(response(version_body("a", "1.0.0", "2026-01-01T00:00:00Z")))
        self.replies.append(response(version_body("b", "1.0.0", "2026-01-01T00:00:00Z")))
        client = self.client()
        client(Crate("a", "1.0.0"))
        self.now += 0.25
        client(Crate("b", "1.0.0"))
        self.assertEqual(self.sleeps, [0.75])

    def test_failures_raise_publish_date_errors(self) -> None:
        """Not found, network errors and unexpected bodies all fail closed."""
        not_found = urllib.error.HTTPError("u", 404, "Not Found", Message(), None)
        cases: dict[str, io.BytesIO | Exception] = {
            "not found": not_found,
            "network": urllib.error.URLError("connection refused"),
            "timeout": TimeoutError("timed out"),
            "incomplete read": BrokenReply(),
            "not json": io.BytesIO(b"<html>"),
            "deeply nested json": io.BytesIO(b"[" * 200_000 + b"]" * 200_000),
            "no version": response({"errors": []}),
            "other version": response(version_body("serde", "1.0.2", "2026-01-01T00:00:00Z")),
            "other crate": response(version_body("serde_json", "1.0.1", "2026-01-01T00:00:00Z")),
            "bad date": response(version_body("serde", "1.0.1", "yesterday")),
            "no timezone": response(version_body("serde", "1.0.1", "2026-01-01T00:00:00")),
        }
        for case, reply in cases.items():
            self.replies.append(reply)
            with self.subTest(case=case), self.assertRaises(PublishDateError):
                self.client()(Crate("serde", "1.0.1"))

    def test_too_large_a_body_fails(self) -> None:
        """A response is read up to a limit; anything larger is refused."""
        self.replies.append(io.BytesIO(b" " * (crate_age.MAX_RESPONSE_BYTES + 1)))
        with self.assertRaisesRegex(PublishDateError, "larger than"):
            self.client()(Crate("serde", "1.0.1"))


# The test repositories ignore the user's and the system's git configuration, so
# commit signing, hooks and templates never apply to them.
GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "alice",
    "GIT_AUTHOR_EMAIL": "alice@example.com",
    "GIT_COMMITTER_NAME": "alice",
    "GIT_COMMITTER_EMAIL": "alice@example.com",
}
YOUNG_LOCK = lockfile(("webpki-root-certs", "1.0.10", CRATES_IO), ("mcp-oauth", "0.3.0", None))
YOUNG_ALLOWED = allow_list(
    ("webpki-root-certs", "1.0.10", "Oldest release fixing RUSTSEC-2026-0383")
)


class GitRepoTest(unittest.TestCase):
    """Reading files from git revisions and running main() in a temporary repository."""

    def setUp(self) -> None:
        """Create an empty repository and point the script's ROOT at it."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.git("init", "--quiet")
        patcher = mock.patch.object(crate_age, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        # main() must never reach crates.io from these tests.
        self.lookup = FakeLookup({Crate("webpki-root-certs", "1.0.10"): NEW})
        client = mock.patch.object(crate_age, "CratesIo", return_value=self.lookup)
        client.start()
        self.addCleanup(client.stop)

    def git(self, *args: str) -> str:
        """Run git in the temporary repository."""
        return subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            cwd=self.root,
            env=GIT_ENV,
        ).stdout

    def write(self, files: dict[str, str | None]) -> None:
        """Write (or, for None, delete) files in the working tree."""
        for path, text in files.items():
            target = self.root / path
            if text is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")

    def commit(self, files: dict[str, str | None]) -> str:
        """Commit `files` on top of the current tree and return the commit id."""
        self.write(files)
        self.git("add", "--all")
        self.git("commit", "--quiet", "--allow-empty", "--message", "Change files")
        return self.git("rev-parse", "HEAD").strip()

    def run_main(self, *args: str) -> tuple[int, str]:
        """Run the script's entry point and return its exit status and output."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = crate_age.main(list(args))
        return status, out.getvalue()

    def test_file_at_reads_a_file_at_a_revision(self) -> None:
        """The file is read as committed, not as it is in the working tree."""
        rev = self.commit({"Cargo.lock": "first"})
        self.commit({"Cargo.lock": "second"})
        self.assertEqual(crate_age.file_at(rev, "Cargo.lock"), "first")

    def test_file_at_returns_none_for_a_file_the_revision_lacks(self) -> None:
        """A missing file is None, not an error."""
        rev = self.commit({"README.md": "hello"})
        self.assertIsNone(crate_age.file_at(rev, "Cargo.lock"))

    def test_file_at_refuses_unknown_and_option_like_revisions(self) -> None:
        """An unknown revision is named in the error; one starting with - never reaches git."""
        self.commit({"README.md": "hello"})
        with self.assertRaisesRegex(ValueError, "unknown revision 'nope'"):
            crate_age.file_at("nope", "Cargo.lock")
        with self.assertRaisesRegex(ValueError, "invalid revision"):
            crate_age.file_at("--output=x", "Cargo.lock")

    def test_head_files_reads_the_working_tree_without_a_revision(self) -> None:
        """Both files come from the working tree, uncommitted changes included."""
        self.commit({"Cargo.lock": "committed", crate_age.ALLOW_LIST: "committed"})
        self.write({"Cargo.lock": "lock", crate_age.ALLOW_LIST: "allow"})
        self.assertEqual(crate_age.head_files(None), ("lock", "allow"))

    def test_head_files_reads_both_files_at_a_revision(self) -> None:
        """The allow-list comes from the same revision as the lockfile."""
        rev = self.commit({"Cargo.lock": "lock", crate_age.ALLOW_LIST: "allow"})
        self.write({"Cargo.lock": "other", crate_age.ALLOW_LIST: "other"})
        self.assertEqual(crate_age.head_files(rev), ("lock", "allow"))

    def test_head_files_without_an_allow_list_allows_nothing(self) -> None:
        """A revision from before the allow-list existed has an empty one."""
        rev = self.commit({"Cargo.lock": "lock"})
        self.assertEqual(crate_age.head_files(rev), ("lock", ""))

    def test_head_files_requires_a_lockfile(self) -> None:
        """A head revision without Cargo.lock is an error."""
        rev = self.commit({"README.md": "hello"})
        with self.assertRaisesRegex(ValueError, "has no Cargo.lock"):
            crate_age.head_files(rev)

    def test_main_with_no_base_checks_every_version(self) -> None:
        """--no-base looks up versions that are in the lockfile already."""
        self.commit({"Cargo.lock": YOUNG_LOCK, crate_age.ALLOW_LIST: ""})
        status, out = self.run_main("--no-base")
        self.assertEqual(status, 1)
        self.assertIn("webpki-root-certs 1.0.10 was published on", out)
        self.assertEqual(self.lookup.calls, [Crate("webpki-root-certs", "1.0.10")])

    def test_main_with_no_base_passes_allowed_versions(self) -> None:
        """--no-base still honours the allow-list, without a lookup."""
        self.commit({"Cargo.lock": YOUNG_LOCK, crate_age.ALLOW_LIST: YOUNG_ALLOWED})
        status, _ = self.run_main("--no-base")
        self.assertEqual(status, 0)
        self.assertEqual(self.lookup.calls, [])

    def test_main_reads_the_allow_list_at_head(self) -> None:
        """--head uses that revision's allow-list, not the working tree's."""
        base = self.commit({"Cargo.lock": lockfile(), crate_age.ALLOW_LIST: ""})
        head = self.commit({"Cargo.lock": YOUNG_LOCK, crate_age.ALLOW_LIST: YOUNG_ALLOWED})
        self.write({crate_age.ALLOW_LIST: ""})
        self.assertEqual(self.run_main("--base", base, "--head", head), (0, mock.ANY))
        status, out = self.run_main("--base", base)
        self.assertEqual(status, 1)
        self.assertIn("webpki-root-certs 1.0.10", out)

    def test_main_reports_an_unknown_base_as_an_annotation(self) -> None:
        """A bad revision fails with an annotation rather than a traceback."""
        self.commit({"Cargo.lock": YOUNG_LOCK, crate_age.ALLOW_LIST: ""})
        status, out = self.run_main("--base", "nope")
        self.assertEqual(status, 1)
        self.assertEqual(out, crate_age.annotation("unknown revision 'nope'") + "\n")


class AnnotationTest(unittest.TestCase):
    """GitHub workflow command output."""

    def test_escapes_workflow_command_characters(self) -> None:
        """A message cannot end the annotation early or start another command."""
        self.assertEqual(
            crate_age.annotation("50%\r\n::error::x"),
            "::error file=Cargo.lock,title=Crate age::50%25%0D%0A::error::x",
        )


if __name__ == "__main__":
    unittest.main()
