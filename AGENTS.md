# AGENTS.md

Context file for AI coding agents working on this repository.

## What this is

`mcp-oauth` is a Rust library crate that provides a reusable OAuth 2.1 layer for MCP (Model Context Protocol) servers. It is designed for compatibility with Claude.ai. It is **not** a standalone binary -- consumers import it and call `build_oauth_router()`.

## Build & development commands

```bash
cargo check          # type-check (fastest feedback loop)
cargo build          # compile the library
cargo test           # run all tests
cargo clippy         # lint (strict: all=deny, pedantic=warn, unsafe_code=forbid)
cargo doc --no-deps  # generate API docs
```

## Architecture

Source is split across `src/lib.rs` (handlers, router, config) and `src/store/` (storage traits and implementations). HTML templates are in `templates/`.

### Public API

- `OAuthConfig` -- configuration struct (server URL, client credentials, app name, token lifetimes). Constructed via `OAuthConfig::with_defaults()`.
- `build_oauth_router_with_stores(protected_router, config, token_store, client_store, passkey_store)` -- wraps an axum `Router` with OAuth endpoints + Bearer token middleware using pluggable storage backends.
- `create_default_stores(&config)` -- creates the default JSON-file-backed stores.
- `build_oauth_router` -- **deprecated** (v0.2.0), delegates to `build_oauth_router_with_stores` with default stores.

### Storage traits

`TokenStore`, `ClientStore`, `PasskeyStore` -- async traits in `src/store/mod.rs` defining all storage operations. Default JSON-file-backed implementations live in `src/store/json_file.rs`. State types (`AuthCode`, `AccessTokenEntry`, `RefreshTokenEntry`, `RegisteredClient`) are `#[non_exhaustive]` with `new()` constructors.

### Internal state

`OAuthServer<T, C, P>` is generic over the three store traits. Transient WebAuthn state (registration/authentication sessions, auth session cookies) lives in-memory on `OAuthServer` and is not behind the storage traits.

### OAuth flow

1. Discovery via `/.well-known/oauth-protected-resource` and `/.well-known/oauth-authorization-server`
2. Dynamic client registration at `POST /register` (RFC 7591)
3. Authorization at `GET|POST /authorize` -- WebAuthn/passkey approval
4. Token exchange at `POST /token` -- `authorization_code` and `refresh_token` grants
5. Protected routes are guarded by Bearer token middleware

### Security posture

- `unsafe_code = "forbid"` -- no unsafe Rust
- `unwrap_used = "deny"` in clippy -- panics are explicitly justified
- Constant-time secret comparison via `subtle`
- PKCE S256 enforced, no plain
- Per-IP rate limiting via `governor`
- Supply-chain auditing via `cargo-deny` and `cargo-audit`

## Dependencies

Key crates: `axum` (HTTP), `tokio` (async), `webauthn-rs` (passkeys), `governor` (rate limiting), `sha2`/`subtle` (crypto), `askama` (templates).

## Testing requirements

**Unit tests are mandatory** for all new features and bug fixes. This is not optional.

- Every new public function, trait method, or behavior change must have corresponding tests.
- Bug fixes must include a regression test that would have caught the bug.
- Store-level tests live in `src/store/json_file.rs` (`mod tests`). Integration tests (HTTP-level) live in `src/lib.rs` (`mod tests`).

### No test slop

Tests must be meaningful. A test that cannot fail is worse than no test — it adds noise and false confidence.

- **Test behavior, not implementation.** Assert on observable outcomes (return values, HTTP status codes, state changes), not on internal details like which function was called.
- **Each test must be able to fail.** If you can't describe a realistic code change that would make the test fail, delete it. A test that just calls a constructor and asserts it didn't panic is not a test.
- **One clear assertion per test.** Test names should describe the scenario and expected outcome (e.g. `test_refresh_token_wrong_client_id_preserves_token`, not `test_refresh_token_3`).
- **No tautological assertions.** Don't assert that `true == true` or that a value you just constructed has the fields you set. Assert on the *system's* behavior after an operation.
- **Prefer fewer strong tests over many weak ones.** A single test that exercises a full flow (store → retrieve → verify → reload → verify persisted) is worth more than five trivial getter tests.

### Design for testability

Write code so it can be tested in isolation:

- **Inject dependencies via traits** -- this is why `TokenStore`, `ClientStore`, `PasskeyStore` exist. New storage operations go through these traits, not directly on structs.
- **Accept time as a parameter** where deterministic behavior matters (e.g. `cleanup_expired_tokens(now: u64)`). Do not call `now_epoch()` deep inside a call chain if the caller could pass the value.
- **Prefer pure functions** -- separate validation logic from I/O so it can be tested without a running server.
- **Atomic operations for check-then-act** -- if a check and a mutation must happen together, make it a single trait method (e.g. `register_client_if_none`, `add_passkey_if_none`) to prevent TOCTOU races and make the atomicity testable.
- **Use `saturating_sub`** for all timestamp arithmetic -- never bare subtraction on `u64` timestamps.

## Contributing

- Run `cargo clippy` before submitting -- the lint config is strict
- Run `cargo deny check` to verify dependency licenses and advisories
- All public items should have doc comments
