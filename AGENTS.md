# AGENTS.md

Context file for AI coding agents working on this repository.

## What this is

`mcp-oauth` is a Rust library crate that provides a reusable OAuth 2.1 layer for MCP (Model Context Protocol) servers. It is designed for compatibility with Claude.ai. It is **not** a standalone binary — consumers import it and call `build_oauth_router()`.

## Build & development commands

```bash
cargo check          # type-check (fastest feedback loop)
cargo build          # compile the library
cargo test           # run all tests
cargo clippy         # lint (strict: all=deny, pedantic=warn, unsafe_code=forbid)
cargo doc --no-deps  # generate API docs
```

## Architecture

The entire library lives in **`src/lib.rs`** (single-file crate). HTML templates are in `templates/`.

### Public API

- `OAuthConfig` — configuration struct (server URL, client credentials, app name, token lifetimes). Constructed via `OAuthConfig::with_defaults()`.
- `build_oauth_router(protected_router: Router, config: OAuthConfig) -> Router` — wraps an axum `Router` with OAuth endpoints + Bearer token middleware.

### Internal state

`OAuthStore` holds all state in `tokio::sync::Mutex<HashMap<…>>` maps (auth codes, access tokens, refresh tokens, registered clients). State is in-memory only — no database required.

### OAuth flow

1. Discovery via `/.well-known/oauth-protected-resource` and `/.well-known/oauth-authorization-server`
2. Dynamic client registration at `POST /register` (RFC 7591)
3. Authorization at `GET|POST /authorize` — WebAuthn/passkey approval
4. Token exchange at `POST /token` — `authorization_code` and `refresh_token` grants
5. Protected routes are guarded by Bearer token middleware

### Security posture

- `unsafe_code = "forbid"` — no unsafe Rust
- `unwrap_used = "deny"` in clippy — panics are explicitly justified
- Constant-time secret comparison via `subtle`
- PKCE S256 enforced, no plain
- Per-IP rate limiting via `governor`
- Supply-chain auditing via `cargo-deny` and `cargo-audit`

## Dependencies

Key crates: `axum` (HTTP), `tokio` (async), `webauthn-rs` (passkeys), `governor` (rate limiting), `sha2`/`subtle` (crypto), `askama` (templates).

## Contributing

- Run `cargo clippy` before submitting — the lint config is strict
- Run `cargo deny check` to verify dependency licenses and advisories
- All public items should have doc comments
