# mcp-oauth

A reusable OAuth 2.1 layer for [MCP](https://modelcontextprotocol.io) (Model Context Protocol) servers, designed for compatibility with [Claude.ai](https://claude.ai).

[![CI](https://github.com/pecheriere/mcp-oauth/actions/workflows/ci.yml/badge.svg)](https://github.com/pecheriere/mcp-oauth/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/mcp-oauth.svg)](https://crates.io/crates/mcp-oauth)
[![Documentation](https://docs.rs/mcp-oauth/badge.svg)](https://docs.rs/mcp-oauth)
[![License](https://img.shields.io/crates/l/mcp-oauth.svg)](LICENSE-MIT)

## What is this?

`mcp-oauth` is a Rust library crate that wraps your [axum](https://docs.rs/axum) router with a complete OAuth 2.1 implementation — discovery endpoints, dynamic client registration, authorization with WebAuthn/passkey approval, token exchange, and Bearer token middleware. Drop it in front of your MCP server and get Claude.ai-compatible authentication out of the box.

This is **not** a standalone binary. You import the crate and call `build_oauth_router()`.

## Features

- **OAuth 2.1 with PKCE** (S256) — authorization code flow with proof key for code exchange
- **Dynamic client registration** ([RFC 7591](https://www.rfc-editor.org/rfc/rfc7591))
- **WebAuthn / passkey authentication** — passwordless approval via hardware keys or biometrics
- **Token refresh** — long-lived sessions via refresh tokens
- **Per-IP rate limiting** — three tiers (auth, registration, general)
- **In-memory state** with TTL-based cleanup — no database required
- **Strict security** — `unsafe` forbidden, constant-time secret comparison, PKCE enforced

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
mcp-oauth = "0.1"
```

Wrap your MCP router:

```rust
use axum::Router;
use mcp_oauth::{OAuthConfig, build_oauth_router};
use std::path::PathBuf;

let mcp_routes = Router::new(); // your protected MCP routes

let config = OAuthConfig::with_defaults(
    "https://my-mcp.example.com".into(),
    "my-client-id".into(),
    "my-client-secret".into(),
    "My MCP Server".into(),
    PathBuf::from("passkeys.json"),
    Some("initial-setup-token".into()),
);

let app = build_oauth_router(mcp_routes, config);
// Serve `app` with axum / hyper as usual.
```

## Configuration

`OAuthConfig` fields:

| Field | Description | Default |
|-------|-------------|---------|
| `server_url` | Public-facing URL of your server | (required) |
| `client_id` | Pre-registered OAuth client ID | (required) |
| `client_secret` | Pre-registered OAuth client secret | (required) |
| `app_name` | Human-readable name shown on approval pages | (required) |
| `passkey_store_path` | Path to JSON file for persisting registered passkeys | (required) |
| `setup_token` | One-time token for first passkey registration | `None` |
| `token_lifetime_secs` | Access token lifetime | 86400 (24h) |
| `code_lifetime_secs` | Authorization code lifetime | 300 (5min) |

## OAuth flow

1. **Discovery** — `/.well-known/oauth-protected-resource` and `/.well-known/oauth-authorization-server`
2. **Client registration** — `POST /register` (RFC 7591, optional)
3. **Authorization** — `GET /authorize` renders a WebAuthn/passkey approval page
4. **Token exchange** — `POST /token` supports `authorization_code` and `refresh_token` grants
5. **Protected routes** — Bearer token middleware validates access tokens on your MCP endpoints

## Security

- No `unsafe` code (`#![forbid(unsafe_code)]`)
- Constant-time secret comparison via [`subtle`](https://docs.rs/subtle)
- PKCE S256 enforced (no plain)
- Per-IP rate limiting via [`governor`](https://docs.rs/governor)
- Supply-chain auditing via [`cargo-deny`](https://github.com/EmbarkStudios/cargo-deny)

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

## License

Licensed under either of

- [MIT license](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
