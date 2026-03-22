# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

- `cargo build` — build the library
- `cargo check` — type-check without producing artifacts (fastest feedback loop)
- `cargo test` — run all tests
- `cargo clippy` — lint

## What This Is

A Rust library crate (`mcp-oauth`) that provides a reusable OAuth 2.1 layer for MCP (Model Context Protocol) servers, designed for compatibility with Claude.ai. It is not a standalone binary — consumers import it and call `build_oauth_router()`.

## Architecture

The entire library lives in `src/lib.rs`. The public API surface is:

- **`OAuthConfig`** — configuration struct (server URL, client credentials, approval password, app name, token lifetimes). Constructed via `OAuthConfig::with_defaults()`.
- **`build_oauth_router(protected_router: Router, config: OAuthConfig) -> Router`** — the main entry point. Takes an axum `Router` containing the consumer's protected MCP routes and wraps it with OAuth endpoints + Bearer token middleware.

Internally, `OAuthStore` holds all state in `tokio::sync::Mutex<HashMap<…>>` maps (auth codes, access tokens, refresh tokens, dynamically registered clients). State is in-memory only — no persistence.

### OAuth Flow

The implementation follows OAuth 2.1 with PKCE (S256 only):

1. Discovery via `/.well-known/oauth-protected-resource` and `/.well-known/oauth-authorization-server`
2. Optional dynamic client registration at `POST /register` (RFC 7591)
3. Authorization at `GET|POST /authorize` — renders an HTML password-approval page
4. Token exchange at `POST /token` — supports `authorization_code` and `refresh_token` grant types
5. Protected routes are guarded by `auth_middleware` which validates Bearer tokens

Redirect URIs are restricted to `ALLOWED_REDIRECT_URIS` (claude.ai and claude.com callback URLs) plus any URIs registered via dynamic client registration.

### Rate Limiting

The library does **not** include per-client or per-IP rate limiting. It provides global capacity limits on in-memory maps (with TTL-based cleanup and 429 rejection when full) as a backstop, but consumers should add their own rate-limiting middleware (e.g. `tower::limit::RateLimitLayer`) to the router returned by `build_oauth_router()`, especially on `/token`, `/register`, and `/passkey/*`.
