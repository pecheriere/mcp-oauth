//! # mcp-oauth
//!
//! A reusable OAuth 2.1 layer for [MCP](https://modelcontextprotocol.io)
//! (Model Context Protocol) servers, designed for compatibility with Claude.ai.
//!
//! This crate is not a standalone binary — consumers import it and call
//! [`build_oauth_router_with_stores`] to wrap their [axum](https://docs.rs/axum) `Router`
//! with a complete OAuth 2.1 implementation.
//!
//! ## Features
//!
//! - **OAuth 2.1 with PKCE** (S256) — authorization code flow with proof key
//! - **Dynamic client registration** ([RFC 7591](https://www.rfc-editor.org/rfc/rfc7591))
//! - **`WebAuthn` / passkey authentication** — passwordless approval via hardware keys or biometrics
//! - **Token refresh** — long-lived sessions via refresh tokens
//! - **Per-IP rate limiting** — three tiers (auth, registration, general)
//! - **Pluggable storage** via [`TokenStore`], [`ClientStore`], [`PasskeyStore`] traits
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use axum::Router;
//! use mcp_oauth::{OAuthConfig, build_oauth_router_with_stores};
//! use std::path::PathBuf;
//!
//! let mcp_routes = Router::new(); // your protected MCP routes
//!
//! // Using the builder (recommended):
//! let config = OAuthConfig::builder(
//!     "https://my-mcp.example.com".into(),
//!     "my-client-id".into(),
//!     "my-client-secret".into(),
//!     "My MCP Server".into(),
//!     PathBuf::from("passkeys.json"),
//! )
//! .setup_token("initial-setup-token")
//! .add_redirect_uri("https://myapp.example.com/callback")
//! .build()
//! .expect("valid config");
//!
//! let (token_store, client_store, passkey_store) =
//!     mcp_oauth::create_default_stores(&config);
//! let app = build_oauth_router_with_stores(
//!     mcp_routes, config, token_store, client_store, passkey_store,
//! );
//! // Serve `app` with axum / hyper as usual.
//! ```

pub mod store;

pub use store::json_file::{JsonFileClientStore, JsonFilePasskeyStore, JsonFileTokenStore};
pub use store::{
    AccessTokenEntry, AuthCode, ClientStore, PasskeyStore, RefreshTokenEntry, RegisteredClient,
    StoreError, TokenStore,
};

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::{Component, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use governor::clock::DefaultClock;
use governor::state::keyed::DashMapStateStore;
use governor::{Quota, RateLimiter};

use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Form, Json, Router};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand::TryRngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::sync::Mutex;
use url::Url;
use uuid::Uuid;
use webauthn_rs::prelude::*;

// L2: use unwrap_or_default to avoid panic on pre-epoch system time
fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// H1: Constant-time comparison for secrets to prevent timing side-channels
fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

use askama::Template;

// L1: Generate 256-bit cryptographically random tokens (base64url-encoded)
#[expect(
    clippy::expect_used,
    reason = "OsRng::try_fill_bytes only fails on catastrophic OS RNG failure; panicking is the correct response for a security-critical token generator"
)]
fn generate_token() -> String {
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng
        .try_fill_bytes(&mut bytes)
        .expect("OS RNG failed");
    URL_SAFE_NO_PAD.encode(bytes)
}

// ---------------------------------------------------------------------------
// Public config
// ---------------------------------------------------------------------------

/// Per-IP rate limiting configuration (requests per minute).
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Rate limit for auth-critical endpoints (`/token`, `/register`, `/passkey/*`).
    /// Default: 10 req/min.
    pub strict: u32,
    /// Rate limit for public endpoints (`/.well-known/*`, `/authorize`, `/health`).
    /// Default: 30 req/min.
    pub moderate: u32,
    /// Rate limit for protected (Bearer-authed) routes.
    /// Default: 60 req/min.
    pub lenient: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            strict: 10,
            moderate: 30,
            lenient: 60,
        }
    }
}

/// Capacity limits for in-memory transient state and persistent stores.
#[derive(Debug, Clone)]
pub struct CapacityConfig {
    /// Max pending passkey registration sessions. Default: 10000.
    pub max_registration_states: usize,
    /// Max pending passkey authentication sessions. Default: 10000.
    pub max_authentication_states: usize,
    /// Max simultaneously stored access tokens. Default: 10000.
    pub max_access_tokens: usize,
    /// Max simultaneously stored refresh tokens. Default: 10000.
    pub max_refresh_tokens: usize,
    /// Max pending (unconsumed) authorization codes. Default: 10000.
    pub max_auth_codes: usize,
    /// Max dynamically registered OAuth clients.
    ///
    /// - `Some(n)` caps the store at `n` clients (default `Some(1)`: preserves
    ///   the historical single-client registration lock).
    /// - `None` allows unlimited dynamic client registrations.
    pub max_registered_clients: Option<usize>,
}

impl Default for CapacityConfig {
    fn default() -> Self {
        Self {
            max_registration_states: 10_000,
            max_authentication_states: 10_000,
            max_access_tokens: 10_000,
            max_refresh_tokens: 10_000,
            max_auth_codes: 10_000,
            max_registered_clients: Some(1),
        }
    }
}

/// Errors that can occur when building an [`OAuthConfig`] via the builder.
#[derive(Debug)]
#[non_exhaustive]
pub enum OAuthConfigError {
    /// `client_id` must not be empty.
    EmptyClientId,
    /// `client_secret` must not be empty.
    EmptyClientSecret,
    /// `passkey_store_path` must not contain `..` components.
    PathTraversal,
    /// Rate limit values must be non-zero.
    ZeroRateLimit,
    /// At least one scope is required.
    EmptyScopes,
    /// A capacity limit was set to zero. Use `None` on `max_registered_clients`
    /// for "unlimited"; all other capacity fields must be at least 1.
    ZeroCapacity,
}

impl std::fmt::Display for OAuthConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyClientId => write!(f, "client_id must not be empty"),
            Self::EmptyClientSecret => write!(f, "client_secret must not be empty"),
            Self::PathTraversal => {
                write!(f, "passkey_store_path must not contain '..' components")
            }
            Self::ZeroRateLimit => write!(f, "rate limit values must be non-zero"),
            Self::EmptyScopes => write!(f, "scopes must not be empty"),
            Self::ZeroCapacity => write!(
                f,
                "capacity limit must be at least 1 (use max_registered_clients: None for unlimited)"
            ),
        }
    }
}

impl std::error::Error for OAuthConfigError {}

/// Returns the default allowed redirect URIs (Claude.ai callbacks).
#[must_use]
pub fn default_redirect_uris() -> Vec<String> {
    vec![
        "https://claude.ai/api/mcp/auth_callback".to_owned(),
        "https://claude.com/api/mcp/auth_callback".to_owned(),
    ]
}

#[non_exhaustive]
pub struct OAuthConfig {
    /// The public-facing URL of this server (e.g. `<https://my-mcp.fly.dev>`).
    pub server_url: String,
    /// Pre-registered OAuth client ID.
    pub client_id: String,
    /// Pre-registered OAuth client secret.
    pub client_secret: String,
    /// Human-readable app name shown on pages.
    pub app_name: String,
    /// Where to persist registered passkeys (JSON file).
    pub passkey_store_path: PathBuf,
    /// One-time token for first passkey registration (when no passkeys exist yet).
    pub setup_token: Option<String>,
    /// Access token lifetime in seconds. Default: 86400 (24 hours).
    pub token_lifetime_secs: u64,
    /// Authorization code lifetime in seconds. Default: 300 (5 minutes).
    pub code_lifetime_secs: u64,
    /// Redirect URIs that are always accepted (beyond per-client registered URIs).
    /// Defaults to the Claude.ai callback URLs.
    pub allowed_redirect_uris: Vec<String>,
    /// Per-IP rate limiting tiers.
    pub rate_limits: RateLimitConfig,
    /// In-memory capacity limits for transient state.
    pub capacity: CapacityConfig,
    /// OAuth scopes supported and returned in token responses.
    /// Defaults to `["mcp:tools"]`.
    pub scopes: Vec<String>,
}

impl OAuthConfig {
    /// Create a new `OAuthConfig` with default token lifetimes.
    ///
    /// # Panics
    ///
    /// Panics if `client_id` or `client_secret` is empty.
    #[must_use]
    pub fn with_defaults(
        server_url: String,
        client_id: String,
        client_secret: String,
        app_name: String,
        passkey_store_path: PathBuf,
        setup_token: Option<String>,
    ) -> Self {
        // L5: Validate non-empty credentials to prevent empty-string bypass
        assert!(!client_id.is_empty(), "client_id must not be empty");
        assert!(!client_secret.is_empty(), "client_secret must not be empty");
        // Defense-in-depth: reject paths with parent-directory traversal components
        assert!(
            !passkey_store_path
                .components()
                .any(|c| c == Component::ParentDir),
            "passkey_store_path must not contain '..' components"
        );

        Self {
            server_url,
            client_id,
            client_secret,
            app_name,
            passkey_store_path,
            setup_token,
            token_lifetime_secs: 86400,
            code_lifetime_secs: 300,
            allowed_redirect_uris: default_redirect_uris(),
            rate_limits: RateLimitConfig::default(),
            capacity: CapacityConfig::default(),
            scopes: vec!["mcp:tools".to_owned()],
        }
    }

    /// Create a builder for `OAuthConfig` with required parameters.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mcp_oauth::OAuthConfig;
    /// use std::path::PathBuf;
    ///
    /// let config = OAuthConfig::builder(
    ///     "https://my-mcp.example.com".into(),
    ///     "my-client-id".into(),
    ///     "my-client-secret".into(),
    ///     "My MCP Server".into(),
    ///     PathBuf::from("passkeys.json"),
    /// )
    /// .setup_token("initial-setup-token")
    /// .token_lifetime_secs(3600)
    /// .add_redirect_uri("https://myapp.example.com/callback")
    /// .build()
    /// .expect("valid config");
    /// ```
    #[must_use]
    pub fn builder(
        server_url: String,
        client_id: String,
        client_secret: String,
        app_name: String,
        passkey_store_path: PathBuf,
    ) -> OAuthConfigBuilder {
        OAuthConfigBuilder {
            server_url,
            client_id,
            client_secret,
            app_name,
            passkey_store_path,
            setup_token: None,
            token_lifetime_secs: 86400,
            code_lifetime_secs: 300,
            allowed_redirect_uris: default_redirect_uris(),
            rate_limits: RateLimitConfig::default(),
            capacity: CapacityConfig::default(),
            scopes: vec!["mcp:tools".to_owned()],
        }
    }
}

/// Builder for [`OAuthConfig`].
///
/// Created via [`OAuthConfig::builder`]. Call [`.build()`](OAuthConfigBuilder::build)
/// to validate and produce the final config.
pub struct OAuthConfigBuilder {
    server_url: String,
    client_id: String,
    client_secret: String,
    app_name: String,
    passkey_store_path: PathBuf,
    setup_token: Option<String>,
    token_lifetime_secs: u64,
    code_lifetime_secs: u64,
    allowed_redirect_uris: Vec<String>,
    rate_limits: RateLimitConfig,
    capacity: CapacityConfig,
    scopes: Vec<String>,
}

impl OAuthConfigBuilder {
    /// Set the one-time setup token for first passkey registration.
    #[must_use]
    pub fn setup_token(mut self, token: impl Into<String>) -> Self {
        self.setup_token = Some(token.into());
        self
    }

    /// Set access token lifetime in seconds (default: 86400 = 24 hours).
    #[must_use]
    pub const fn token_lifetime_secs(mut self, secs: u64) -> Self {
        self.token_lifetime_secs = secs;
        self
    }

    /// Set authorization code lifetime in seconds (default: 300 = 5 minutes).
    #[must_use]
    pub const fn code_lifetime_secs(mut self, secs: u64) -> Self {
        self.code_lifetime_secs = secs;
        self
    }

    /// Replace all allowed redirect URIs (overrides the default Claude.ai URIs).
    #[must_use]
    pub fn allowed_redirect_uris(mut self, uris: Vec<String>) -> Self {
        self.allowed_redirect_uris = uris;
        self
    }

    /// Add an additional allowed redirect URI (appends to defaults).
    #[must_use]
    pub fn add_redirect_uri(mut self, uri: impl Into<String>) -> Self {
        self.allowed_redirect_uris.push(uri.into());
        self
    }

    /// Set per-IP rate limiting configuration.
    #[must_use]
    pub const fn rate_limits(mut self, config: RateLimitConfig) -> Self {
        self.rate_limits = config;
        self
    }

    /// Set the full capacity configuration.
    #[must_use]
    pub const fn capacity(mut self, config: CapacityConfig) -> Self {
        self.capacity = config;
        self
    }

    /// Set the maximum number of simultaneously stored access tokens.
    #[must_use]
    pub const fn max_access_tokens(mut self, n: usize) -> Self {
        self.capacity.max_access_tokens = n;
        self
    }

    /// Set the maximum number of simultaneously stored refresh tokens.
    #[must_use]
    pub const fn max_refresh_tokens(mut self, n: usize) -> Self {
        self.capacity.max_refresh_tokens = n;
        self
    }

    /// Set the maximum number of pending authorization codes.
    #[must_use]
    pub const fn max_auth_codes(mut self, n: usize) -> Self {
        self.capacity.max_auth_codes = n;
        self
    }

    /// Set the cap on dynamically registered clients.
    ///
    /// Pass `None` for unlimited registrations, or `Some(n)` to cap the store
    /// at `n` clients. The default is `Some(1)`, which preserves the
    /// historical single-client registration lock.
    #[must_use]
    pub const fn max_registered_clients(mut self, n: Option<usize>) -> Self {
        self.capacity.max_registered_clients = n;
        self
    }

    /// Replace all supported OAuth scopes (overrides the default `["mcp:tools"]`).
    #[must_use]
    pub fn scopes(mut self, scopes: Vec<String>) -> Self {
        self.scopes = scopes;
        self
    }

    /// Add an additional OAuth scope (appends to defaults).
    #[must_use]
    pub fn add_scope(mut self, scope: impl Into<String>) -> Self {
        self.scopes.push(scope.into());
        self
    }

    /// Validate and build the [`OAuthConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`OAuthConfigError`] if validation fails.
    pub fn build(self) -> Result<OAuthConfig, OAuthConfigError> {
        if self.client_id.is_empty() {
            return Err(OAuthConfigError::EmptyClientId);
        }
        if self.client_secret.is_empty() {
            return Err(OAuthConfigError::EmptyClientSecret);
        }
        if self
            .passkey_store_path
            .components()
            .any(|c| c == Component::ParentDir)
        {
            return Err(OAuthConfigError::PathTraversal);
        }
        if self.rate_limits.strict == 0
            || self.rate_limits.moderate == 0
            || self.rate_limits.lenient == 0
        {
            return Err(OAuthConfigError::ZeroRateLimit);
        }
        if self.scopes.is_empty() {
            return Err(OAuthConfigError::EmptyScopes);
        }
        if self.capacity.max_access_tokens == 0
            || self.capacity.max_refresh_tokens == 0
            || self.capacity.max_auth_codes == 0
            || self.capacity.max_registration_states == 0
            || self.capacity.max_authentication_states == 0
            || self.capacity.max_registered_clients == Some(0)
        {
            return Err(OAuthConfigError::ZeroCapacity);
        }

        Ok(OAuthConfig {
            server_url: self.server_url,
            client_id: self.client_id,
            client_secret: self.client_secret,
            app_name: self.app_name,
            passkey_store_path: self.passkey_store_path,
            setup_token: self.setup_token,
            token_lifetime_secs: self.token_lifetime_secs,
            code_lifetime_secs: self.code_lifetime_secs,
            allowed_redirect_uris: self.allowed_redirect_uris,
            rate_limits: self.rate_limits,
            capacity: self.capacity,
            scopes: self.scopes,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal server state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct PendingAuthApproval {
    client_id: String,
    redirect_uri: String,
    state: Option<String>,
    code_challenge: String,
    #[expect(
        dead_code,
        reason = "retained for Debug logging; the OAuth spec only defines S256 for us, but the field is kept so the pending-approval record round-trips exactly what the client sent"
    )]
    code_challenge_method: String,
}

// H2: Capacity limits now configurable via OAuthConfig.capacity
use store::TRANSIENT_STATE_TTL_SECS;

struct OAuthServer<T: TokenStore, C: ClientStore, P: PasskeyStore> {
    config: OAuthConfig,
    token_store: T,
    client_store: C,
    passkey_store: P,
    // Passkey / WebAuthn state
    webauthn: Webauthn,
    // H2: Timestamps added for TTL-based cleanup
    registration_states: Mutex<HashMap<String, (PasskeyRegistration, u64)>>,
    authentication_states:
        Mutex<HashMap<String, (PasskeyAuthentication, PendingAuthApproval, u64)>>,
    // Session cookie for auto-approving /authorize after first passkey auth
    auth_session_token: Mutex<Option<(String, u64)>>, // (token, created_at_epoch)
}

// Allowed redirect URIs now configurable via OAuthConfig.allowed_redirect_uris

type AppState<T, C, P> = Arc<OAuthServer<T, C, P>>;

// L3: Return Result instead of silently falling back to "localhost"
fn extract_domain(server_url: &str) -> Result<String, String> {
    Url::parse(server_url)
        .ok()
        .and_then(|u| u.host_str().map(ToString::to_string))
        .ok_or_else(|| format!("cannot extract domain from URL: {server_url}"))
}

impl<T: TokenStore, C: ClientStore, P: PasskeyStore> OAuthServer<T, C, P> {
    // H1: Constant-time secret comparison to prevent timing side-channels
    async fn validate_client(&self, client_id: &str, client_secret: &str) -> bool {
        let id_match = constant_time_eq(client_id, &self.config.client_id);
        let secret_match = constant_time_eq(client_secret, &self.config.client_secret);
        if id_match && secret_match {
            return true;
        }
        match self.client_store.get_client(client_id).await {
            Ok(Some(c)) => constant_time_eq(client_secret, &c.client_secret),
            _ => false,
        }
    }

    async fn is_known_client(&self, client_id: &str) -> bool {
        if client_id == self.config.client_id {
            return true;
        }
        matches!(self.client_store.get_client(client_id).await, Ok(Some(_)))
    }

    async fn is_redirect_uri_allowed(&self, client_id: &str, redirect_uri: &str) -> bool {
        if self
            .config
            .allowed_redirect_uris
            .iter()
            .any(|u| u == redirect_uri)
        {
            return true;
        }
        match self.client_store.get_client(client_id).await {
            Ok(Some(c)) => c.redirect_uris.iter().any(|u| u == redirect_uri),
            _ => false,
        }
    }

    async fn has_passkeys(&self) -> bool {
        match self.passkey_store.has_passkeys().await {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("Passkey store error in has_passkeys: {e}");
                false
            }
        }
    }

    async fn create_auth_session(&self) -> String {
        let token = generate_token();
        *self.auth_session_token.lock().await = Some((token.clone(), now_epoch()));
        token
    }

    async fn validate_auth_session(&self, cookie_token: &str) -> bool {
        let session = self.auth_session_token.lock().await;
        match session.as_ref() {
            Some((token, created_at)) => {
                let age = now_epoch().saturating_sub(*created_at);
                age < self.config.token_lifetime_secs && constant_time_eq(cookie_token, token)
            }
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Rate limiting
// ---------------------------------------------------------------------------

type IpRateLimiter = RateLimiter<String, DashMapStateStore<String>, DefaultClock>;

#[expect(
    clippy::unwrap_used,
    reason = "requests_per_minute is validated as non-zero by OAuthConfigBuilder::build (ZeroRateLimit error), so NonZeroU32::new cannot return None here"
)]
fn create_rate_limiter(requests_per_minute: u32) -> Arc<IpRateLimiter> {
    let quota = Quota::per_minute(NonZeroU32::new(requests_per_minute).unwrap());
    Arc::new(RateLimiter::dashmap(quota))
}

/// Extract the client IP from the request.
/// Prefers `CF-Connecting-IP` (set by Cloudflare Tunnel), falls back to
/// `X-Forwarded-For`, then peer socket address, then a static "unknown" key.
fn extract_client_ip(req: &axum::extract::Request) -> String {
    if let Some(ip) = req
        .headers()
        .get("CF-Connecting-IP")
        .and_then(|v| v.to_str().ok())
    {
        return ip.to_string();
    }
    if let Some(xff) = req
        .headers()
        .get("X-Forwarded-For")
        .and_then(|v| v.to_str().ok())
        && let Some(first) = xff.split(',').next()
    {
        return first.trim().to_string();
    }
    // Fall back to peer socket address (available when server uses
    // into_make_service_with_connect_info::<SocketAddr>())
    if let Some(connect_info) = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
    {
        return connect_info.0.ip().to_string();
    }
    tracing::warn!("Could not determine client IP; using shared \"unknown\" rate-limit bucket");
    "unknown".to_string()
}

async fn rate_limit_middleware(
    State(limiter): State<Arc<IpRateLimiter>>,
    req: axum::extract::Request,
    next: Next,
) -> Result<Response, Response> {
    let ip = extract_client_ip(&req);
    if limiter.check_key(&ip).is_ok() {
        Ok(next.run(req).await)
    } else {
        tracing::warn!("Rate limit exceeded for IP: {ip}");
        Err((StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded\n").into_response())
    }
}

// ---------------------------------------------------------------------------
// Public API: create_default_stores
// ---------------------------------------------------------------------------

/// Create the default JSON-file-backed stores from an [`OAuthConfig`].
///
/// Returns `(token_store, client_store, passkey_store)` suitable for passing
/// to [`build_oauth_router_with_stores`].
pub fn create_default_stores(
    config: &OAuthConfig,
) -> (impl TokenStore, impl ClientStore, impl PasskeyStore) {
    let caps = store::json_file::StoreCaps {
        max_access_tokens: config.capacity.max_access_tokens,
        max_refresh_tokens: config.capacity.max_refresh_tokens,
        max_auth_codes: config.capacity.max_auth_codes,
        max_registered_clients: config.capacity.max_registered_clients,
    };
    let (token_store, client_store, summary) =
        store::json_file::create_json_file_stores(&config.passkey_store_path, caps);

    tracing::info!(
        "OAuth store loaded: {} access_tokens, {} refresh_tokens, {} registered_clients from {:?}",
        summary.access_tokens,
        summary.refresh_tokens,
        summary.registered_clients,
        summary.tokens_path,
    );

    let passkey_store = JsonFilePasskeyStore::new(config.passkey_store_path.clone());

    (token_store, client_store, passkey_store)
}

// ---------------------------------------------------------------------------
// Public API: build_oauth_router (deprecated) and build_oauth_router_with_stores
// ---------------------------------------------------------------------------

/// Wraps `protected_router` with OAuth 2.1 endpoints and Bearer-token middleware.
///
/// # Deprecated
///
/// Use [`build_oauth_router_with_stores`] with explicit store implementations instead.
/// This function creates default JSON-file-backed stores from the config.
#[deprecated(
    since = "0.2.0",
    note = "use `build_oauth_router_with_stores` with explicit store implementations"
)]
pub fn build_oauth_router(protected_router: Router, config: OAuthConfig) -> Router {
    let (token_store, client_store, passkey_store) = create_default_stores(&config);
    build_oauth_router_with_stores(
        protected_router,
        config,
        token_store,
        client_store,
        passkey_store,
    )
}

/// Wraps `protected_router` with OAuth 2.1 endpoints and Bearer-token middleware.
///
/// # Rate Limiting
///
/// Three tiers of per-IP rate limiting (keyed by `CF-Connecting-IP` header):
/// - **Strict (10 req/min):** `/token`, `/register`, `/passkey/*` — auth brute-force protection
/// - **Moderate (30 req/min):** `/authorize`, `/.well-known/*`, `/health` — OAuth flow, metadata
/// - **Lenient (60 req/min):** `/mcp` (protected routes) — already behind Bearer auth
///
/// # Panics
///
/// Panics if `config.server_url` is not a valid URL or has no host component,
/// or if the `WebAuthn` builder fails (invalid RP configuration).
#[expect(
    clippy::expect_used,
    reason = "invalid server_url / WebAuthn RP config is a caller bug at startup, not a runtime condition; panicking surfaces it immediately rather than threading a Result through the public API"
)]
pub fn build_oauth_router_with_stores<T, C, P>(
    protected_router: Router,
    config: OAuthConfig,
    token_store: T,
    client_store: C,
    passkey_store: P,
) -> Router
where
    T: TokenStore,
    C: ClientStore,
    P: PasskeyStore,
{
    let rp_id =
        extract_domain(&config.server_url).expect("invalid server_url: cannot extract domain");
    let rp_origin = Url::parse(&config.server_url).expect("invalid server_url");
    let webauthn = WebauthnBuilder::new(&rp_id, &rp_origin)
        .expect("Failed to build WebAuthn")
        .rp_name(&config.app_name)
        .build()
        .expect("Failed to build WebAuthn");

    tracing::info!(
        "Token/passkey files are stored at {:?}. Ensure this directory is owned by the service user with 0o700 permissions.",
        config
            .passkey_store_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new(".")),
    );

    let store: AppState<T, C, P> = Arc::new(OAuthServer {
        config,
        token_store,
        client_store,
        passkey_store,
        webauthn,
        registration_states: Mutex::new(HashMap::new()),
        authentication_states: Mutex::new(HashMap::new()),
        auth_session_token: Mutex::new(None),
    });

    let strict_limiter = create_rate_limiter(store.config.rate_limits.strict);
    let moderate_limiter = create_rate_limiter(store.config.rate_limits.moderate);
    let lenient_limiter = create_rate_limiter(store.config.rate_limits.lenient);

    // Auth routes: strict rate limiting (10 req/min per IP)
    let auth_routes = Router::new()
        .route("/register", post(register_client::<T, C, P>))
        .route("/token", post(token::<T, C, P>))
        .route("/passkey/register", get(passkey_register_page::<T, C, P>))
        .route(
            "/passkey/register/start",
            post(passkey_register_start::<T, C, P>),
        )
        .route(
            "/passkey/register/finish",
            post(passkey_register_finish::<T, C, P>),
        )
        .route("/passkey/auth/start", post(passkey_auth_start::<T, C, P>))
        .route("/passkey/auth/finish", post(passkey_auth_finish::<T, C, P>))
        .with_state(store.clone())
        .layer(middleware::from_fn_with_state(
            strict_limiter,
            rate_limit_middleware,
        ));

    // Other public routes: moderate rate limiting (30 req/min per IP)
    let other_public = Router::new()
        .route(
            "/.well-known/oauth-protected-resource",
            get(protected_resource_metadata::<T, C, P>),
        )
        .route(
            "/.well-known/oauth-authorization-server",
            get(authorization_server_metadata::<T, C, P>),
        )
        .route("/authorize", get(authorize_get::<T, C, P>))
        .route("/health", get(|| async { "ok" }))
        .with_state(store.clone())
        .layer(middleware::from_fn_with_state(
            moderate_limiter,
            rate_limit_middleware,
        ));

    // M5: Security headers on all public responses
    let public_routes = auth_routes
        .merge(other_public)
        .layer(middleware::from_fn(security_headers_middleware));

    // Protected routes: lenient rate limiting (60 req/min per IP), then auth
    let protected = protected_router
        .layer(middleware::from_fn_with_state(
            store,
            auth_middleware::<T, C, P>,
        ))
        .layer(middleware::from_fn_with_state(
            lenient_limiter,
            rate_limit_middleware,
        ));

    public_routes
        .merge(protected)
        .layer(middleware::from_fn(request_logging_middleware))
}

// Request logging middleware — logs ALL incoming requests
async fn request_logging_middleware(req: axum::extract::Request, next: Next) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let has_auth = req.headers().contains_key(header::AUTHORIZATION);
    let session_id = req
        .headers()
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s[..s.len().min(12)].to_owned());
    tracing::info!(
        "-> {method} {uri} (auth={has_auth}, session={session})",
        session = session_id.as_deref().unwrap_or("none")
    );
    next.run(req).await
}

// M5: Security headers middleware
#[expect(
    clippy::unwrap_used,
    reason = "HeaderValue::from_static equivalents parsed from ASCII-only string literals cannot fail; any failure would be a compile-time bug in the literal"
)]
async fn security_headers_middleware(req: axum::extract::Request, next: Next) -> Response {
    let mut response = next.run(req).await;
    let headers = response.headers_mut();
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'"
            .parse()
            .unwrap(),
    );
    headers.insert("Referrer-Policy", "no-referrer".parse().unwrap());
    headers.insert(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=(), payment=()"
            .parse()
            .unwrap(),
    );
    response
}

// ---------------------------------------------------------------------------
// Well-known metadata endpoints
// ---------------------------------------------------------------------------

async fn protected_resource_metadata<T: TokenStore, C: ClientStore, P: PasskeyStore>(
    State(store): State<AppState<T, C, P>>,
) -> impl IntoResponse {
    let url = &store.config.server_url;
    Json(serde_json::json!({
        "resource": url,
        "authorization_servers": [url],
        "bearer_methods_supported": ["header"]
    }))
}

async fn authorization_server_metadata<T: TokenStore, C: ClientStore, P: PasskeyStore>(
    State(store): State<AppState<T, C, P>>,
) -> impl IntoResponse {
    let url = &store.config.server_url;
    let client_count = store.client_store.client_count().await.unwrap_or(0);
    let mut metadata = serde_json::json!({
        "issuer": url,
        "authorization_endpoint": format!("{url}/authorize"),
        "token_endpoint": format!("{url}/token"),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_post"],
        "scopes_supported": store.config.scopes
    });
    // Only advertise registration endpoint when no client has been registered yet
    if client_count == 0 {
        metadata["registration_endpoint"] = serde_json::json!(format!("{url}/register"));
    }
    Json(metadata)
}

// ---------------------------------------------------------------------------
// Dynamic Client Registration (RFC 7591)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RegisterClientRequest {
    client_name: Option<String>,
    redirect_uris: Vec<String>,
    #[expect(
        dead_code,
        reason = "deserialized per RFC 7591 but intentionally ignored: this server only issues authorization_code + refresh_token grants with client_secret_post auth, advertised via metadata"
    )]
    grant_types: Option<Vec<String>>,
    #[expect(
        dead_code,
        reason = "deserialized per RFC 7591 but intentionally ignored: see grant_types above"
    )]
    response_types: Option<Vec<String>>,
    #[expect(
        dead_code,
        reason = "deserialized per RFC 7591 but intentionally ignored: see grant_types above"
    )]
    token_endpoint_auth_method: Option<String>,
}

#[derive(Serialize)]
struct RegisterClientResponse {
    client_id: String,
    client_secret: String,
    client_name: String,
    redirect_uris: Vec<String>,
    grant_types: Vec<String>,
    response_types: Vec<String>,
    token_endpoint_auth_method: String,
}

async fn register_client<T: TokenStore, C: ClientStore, P: PasskeyStore>(
    State(store): State<AppState<T, C, P>>,
    Json(body): Json<RegisterClientRequest>,
) -> Result<Json<RegisterClientResponse>, (StatusCode, Json<ErrorResponse>)> {
    for uri in &body.redirect_uris {
        if !store.config.allowed_redirect_uris.iter().any(|u| u == uri) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "invalid_redirect_uri".into(),
                    error_description: Some("Redirect URI not allowed".into()),
                }),
            ));
        }
    }

    // L1: Use cryptographically strong tokens instead of UUID v4
    let client_id = generate_token();
    let client_secret = generate_token();
    let client_name = body
        .client_name
        .clone()
        .unwrap_or_else(|| "MCP Client".into());

    tracing::info!(
        "POST /register: new client_id={} name={:?} redirect_uris={:?}",
        &client_id[..8],
        client_name,
        body.redirect_uris,
    );

    // Atomically check-and-insert to prevent TOCTOU race between
    // concurrent registration requests.
    let registered = store
        .client_store
        .try_register_client(
            client_id.clone(),
            RegisteredClient {
                client_secret: client_secret.clone(),
                redirect_uris: body.redirect_uris.clone(),
            },
        )
        .await
        .map_err(|e| store_error_response("Failed to persist client registration", &e))?;

    if !registered {
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "registration_locked".into(),
                error_description: Some(
                    "Client registration is locked: the configured max_registered_clients cap has been reached."
                        .into(),
                ),
            }),
        ));
    }

    Ok(Json(RegisterClientResponse {
        client_id,
        client_secret,
        client_name,
        redirect_uris: body.redirect_uris,
        grant_types: vec!["authorization_code".into(), "refresh_token".into()],
        response_types: vec!["code".into()],
        token_endpoint_auth_method: "client_secret_post".into(),
    }))
}

// ---------------------------------------------------------------------------
// Authorization endpoint (now passkey-driven)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct AuthorizeParams {
    response_type: Option<String>,
    client_id: Option<String>,
    redirect_uri: Option<String>,
    state: Option<String>,
    code_challenge: Option<String>,
    code_challenge_method: Option<String>,
    scope: Option<String>,
    #[expect(
        dead_code,
        reason = "RFC 8707 Resource Indicator placeholder; tracked for issue #14 but not yet honoured"
    )]
    resource: Option<String>,
}

#[expect(
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "`redirect_uri` (OAuth parameter) and `redirect_url` (parsed Url for redirect building) are distinct and canonically named; the authorize flow is linear and splitting it would obscure the check-then-issue logic"
)]
async fn authorize_get<T: TokenStore, C: ClientStore, P: PasskeyStore>(
    State(store): State<AppState<T, C, P>>,
    req: axum::extract::Request,
) -> Result<Response, (StatusCode, Html<String>)> {
    let query = req.uri().query().unwrap_or("");
    let params: AuthorizeParams = match serde_urlencoded::from_str(query) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("Malformed /authorize query string: {e}");
            AuthorizeParams {
                response_type: None,
                client_id: None,
                redirect_uri: None,
                state: None,
                code_challenge: None,
                code_challenge_method: None,
                scope: None,
                resource: None,
            }
        }
    };

    let response_type = params.response_type.as_deref().unwrap_or("");
    let client_id = params.client_id.as_deref().unwrap_or("");
    let redirect_uri = params.redirect_uri.as_deref().unwrap_or("");
    let code_challenge = params.code_challenge.as_deref().unwrap_or("");
    let code_challenge_method = params.code_challenge_method.as_deref().unwrap_or("");

    if response_type != "code" {
        return Err((
            StatusCode::BAD_REQUEST,
            Html(error_page(
                &store.config.app_name,
                "Invalid response_type. Expected 'code'.",
            )),
        ));
    }
    if !store.is_known_client(client_id).await {
        return Err((
            StatusCode::BAD_REQUEST,
            Html(error_page(&store.config.app_name, "Unknown client_id.")),
        ));
    }
    if !store.is_redirect_uri_allowed(client_id, redirect_uri).await {
        return Err((
            StatusCode::BAD_REQUEST,
            Html(error_page(
                &store.config.app_name,
                "Redirect URI not allowed.",
            )),
        ));
    }
    if code_challenge_method != "S256" {
        return Err((
            StatusCode::BAD_REQUEST,
            Html(error_page(
                &store.config.app_name,
                "code_challenge_method must be S256.",
            )),
        ));
    }
    if code_challenge.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Html(error_page(
                &store.config.app_name,
                "code_challenge is required.",
            )),
        ));
    }

    // Check for valid auth session cookie — auto-approve without passkey
    let cookie_header = req
        .headers()
        .get(header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let session_cookie = cookie_header
        .split(';')
        .find_map(|c| c.trim().strip_prefix("auth_session="));

    if let Some(cookie_val) = session_cookie
        && store.validate_auth_session(cookie_val).await
    {
        tracing::info!(
            "Auto-approving /authorize via session cookie for client {}...",
            &client_id[..client_id.len().min(8)]
        );
        let code = generate_token();
        let now = now_epoch();

        if let Err(e) = store
            .token_store
            .store_auth_code(
                code.clone(),
                AuthCode {
                    client_id: client_id.to_owned(),
                    redirect_uri: redirect_uri.to_owned(),
                    code_challenge: code_challenge.to_owned(),
                    created_at: now,
                },
            )
            .await
        {
            tracing::error!("Failed to store auth code: {e}");
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Html(error_page(
                    &store.config.app_name,
                    "Too many pending authorization codes.",
                )),
            ));
        }

        // C2: Build redirect URL safely using url::Url to properly encode parameters
        let mut redirect_url = Url::parse(redirect_uri).map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                Html(error_page(&store.config.app_name, "Invalid redirect URI.")),
            )
        })?;
        {
            let mut pairs = redirect_url.query_pairs_mut();
            pairs.append_pair("code", &code);
            if let Some(state) = &params.state {
                pairs.append_pair("state", state);
            }
        }
        return Ok((
            StatusCode::FOUND,
            [(header::LOCATION, redirect_url.to_string())],
        )
            .into_response());
    }

    let has_passkeys = store.has_passkeys().await;

    Ok(Html(authorize_page(
        &store.config.app_name,
        client_id,
        redirect_uri,
        params.state.as_deref().unwrap_or(""),
        code_challenge,
        code_challenge_method,
        params.scope.as_deref().unwrap_or(""),
        has_passkeys,
    ))
    .into_response())
}

// ---------------------------------------------------------------------------
// Token endpoint
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TokenRequest {
    grant_type: String,
    code: Option<String>,
    redirect_uri: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
    code_verifier: Option<String>,
    refresh_token: Option<String>,
}

#[derive(Serialize)]
struct TokenResponse {
    access_token: String,
    token_type: String,
    expires_in: u64,
    refresh_token: String,
    scope: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    error_description: Option<String>,
}

async fn token<T: TokenStore, C: ClientStore, P: PasskeyStore>(
    State(store): State<AppState<T, C, P>>,
    Form(params): Form<TokenRequest>,
) -> Result<Json<TokenResponse>, (StatusCode, Json<ErrorResponse>)> {
    let client_id = params.client_id.as_deref().unwrap_or("");
    let client_secret = params.client_secret.as_deref().unwrap_or("");

    tracing::info!(
        "POST /token: grant_type={} client_id={}...",
        params.grant_type,
        &client_id[..client_id.len().min(8)]
    );

    if !store.validate_client(client_id, client_secret).await {
        let known = store.is_known_client(client_id).await;
        tracing::warn!(
            "POST /token: invalid client credentials for client_id={}... (client known={})",
            &client_id[..client_id.len().min(8)],
            known
        );
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "invalid_client".into(),
                error_description: Some("Invalid client credentials".into()),
            }),
        ));
    }

    match params.grant_type.as_str() {
        "authorization_code" => handle_authorization_code(&store, client_id, &params).await,
        "refresh_token" => handle_refresh_token(&store, client_id, &params).await,
        _ => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "unsupported_grant_type".into(),
                error_description: None,
            }),
        )),
    }
}

async fn handle_authorization_code<T: TokenStore, C: ClientStore, P: PasskeyStore>(
    store: &OAuthServer<T, C, P>,
    client_id: &str,
    params: &TokenRequest,
) -> Result<Json<TokenResponse>, (StatusCode, Json<ErrorResponse>)> {
    let code = params.code.as_deref().unwrap_or("");
    let redirect_uri = params.redirect_uri.as_deref().unwrap_or("");
    let code_verifier = params.code_verifier.as_deref().unwrap_or("");

    // M4: Validate PKCE code_verifier length per RFC 7636 (43-128 characters)
    if code_verifier.len() < 43 || code_verifier.len() > 128 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_grant".into(),
                error_description: Some(
                    "code_verifier must be 43-128 characters (RFC 7636)".into(),
                ),
            }),
        ));
    }

    let auth_code = store
        .token_store
        .consume_auth_code(code)
        .await
        .map_err(|e| store_error_response("Internal storage error", &e))?;

    let Some(auth_code) = auth_code else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_grant".into(),
                error_description: Some("Invalid or expired authorization code".into()),
            }),
        ));
    };

    if now_epoch().saturating_sub(auth_code.created_at) > store.config.code_lifetime_secs {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_grant".into(),
                error_description: Some("Authorization code expired".into()),
            }),
        ));
    }

    if auth_code.redirect_uri != redirect_uri {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_grant".into(),
                error_description: Some("redirect_uri mismatch".into()),
            }),
        ));
    }
    if auth_code.client_id != client_id {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_grant".into(),
                error_description: Some("client_id mismatch".into()),
            }),
        ));
    }

    let computed_challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(code_verifier.as_bytes()));
    if !constant_time_eq(&computed_challenge, &auth_code.code_challenge) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_grant".into(),
                error_description: Some("PKCE verification failed".into()),
            }),
        ));
    }

    // L1: Use cryptographically strong tokens
    let access_token = generate_token();
    let refresh_token = generate_token();

    // Store access token (capacity checks happen inside the store)
    store
        .token_store
        .store_access_token(
            access_token.clone(),
            AccessTokenEntry {
                client_id: client_id.to_owned(),
                created_at: now_epoch(),
                expires_in_secs: store.config.token_lifetime_secs,
                refresh_token: refresh_token.clone(),
            },
        )
        .await
        .map_err(|e| store_error_response("Too many active tokens", &e))?;

    store
        .token_store
        .store_refresh_token(
            refresh_token.clone(),
            RefreshTokenEntry {
                client_id: client_id.to_owned(),
            },
        )
        .await
        .map_err(|e| store_error_response("Too many active refresh tokens", &e))?;

    Ok(Json(TokenResponse {
        access_token,
        token_type: "Bearer".into(),
        expires_in: store.config.token_lifetime_secs,
        refresh_token,
        scope: store.config.scopes.join(" "),
    }))
}

async fn handle_refresh_token<T: TokenStore, C: ClientStore, P: PasskeyStore>(
    store: &OAuthServer<T, C, P>,
    client_id: &str,
    params: &TokenRequest,
) -> Result<Json<TokenResponse>, (StatusCode, Json<ErrorResponse>)> {
    let refresh_token_val = params.refresh_token.as_deref().unwrap_or("");

    // Peek at the refresh token first (non-destructive) to validate client_id
    // before consuming it. This prevents an attacker who knows a token value
    // but not the correct client_id from destroying the victim's refresh token.
    let entry = store
        .token_store
        .get_refresh_token(refresh_token_val)
        .await
        .map_err(|e| store_error_response("Internal storage error", &e))?;

    let Some(entry) = entry else {
        tracing::warn!(
            "Refresh token not found (already consumed or never existed), client_id={}...",
            &client_id[..client_id.len().min(8)]
        );
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_grant".into(),
                error_description: Some("Invalid refresh token".into()),
            }),
        ));
    };

    if entry.client_id != client_id {
        tracing::warn!(
            "Refresh token client_id mismatch: token belongs to {} but request from {}",
            &entry.client_id[..entry.client_id.len().min(8)],
            &client_id[..client_id.len().min(8)]
        );
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_grant".into(),
                error_description: Some("client_id mismatch".into()),
            }),
        ));
    }

    // Now consume (remove) the validated refresh token
    store
        .token_store
        .consume_refresh_token(refresh_token_val)
        .await
        .map_err(|e| store_error_response("Internal storage error", &e))?;

    tracing::info!(
        "Refresh token valid, issuing new tokens for client_id={}...",
        &client_id[..client_id.len().min(8)]
    );

    // M1: Only revoke the specific access token associated with the consumed refresh token,
    // not all tokens for the client
    store
        .token_store
        .revoke_access_tokens_by_refresh(refresh_token_val)
        .await
        .map_err(|e| store_error_response("Failed to revoke old access tokens", &e))?;

    // L1: Use cryptographically strong tokens
    let new_access_token = generate_token();
    let new_refresh_token = generate_token();

    store
        .token_store
        .store_access_token(
            new_access_token.clone(),
            AccessTokenEntry {
                client_id: client_id.to_owned(),
                created_at: now_epoch(),
                expires_in_secs: store.config.token_lifetime_secs,
                refresh_token: new_refresh_token.clone(),
            },
        )
        .await
        .map_err(|e| store_error_response("Failed to store access token", &e))?;

    store
        .token_store
        .store_refresh_token(
            new_refresh_token.clone(),
            RefreshTokenEntry {
                client_id: client_id.to_owned(),
            },
        )
        .await
        .map_err(|e| store_error_response("Failed to store refresh token", &e))?;

    Ok(Json(TokenResponse {
        access_token: new_access_token,
        token_type: "Bearer".into(),
        expires_in: store.config.token_lifetime_secs,
        refresh_token: new_refresh_token,
        scope: store.config.scopes.join(" "),
    }))
}

// ---------------------------------------------------------------------------
// Auth middleware for protected routes
// ---------------------------------------------------------------------------

async fn auth_middleware<T: TokenStore, C: ClientStore, P: PasskeyStore>(
    State(store): State<AppState<T, C, P>>,
    req: axum::extract::Request,
    next: Next,
) -> Result<Response, Response> {
    let auth_header = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    let Some(h) = auth_header.filter(|h| h.len() > 7 && h[..7].eq_ignore_ascii_case("bearer "))
    else {
        tracing::info!("Auth middleware: no Bearer token in request");
        return Err(unauthorized_response(&store.config.server_url));
    };
    let token = &h[7..];

    let token_prefix = &token[..token.len().min(8)];
    let now = now_epoch();
    match store.token_store.get_access_token(token).await {
        Ok(Some(entry)) if now.saturating_sub(entry.created_at) < entry.expires_in_secs => {
            tracing::info!(
                "Auth middleware: token {}... valid (age={}s)",
                token_prefix,
                now.saturating_sub(entry.created_at)
            );
            let response = next.run(req).await;
            // If the inner service returned 401 but our auth was valid,
            // it's a session-not-found error from rmcp (e.g. after server restart).
            // Convert to 404 so MCP clients create a new session instead of
            // endlessly refreshing their token.
            if response.status() == StatusCode::UNAUTHORIZED {
                tracing::info!(
                    "Auth middleware: converting inner 401 to 404 (session not found, auth was valid)"
                );
                return Ok((StatusCode::NOT_FOUND, "Session not found").into_response());
            }
            Ok(response)
        }
        Ok(Some(entry)) => {
            tracing::warn!(
                "Auth middleware: token {}... EXPIRED (age={}s, max={}s)",
                token_prefix,
                now.saturating_sub(entry.created_at),
                entry.expires_in_secs
            );
            Err(unauthorized_response(&store.config.server_url))
        }
        Ok(None) => {
            tracing::warn!("Auth middleware: token {}... NOT FOUND", token_prefix,);
            Err(unauthorized_response(&store.config.server_url))
        }
        Err(e) => {
            tracing::error!("Auth middleware: token store error: {e}");
            Err(unauthorized_response(&store.config.server_url))
        }
    }
}

/// Map a [`StoreError`] to an HTTP error response.
fn store_error_response(description: &str, err: &StoreError) -> (StatusCode, Json<ErrorResponse>) {
    tracing::error!("Store error: {err}");
    let status = match err {
        StoreError::CapacityExceeded => StatusCode::TOO_MANY_REQUESTS,
        StoreError::Backend(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (
        status,
        Json(ErrorResponse {
            error: "server_error".into(),
            error_description: Some(description.into()),
        }),
    )
}

fn unauthorized_response(server_url: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(
            header::WWW_AUTHENTICATE,
            format!(
                "Bearer realm=\"mcp-server\", resource_metadata=\"{server_url}/.well-known/oauth-protected-resource\""
            ),
        )],
        "Unauthorized",
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Passkey registration endpoints
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PasskeyRegisterStartRequest {
    setup_token: Option<String>,
}

#[derive(Serialize)]
struct PasskeyRegisterStartResponse {
    session_id: String,
    creation_options: CreationChallengeResponse,
}

#[derive(Deserialize)]
struct PasskeyRegisterFinishRequest {
    session_id: String,
    credential: RegisterPublicKeyCredential,
}

#[derive(Deserialize)]
struct PasskeyRegisterPageQuery {
    setup_token: Option<String>,
}

async fn passkey_register_page<T: TokenStore, C: ClientStore, P: PasskeyStore>(
    State(store): State<AppState<T, C, P>>,
    axum::extract::Query(query): axum::extract::Query<PasskeyRegisterPageQuery>,
) -> Html<String> {
    let has_passkeys = store.has_passkeys().await;
    if has_passkeys {
        return Html(error_page(
            &store.config.app_name,
            "Passkey registration is locked. A passkey already exists. Delete passkeys.json and restart to reset.",
        ));
    }
    Html(register_page(
        &store.config.app_name,
        has_passkeys,
        query.setup_token.as_deref(),
    ))
}

async fn passkey_register_start<T: TokenStore, C: ClientStore, P: PasskeyStore>(
    State(store): State<AppState<T, C, P>>,
    Json(body): Json<PasskeyRegisterStartRequest>,
) -> Result<Json<PasskeyRegisterStartResponse>, (StatusCode, Json<ErrorResponse>)> {
    let has_passkeys = store.has_passkeys().await;

    if has_passkeys {
        // Passkey registration permanently locked after first passkey.
        // To reset: delete passkeys.json and restart the server.
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "registration_locked".into(),
                error_description: Some(
                    "Passkey registration is locked. A passkey already exists. Delete passkeys.json and restart to reset."
                        .into(),
                ),
            }),
        ));
    }
    // First registration: require setup token
    // H1: Constant-time comparison for setup token
    let expected = store.config.setup_token.as_deref().unwrap_or("");
    let provided = body.setup_token.as_deref().unwrap_or("");
    if expected.is_empty() || !constant_time_eq(provided, expected) {
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "invalid_setup_token".into(),
                error_description: Some("Invalid or missing setup token.".into()),
            }),
        ));
    }

    let user_unique_id = [0u8; 16]; // single-user system
    let existing = store.passkey_store.list_passkeys().await.map_err(|e| {
        tracing::error!("Passkey store error: {e}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "server_error".into(),
                error_description: Some("Internal storage error".into()),
            }),
        )
    })?;
    let exclude: Vec<CredentialID> = existing.iter().map(|pk| pk.cred_id().clone()).collect();

    let (ccr, reg_state) = store
        .webauthn
        .start_passkey_registration(
            Uuid::from_bytes(user_unique_id),
            "admin",
            "Admin",
            Some(exclude),
        )
        // M6: Log full error, return generic message to client
        .map_err(|e| {
            tracing::error!("WebAuthn registration start failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "webauthn_error".into(),
                    error_description: Some("Passkey registration could not be started.".into()),
                }),
            )
        })?;

    let session_id = generate_token();

    // H2: Cleanup expired entries and enforce capacity on registration_states
    {
        let now = now_epoch();
        let mut states = store.registration_states.lock().await;
        states.retain(|_, (_, created_at)| {
            now.saturating_sub(*created_at) <= TRANSIENT_STATE_TTL_SECS
        });
        if states.len() >= store.config.capacity.max_registration_states {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: "capacity_exceeded".into(),
                    error_description: Some("Too many pending registrations".into()),
                }),
            ));
        }
        states.insert(session_id.clone(), (reg_state, now));
    }

    Ok(Json(PasskeyRegisterStartResponse {
        session_id,
        creation_options: ccr,
    }))
}

async fn passkey_register_finish<T: TokenStore, C: ClientStore, P: PasskeyStore>(
    State(store): State<AppState<T, C, P>>,
    Json(body): Json<PasskeyRegisterFinishRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let reg_state = store
        .registration_states
        .lock()
        .await
        .remove(&body.session_id)
        .map(|(state, _timestamp)| state);

    let Some(reg_state) = reg_state else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_session".into(),
                error_description: Some("Unknown or expired registration session.".into()),
            }),
        ));
    };

    let passkey = store
        .webauthn
        .finish_passkey_registration(&body.credential, &reg_state)
        // M6: Log full error, return generic message to client
        .map_err(|e| {
            tracing::error!("WebAuthn registration finish failed: {e}");
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "registration_failed".into(),
                    error_description: Some("Passkey registration failed.".into()),
                }),
            )
        })?;

    // Atomically check-and-insert to prevent TOCTOU race where multiple
    // registrations are started concurrently before the first one completes.
    let added = store
        .passkey_store
        .add_passkey_if_none(passkey)
        .await
        .map_err(|e| {
            tracing::error!("Failed to save passkey: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "storage_error".into(),
                    error_description: Some("Failed to persist passkey.".into()),
                }),
            )
        })?;

    if !added {
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "registration_locked".into(),
                error_description: Some(
                    "Passkey registration is locked. A passkey already exists.".into(),
                ),
            }),
        ));
    }

    // Invalidate all other pending registration sessions to prevent
    // concurrent registrations started before lockdown from completing.
    store.registration_states.lock().await.clear();

    Ok(Json(serde_json::json!({ "ok": true })))
}

// ---------------------------------------------------------------------------
// Passkey authentication endpoints
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PasskeyAuthStartRequest {
    client_id: String,
    redirect_uri: String,
    state: Option<String>,
    code_challenge: String,
    code_challenge_method: String,
}

#[derive(Serialize)]
struct PasskeyAuthStartResponse {
    session_id: String,
    request_options: RequestChallengeResponse,
}

#[derive(Deserialize)]
struct PasskeyAuthFinishRequest {
    session_id: String,
    credential: PublicKeyCredential,
}

#[derive(Serialize)]
struct PasskeyAuthFinishResponse {
    redirect_url: String,
}

async fn passkey_auth_start<T: TokenStore, C: ClientStore, P: PasskeyStore>(
    State(store): State<AppState<T, C, P>>,
    Json(body): Json<PasskeyAuthStartRequest>,
) -> Result<Json<PasskeyAuthStartResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Validate OAuth params
    if !store.is_known_client(&body.client_id).await {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_client".into(),
                error_description: Some("Unknown client_id.".into()),
            }),
        ));
    }
    if !store
        .is_redirect_uri_allowed(&body.client_id, &body.redirect_uri)
        .await
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_redirect_uri".into(),
                error_description: Some("Redirect URI not allowed.".into()),
            }),
        ));
    }
    if body.code_challenge_method != "S256" || body.code_challenge.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_request".into(),
                error_description: Some("Invalid PKCE parameters.".into()),
            }),
        ));
    }

    let passkeys = store.passkey_store.list_passkeys().await.map_err(|e| {
        tracing::error!("Passkey store error: {e}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "server_error".into(),
                error_description: Some("Internal storage error".into()),
            }),
        )
    })?;
    if passkeys.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "no_passkeys".into(),
                error_description: Some("No passkeys registered.".into()),
            }),
        ));
    }
    let (rcr, auth_state) = store
        .webauthn
        .start_passkey_authentication(&passkeys)
        // M6: Log full error, return generic message to client
        .map_err(|e| {
            tracing::error!("WebAuthn authentication start failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "webauthn_error".into(),
                    error_description: Some("Passkey authentication could not be started.".into()),
                }),
            )
        })?;

    let session_id = generate_token();
    let pending = PendingAuthApproval {
        client_id: body.client_id,
        redirect_uri: body.redirect_uri,
        state: body.state,
        code_challenge: body.code_challenge,
        code_challenge_method: body.code_challenge_method,
    };

    // H2: Cleanup expired entries and enforce capacity on authentication_states
    {
        let now = now_epoch();
        let mut states = store.authentication_states.lock().await;
        states.retain(|_, (_, _, created_at)| {
            now.saturating_sub(*created_at) <= TRANSIENT_STATE_TTL_SECS
        });
        if states.len() >= store.config.capacity.max_authentication_states {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: "capacity_exceeded".into(),
                    error_description: Some("Too many pending authentications".into()),
                }),
            ));
        }
        states.insert(session_id.clone(), (auth_state, pending, now));
    }

    Ok(Json(PasskeyAuthStartResponse {
        session_id,
        request_options: rcr,
    }))
}

async fn passkey_auth_finish<T: TokenStore, C: ClientStore, P: PasskeyStore>(
    State(store): State<AppState<T, C, P>>,
    Json(body): Json<PasskeyAuthFinishRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let entry = store
        .authentication_states
        .lock()
        .await
        .remove(&body.session_id);

    let Some((auth_state, pending, _timestamp)) = entry else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_session".into(),
                error_description: Some("Unknown or expired authentication session.".into()),
            }),
        ));
    };

    let auth_result = store
        .webauthn
        .finish_passkey_authentication(&body.credential, &auth_state)
        // M6: Log full error, return generic message to client
        .map_err(|e| {
            tracing::error!("WebAuthn authentication finish failed: {e}");
            (
                StatusCode::FORBIDDEN,
                Json(ErrorResponse {
                    error: "authentication_failed".into(),
                    error_description: Some("Passkey authentication failed.".into()),
                }),
            )
        })?;

    // Update credential counter for replay protection
    if let Err(e) = store.passkey_store.update_passkey(&auth_result).await {
        tracing::error!("Failed to save updated passkey counters: {e}");
    }

    // L1: Use cryptographically strong token for auth code
    let code = generate_token();
    let now = now_epoch();

    // Store auth code (capacity checks happen inside the store)
    store
        .token_store
        .store_auth_code(
            code.clone(),
            AuthCode {
                client_id: pending.client_id.clone(),
                redirect_uri: pending.redirect_uri.clone(),
                code_challenge: pending.code_challenge,
                created_at: now,
            },
        )
        .await
        .map_err(|e| {
            tracing::error!("Token store error: {e}");
            (
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: "capacity_exceeded".into(),
                    error_description: Some("Too many pending authorization codes".into()),
                }),
            )
        })?;

    // C2: Build redirect URL safely using url::Url to properly encode parameters
    let mut redirect_url = Url::parse(&pending.redirect_uri).map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_redirect_uri".into(),
                error_description: Some("Invalid redirect URI.".into()),
            }),
        )
    })?;
    {
        let mut pairs = redirect_url.query_pairs_mut();
        pairs.append_pair("code", &code);
        if let Some(state) = &pending.state {
            pairs.append_pair("state", state);
        }
    }

    // Create auth session cookie so subsequent /authorize auto-approves
    let session_token = store.create_auth_session().await;
    let cookie_value = format!(
        "auth_session={}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age={}",
        session_token, store.config.token_lifetime_secs
    );

    Ok((
        [(header::SET_COOKIE, cookie_value)],
        Json(PasskeyAuthFinishResponse {
            redirect_url: redirect_url.to_string(),
        }),
    )
        .into_response())
}

// ---------------------------------------------------------------------------
// HTML templates (Askama — compiled at build time from templates/ directory)
// ---------------------------------------------------------------------------

const COMMON_STYLE: &str = include_str!("../templates/common.css");

#[derive(Template)]
#[template(path = "error.html")]
struct ErrorTemplate<'a> {
    app_name: &'a str,
    style: &'a str,
    message: &'a str,
}

#[derive(Template)]
#[template(path = "authorize_setup.html")]
struct AuthorizeSetupTemplate<'a> {
    app_name: &'a str,
    style: &'a str,
}

#[derive(Template)]
#[template(path = "authorize.html")]
struct AuthorizeTemplate<'a> {
    app_name: &'a str,
    style: &'a str,
    params_json: &'a str,
}

#[derive(Template)]
#[template(path = "register.html")]
struct RegisterTemplate<'a> {
    app_name: &'a str,
    style: &'a str,
    title: &'a str,
    prefill_token: &'a str,
    auto_start: bool,
}

#[expect(
    clippy::too_many_arguments,
    reason = "each argument is an independent OAuth/template field; collecting them into a struct would just move the same count to the struct literal at the call site"
)]
fn authorize_page(
    app_name: &str,
    client_id: &str,
    redirect_uri: &str,
    state: &str,
    code_challenge: &str,
    code_challenge_method: &str,
    _scope: &str,
    has_passkeys: bool,
) -> String {
    if !has_passkeys {
        return AuthorizeSetupTemplate {
            app_name,
            style: COMMON_STYLE,
        }
        .render()
        .unwrap_or_default();
    }

    // C1: Serialize OAuth params as JSON and embed in a non-executable <script> data block
    // to prevent XSS via malicious parameter values.
    // In <script> tags, HTML entities are NOT decoded, so we use |safe in the template.
    // Only escape </ to prevent </script> injection.
    #[expect(
        clippy::expect_used,
        reason = "serde_json::to_string on a statically-constructed json! literal containing only &str values is infallible modulo OOM"
    )]
    let params_json = serde_json::to_string(&serde_json::json!({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
    }))
    .expect("JSON serialization of string values is infallible");
    let params_json_safe = params_json.replace("</", "<\\/");

    AuthorizeTemplate {
        app_name,
        style: COMMON_STYLE,
        params_json: &params_json_safe,
    }
    .render()
    .unwrap_or_default()
}

fn register_page(app_name: &str, has_passkeys: bool, prefill_token: Option<&str>) -> String {
    let title = if has_passkeys {
        "Register Additional Passkey"
    } else {
        "Register Your First Passkey"
    };

    RegisterTemplate {
        app_name,
        style: COMMON_STYLE,
        title,
        prefill_token: prefill_token.unwrap_or_default(),
        auto_start: prefill_token.is_some(),
    }
    .render()
    .unwrap_or_default()
}

fn error_page(app_name: &str, message: &str) -> String {
    ErrorTemplate {
        app_name,
        style: COMMON_STYLE,
        message,
    }
    .render()
    .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: invariants are established by the test fixtures themselves, so .unwrap() is idiomatic and a panic on violation is the desired test failure mode"
)]
mod tests {
    use super::*;
    use axum::routing::get as get_route;
    use axum_test::TestServer;

    fn test_config(dir: &std::path::Path) -> OAuthConfig {
        OAuthConfig::with_defaults(
            "https://mcp.example.com".into(),
            "test-client-id".into(),
            "test-client-secret".into(),
            "Test App".into(),
            dir.join("passkeys.json"),
            Some("setup-token-123".into()),
        )
    }

    fn build_test_app(dir: &std::path::Path) -> Router {
        build_test_app_with_config(test_config(dir))
    }

    fn build_test_app_with_config(config: OAuthConfig) -> Router {
        let protected = Router::new().route("/mcp", get_route(|| async { "protected content" }));
        let (token_store, client_store, passkey_store) = create_default_stores(&config);
        build_oauth_router_with_stores(protected, config, token_store, client_store, passkey_store)
    }

    // -- Unit tests for helper functions --

    #[test]
    fn test_constant_time_eq_same() {
        assert!(constant_time_eq("hello", "hello"));
    }

    #[test]
    fn test_constant_time_eq_different() {
        assert!(!constant_time_eq("hello", "world"));
    }

    #[test]
    fn test_constant_time_eq_different_lengths() {
        assert!(!constant_time_eq("short", "longer string"));
    }

    #[test]
    fn test_constant_time_eq_empty() {
        assert!(constant_time_eq("", ""));
    }

    #[test]
    fn test_generate_token_length() {
        let token = generate_token();
        // 32 bytes -> 43 base64url chars (no padding)
        assert_eq!(token.len(), 43);
    }

    #[test]
    fn test_generate_token_uniqueness() {
        let t1 = generate_token();
        let t2 = generate_token();
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_generate_token_is_base64url() {
        let token = generate_token();
        assert!(URL_SAFE_NO_PAD.decode(&token).is_ok());
    }

    #[test]
    fn test_extract_domain_valid() {
        assert_eq!(
            extract_domain("https://mcp.example.com").unwrap(),
            "mcp.example.com"
        );
    }

    #[test]
    fn test_extract_domain_with_port() {
        assert_eq!(
            extract_domain("https://mcp.example.com:8443").unwrap(),
            "mcp.example.com"
        );
    }

    #[test]
    fn test_extract_domain_invalid() {
        assert!(extract_domain("not a url").is_err());
    }

    #[test]
    fn test_now_epoch_reasonable() {
        let now = now_epoch();
        // Should be after 2024-01-01
        assert!(now > 1_704_067_200);
    }

    // -- OAuthConfig tests --

    #[test]
    fn test_config_defaults() {
        let cfg = OAuthConfig::with_defaults(
            "https://example.com".into(),
            "id".into(),
            "secret".into(),
            "App".into(),
            PathBuf::from("pk.json"),
            None,
        );
        assert_eq!(cfg.token_lifetime_secs, 86400);
        assert_eq!(cfg.code_lifetime_secs, 300);
        assert!(cfg.setup_token.is_none());
    }

    #[test]
    #[should_panic(expected = "client_id must not be empty")]
    fn test_config_empty_client_id_panics() {
        let _ = OAuthConfig::with_defaults(
            "https://example.com".into(),
            String::new(),
            "secret".into(),
            "App".into(),
            PathBuf::from("pk.json"),
            None,
        );
    }

    #[test]
    #[should_panic(expected = "client_secret must not be empty")]
    fn test_config_empty_client_secret_panics() {
        let _ = OAuthConfig::with_defaults(
            "https://example.com".into(),
            "id".into(),
            String::new(),
            "App".into(),
            PathBuf::from("pk.json"),
            None,
        );
    }

    #[test]
    #[should_panic(expected = "passkey_store_path must not contain '..' components")]
    fn test_config_rejects_path_traversal() {
        let _ = OAuthConfig::with_defaults(
            "https://example.com".into(),
            "id".into(),
            "secret".into(),
            "App".into(),
            PathBuf::from("/data/../etc/passkeys.json"),
            None,
        );
    }

    // -- Builder tests --

    #[test]
    fn test_builder_defaults_match_with_defaults() {
        let from_defaults = OAuthConfig::with_defaults(
            "https://example.com".into(),
            "id".into(),
            "secret".into(),
            "App".into(),
            PathBuf::from("pk.json"),
            None,
        );
        let from_builder = OAuthConfig::builder(
            "https://example.com".into(),
            "id".into(),
            "secret".into(),
            "App".into(),
            PathBuf::from("pk.json"),
        )
        .build()
        .unwrap();

        assert_eq!(
            from_defaults.token_lifetime_secs,
            from_builder.token_lifetime_secs
        );
        assert_eq!(
            from_defaults.code_lifetime_secs,
            from_builder.code_lifetime_secs
        );
        assert_eq!(
            from_defaults.allowed_redirect_uris,
            from_builder.allowed_redirect_uris
        );
        assert_eq!(
            from_defaults.rate_limits.strict,
            from_builder.rate_limits.strict
        );
        assert_eq!(
            from_defaults.rate_limits.moderate,
            from_builder.rate_limits.moderate
        );
        assert_eq!(
            from_defaults.rate_limits.lenient,
            from_builder.rate_limits.lenient
        );
        assert_eq!(
            from_defaults.capacity.max_registration_states,
            from_builder.capacity.max_registration_states
        );
        assert_eq!(
            from_defaults.capacity.max_authentication_states,
            from_builder.capacity.max_authentication_states
        );
        assert_eq!(
            from_defaults.capacity.max_access_tokens,
            from_builder.capacity.max_access_tokens
        );
        assert_eq!(
            from_defaults.capacity.max_refresh_tokens,
            from_builder.capacity.max_refresh_tokens
        );
        assert_eq!(
            from_defaults.capacity.max_auth_codes,
            from_builder.capacity.max_auth_codes
        );
        assert_eq!(
            from_defaults.capacity.max_registered_clients,
            from_builder.capacity.max_registered_clients
        );
        assert_eq!(from_defaults.scopes, from_builder.scopes);
    }

    #[test]
    fn test_builder_empty_client_id_fails() {
        let result = OAuthConfig::builder(
            "https://example.com".into(),
            String::new(),
            "secret".into(),
            "App".into(),
            PathBuf::from("pk.json"),
        )
        .build();
        assert!(matches!(result, Err(OAuthConfigError::EmptyClientId)));
    }

    #[test]
    fn test_builder_empty_client_secret_fails() {
        let result = OAuthConfig::builder(
            "https://example.com".into(),
            "id".into(),
            String::new(),
            "App".into(),
            PathBuf::from("pk.json"),
        )
        .build();
        assert!(matches!(result, Err(OAuthConfigError::EmptyClientSecret)));
    }

    #[test]
    fn test_builder_path_traversal_fails() {
        let result = OAuthConfig::builder(
            "https://example.com".into(),
            "id".into(),
            "secret".into(),
            "App".into(),
            PathBuf::from("/data/../etc/passkeys.json"),
        )
        .build();
        assert!(matches!(result, Err(OAuthConfigError::PathTraversal)));
    }

    #[test]
    fn test_builder_zero_rate_limit_fails() {
        let result = OAuthConfig::builder(
            "https://example.com".into(),
            "id".into(),
            "secret".into(),
            "App".into(),
            PathBuf::from("pk.json"),
        )
        .rate_limits(RateLimitConfig {
            strict: 0,
            moderate: 30,
            lenient: 60,
        })
        .build();
        assert!(matches!(result, Err(OAuthConfigError::ZeroRateLimit)));
    }

    #[test]
    fn test_builder_empty_scopes_fails() {
        let result = OAuthConfig::builder(
            "https://example.com".into(),
            "id".into(),
            "secret".into(),
            "App".into(),
            PathBuf::from("pk.json"),
        )
        .scopes(vec![])
        .build();
        assert!(matches!(result, Err(OAuthConfigError::EmptyScopes)));
    }

    #[test]
    fn test_builder_custom_redirect_uris_replaces() {
        let cfg = OAuthConfig::builder(
            "https://example.com".into(),
            "id".into(),
            "secret".into(),
            "App".into(),
            PathBuf::from("pk.json"),
        )
        .allowed_redirect_uris(vec!["https://custom.example.com/cb".to_owned()])
        .build()
        .unwrap();
        assert_eq!(
            cfg.allowed_redirect_uris,
            vec!["https://custom.example.com/cb"]
        );
    }

    #[test]
    fn test_builder_add_redirect_uri_appends() {
        let cfg = OAuthConfig::builder(
            "https://example.com".into(),
            "id".into(),
            "secret".into(),
            "App".into(),
            PathBuf::from("pk.json"),
        )
        .add_redirect_uri("https://custom.example.com/cb")
        .build()
        .unwrap();
        assert_eq!(
            cfg.allowed_redirect_uris.len(),
            default_redirect_uris().len() + 1
        );
        assert!(
            cfg.allowed_redirect_uris
                .contains(&"https://claude.ai/api/mcp/auth_callback".to_owned())
        );
        assert!(
            cfg.allowed_redirect_uris
                .contains(&"https://custom.example.com/cb".to_owned())
        );
    }

    #[test]
    fn test_builder_custom_scopes() {
        let cfg = OAuthConfig::builder(
            "https://example.com".into(),
            "id".into(),
            "secret".into(),
            "App".into(),
            PathBuf::from("pk.json"),
        )
        .scopes(vec!["read".to_owned(), "write".to_owned()])
        .build()
        .unwrap();
        assert_eq!(cfg.scopes, vec!["read", "write"]);
    }

    #[test]
    fn test_builder_add_scope_appends() {
        let cfg = OAuthConfig::builder(
            "https://example.com".into(),
            "id".into(),
            "secret".into(),
            "App".into(),
            PathBuf::from("pk.json"),
        )
        .add_scope("admin")
        .build()
        .unwrap();
        assert_eq!(cfg.scopes, vec!["mcp:tools", "admin"]);
    }

    #[test]
    fn test_builder_zero_max_access_tokens_fails() {
        let result = OAuthConfig::builder(
            "https://example.com".into(),
            "id".into(),
            "secret".into(),
            "App".into(),
            PathBuf::from("pk.json"),
        )
        .max_access_tokens(0)
        .build();
        assert!(matches!(result, Err(OAuthConfigError::ZeroCapacity)));
    }

    #[test]
    fn test_builder_some_zero_max_registered_clients_fails() {
        let result = OAuthConfig::builder(
            "https://example.com".into(),
            "id".into(),
            "secret".into(),
            "App".into(),
            PathBuf::from("pk.json"),
        )
        .max_registered_clients(Some(0))
        .build();
        assert!(matches!(result, Err(OAuthConfigError::ZeroCapacity)));
    }

    #[test]
    fn test_builder_none_max_registered_clients_allowed() {
        let cfg = OAuthConfig::builder(
            "https://example.com".into(),
            "id".into(),
            "secret".into(),
            "App".into(),
            PathBuf::from("pk.json"),
        )
        .max_registered_clients(None)
        .build()
        .unwrap();
        assert_eq!(cfg.capacity.max_registered_clients, None);
    }

    #[test]
    fn test_oauth_config_error_display_all_variants() {
        // Each variant's Display output must contain a recognisable substring
        // so error messages remain useful in logs/telemetry.
        assert!(
            OAuthConfigError::EmptyClientId
                .to_string()
                .contains("client_id")
        );
        assert!(
            OAuthConfigError::EmptyClientSecret
                .to_string()
                .contains("client_secret")
        );
        assert!(
            OAuthConfigError::PathTraversal
                .to_string()
                .contains("passkey_store_path")
        );
        assert!(
            OAuthConfigError::ZeroRateLimit
                .to_string()
                .contains("rate limit")
        );
        assert!(OAuthConfigError::EmptyScopes.to_string().contains("scopes"));
        assert!(
            OAuthConfigError::ZeroCapacity
                .to_string()
                .contains("capacity")
        );
    }

    // -- Integration tests --

    #[tokio::test]
    async fn test_health_endpoint() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server.get("/health").await;
        resp.assert_status_ok();
        resp.assert_text("ok");
    }

    #[tokio::test]
    async fn test_protected_resource_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server.get("/.well-known/oauth-protected-resource").await;
        resp.assert_status_ok();
        let body: serde_json::Value = resp.json();
        assert_eq!(body["resource"], "https://mcp.example.com");
        assert_eq!(
            body["bearer_methods_supported"],
            serde_json::json!(["header"])
        );
    }

    #[tokio::test]
    async fn test_authorization_server_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server.get("/.well-known/oauth-authorization-server").await;
        resp.assert_status_ok();
        let body: serde_json::Value = resp.json();
        assert_eq!(body["issuer"], "https://mcp.example.com");
        assert_eq!(
            body["authorization_endpoint"],
            "https://mcp.example.com/authorize"
        );
        assert_eq!(body["token_endpoint"], "https://mcp.example.com/token");
        assert_eq!(
            body["code_challenge_methods_supported"],
            serde_json::json!(["S256"])
        );
        // Registration should be advertised when no clients registered
        assert!(body["registration_endpoint"].is_string());
    }

    #[tokio::test]
    async fn test_protected_route_requires_auth() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server.get("/mcp").await;
        resp.assert_status(StatusCode::UNAUTHORIZED);
        // Should include WWW-Authenticate header
        let www_auth = resp.header("WWW-Authenticate");
        assert!(www_auth.to_str().unwrap().contains("Bearer"));
    }

    #[tokio::test]
    async fn test_protected_route_invalid_token() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server
            .get("/mcp")
            .add_header(
                header::AUTHORIZATION,
                "Bearer invalid-token"
                    .parse::<axum::http::HeaderValue>()
                    .unwrap(),
            )
            .await;
        resp.assert_status(StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_token_invalid_client() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server
            .post("/token")
            .form(&serde_json::json!({
                "grant_type": "authorization_code",
                "client_id": "wrong",
                "client_secret": "wrong",
                "code": "abc",
                "redirect_uri": "https://example.com",
                "code_verifier": "x"
            }))
            .await;
        resp.assert_status(StatusCode::UNAUTHORIZED);
        let body: serde_json::Value = resp.json();
        assert_eq!(body["error"], "invalid_client");
    }

    #[tokio::test]
    async fn test_token_unsupported_grant_type() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server
            .post("/token")
            .form(&serde_json::json!({
                "grant_type": "client_credentials",
                "client_id": "test-client-id",
                "client_secret": "test-client-secret"
            }))
            .await;
        resp.assert_status(StatusCode::BAD_REQUEST);
        let body: serde_json::Value = resp.json();
        assert_eq!(body["error"], "unsupported_grant_type");
    }

    #[tokio::test]
    async fn test_authorize_missing_params() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        // Missing response_type
        let resp = server.get("/authorize").await;
        resp.assert_status(StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_authorize_invalid_response_type() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server
            .get("/authorize?response_type=token&client_id=test-client-id&redirect_uri=https://claude.ai/api/mcp/auth_callback&code_challenge=abc&code_challenge_method=S256")
            .await;
        resp.assert_status(StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_authorize_unknown_client() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server
            .get("/authorize?response_type=code&client_id=unknown&redirect_uri=https://claude.ai/api/mcp/auth_callback&code_challenge=abc&code_challenge_method=S256")
            .await;
        resp.assert_status(StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_authorize_disallowed_redirect_uri() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server
            .get("/authorize?response_type=code&client_id=test-client-id&redirect_uri=https://evil.com/callback&code_challenge=abc&code_challenge_method=S256")
            .await;
        resp.assert_status(StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_authorize_wrong_code_challenge_method() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server
            .get("/authorize?response_type=code&client_id=test-client-id&redirect_uri=https://claude.ai/api/mcp/auth_callback&code_challenge=abc&code_challenge_method=plain")
            .await;
        resp.assert_status(StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_authorize_valid_params_shows_setup_page() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        // No passkeys registered, so should show setup page
        let resp = server
            .get("/authorize?response_type=code&client_id=test-client-id&redirect_uri=https://claude.ai/api/mcp/auth_callback&code_challenge=abc&code_challenge_method=S256")
            .await;
        resp.assert_status_ok();
        let body = resp.text();
        assert!(
            body.contains("setup")
                || body.contains("Setup")
                || body.contains("register")
                || body.contains("Register")
        );
    }

    #[tokio::test]
    async fn test_passkey_register_without_setup_token() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server
            .post("/passkey/register/start")
            .json(&serde_json::json!({}))
            .await;
        resp.assert_status(StatusCode::FORBIDDEN);
        let body: serde_json::Value = resp.json();
        assert_eq!(body["error"], "invalid_setup_token");
    }

    #[tokio::test]
    async fn test_passkey_register_wrong_setup_token() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server
            .post("/passkey/register/start")
            .json(&serde_json::json!({ "setup_token": "wrong-token" }))
            .await;
        resp.assert_status(StatusCode::FORBIDDEN);
        let body: serde_json::Value = resp.json();
        assert_eq!(body["error"], "invalid_setup_token");
    }

    #[tokio::test]
    async fn test_passkey_register_valid_setup_token() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server
            .post("/passkey/register/start")
            .json(&serde_json::json!({ "setup_token": "setup-token-123" }))
            .await;
        resp.assert_status_ok();
        let body: serde_json::Value = resp.json();
        assert!(body["session_id"].is_string());
        assert!(body["creation_options"].is_object());
    }

    #[tokio::test]
    async fn test_register_client_first_time() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server
            .post("/register")
            .json(&serde_json::json!({
                "client_name": "My Client",
                "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "client_secret_post"
            }))
            .await;
        resp.assert_status_ok();
        let body: serde_json::Value = resp.json();
        assert!(body["client_id"].is_string());
        assert!(body["client_secret"].is_string());
        assert_eq!(body["client_name"], "My Client");
    }

    #[tokio::test]
    async fn test_register_client_locks_after_first() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        // First registration succeeds
        let resp = server
            .post("/register")
            .json(&serde_json::json!({
                "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"]
            }))
            .await;
        resp.assert_status_ok();

        // Second registration is locked
        let resp = server
            .post("/register")
            .json(&serde_json::json!({
                "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"]
            }))
            .await;
        resp.assert_status(StatusCode::FORBIDDEN);
        let body: serde_json::Value = resp.json();
        assert_eq!(body["error"], "registration_locked");
    }

    #[tokio::test]
    async fn test_register_client_invalid_redirect_uri() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server
            .post("/register")
            .json(&serde_json::json!({
                "redirect_uris": ["https://evil.com/callback"]
            }))
            .await;
        resp.assert_status(StatusCode::BAD_REQUEST);
        let body: serde_json::Value = resp.json();
        assert_eq!(body["error"], "invalid_redirect_uri");
    }

    #[tokio::test]
    async fn test_security_headers_present() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server.get("/health").await;
        resp.assert_status_ok();
        assert_eq!(resp.header("X-Frame-Options").to_str().unwrap(), "DENY");
        assert_eq!(
            resp.header("X-Content-Type-Options").to_str().unwrap(),
            "nosniff"
        );
        assert_eq!(
            resp.header("Referrer-Policy").to_str().unwrap(),
            "no-referrer"
        );
        assert!(
            resp.header("Content-Security-Policy")
                .to_str()
                .unwrap()
                .contains("default-src 'self'")
        );
        assert!(
            resp.header("Permissions-Policy")
                .to_str()
                .unwrap()
                .contains("camera=()")
        );
    }

    #[tokio::test]
    async fn test_pkce_code_verifier_too_short() {
        let dir = tempfile::tempdir().unwrap();
        let server = TestServer::new(build_test_app(dir.path()));

        let resp = server
            .post("/token")
            .form(&serde_json::json!({
                "grant_type": "authorization_code",
                "client_id": "test-client-id",
                "client_secret": "test-client-secret",
                "code": "abc",
                "redirect_uri": "https://example.com",
                "code_verifier": "tooshort"
            }))
            .await;
        resp.assert_status(StatusCode::BAD_REQUEST);
        let body: serde_json::Value = resp.json();
        assert_eq!(body["error"], "invalid_grant");
        assert!(
            body["error_description"]
                .as_str()
                .unwrap()
                .contains("43-128")
        );
    }

    // -- Persistence tests --

    #[test]
    fn test_atomic_write_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.json");
        store::json_file::atomic_write(&path, b"hello").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    #[test]
    fn test_atomic_write_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("dir").join("test.json");
        store::json_file::atomic_write(&path, b"nested").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "nested");
    }

    #[test]
    fn test_load_passkeys_missing_file() {
        let passkeys =
            store::json_file::load_passkeys(std::path::Path::new("/nonexistent/passkeys.json"));
        assert!(passkeys.is_empty());
    }

    #[test]
    fn test_load_tokens_missing_file() {
        let caps = store::json_file::StoreCaps {
            max_access_tokens: 10,
            max_refresh_tokens: 10,
            max_auth_codes: 10,
            max_registered_clients: Some(1),
        };
        let (_, _, summary) = store::json_file::create_json_file_stores(
            std::path::Path::new("/nonexistent/passkeys.json"),
            caps,
        );
        assert_eq!(summary.access_tokens, 0);
        assert_eq!(summary.refresh_tokens, 0);
        assert_eq!(summary.registered_clients, 0);
    }

    // -- Template rendering tests --

    #[test]
    fn test_error_page_renders() {
        let html = error_page("Test App", "Something went wrong");
        assert!(html.contains("Test App"));
        assert!(html.contains("Something went wrong"));
    }

    #[test]
    fn test_authorize_page_no_passkeys_shows_setup() {
        let html = authorize_page("App", "cid", "https://r.com", "", "ch", "S256", "", false);
        // Should render setup template when no passkeys
        assert!(html.contains("App"));
    }

    #[test]
    fn test_authorize_page_with_passkeys_embeds_params() {
        let html = authorize_page("App", "cid", "https://r.com", "st", "ch", "S256", "", true);
        assert!(html.contains("App"));
        // Should embed OAuth params as JSON
        assert!(html.contains("cid"));
    }

    #[test]
    fn test_authorize_page_xss_prevention() {
        // Verify that </script> in params doesn't break out of the JSON block
        let html = authorize_page(
            "App",
            "</script><script>alert(1)",
            "https://r.com",
            "",
            "ch",
            "S256",
            "",
            true,
        );
        assert!(!html.contains("</script><script>"));
        assert!(html.contains("<\\/script>"));
    }

    #[test]
    fn test_register_page_renders() {
        let html = register_page("App", false, Some("tok123"));
        assert!(html.contains("App"));
        assert!(html.contains("tok123"));
    }

    // -- Builder integration tests --

    #[tokio::test]
    async fn test_custom_redirect_uri_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let config = OAuthConfig::builder(
            "https://mcp.example.com".into(),
            "test-client-id".into(),
            "test-client-secret".into(),
            "Test App".into(),
            dir.path().join("passkeys.json"),
        )
        .setup_token("setup-token-123")
        .add_redirect_uri("https://custom.example.com/callback")
        .build()
        .unwrap();
        let server = TestServer::new(build_test_app_with_config(config));

        let resp = server
            .post("/register")
            .json(&serde_json::json!({
                "redirect_uris": ["https://custom.example.com/callback"]
            }))
            .await;
        resp.assert_status_ok();
    }

    #[tokio::test]
    async fn test_default_redirect_uri_rejected_when_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let config = OAuthConfig::builder(
            "https://mcp.example.com".into(),
            "test-client-id".into(),
            "test-client-secret".into(),
            "Test App".into(),
            dir.path().join("passkeys.json"),
        )
        .setup_token("setup-token-123")
        .allowed_redirect_uris(vec!["https://custom.example.com/callback".to_owned()])
        .build()
        .unwrap();
        let server = TestServer::new(build_test_app_with_config(config));

        let resp = server
            .post("/register")
            .json(&serde_json::json!({
                "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"]
            }))
            .await;
        resp.assert_status(StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_custom_scope_in_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let config = OAuthConfig::builder(
            "https://mcp.example.com".into(),
            "test-client-id".into(),
            "test-client-secret".into(),
            "Test App".into(),
            dir.path().join("passkeys.json"),
        )
        .setup_token("setup-token-123")
        .scopes(vec!["read".to_owned(), "write".to_owned()])
        .build()
        .unwrap();
        let server = TestServer::new(build_test_app_with_config(config));

        let resp = server.get("/.well-known/oauth-authorization-server").await;
        resp.assert_status_ok();
        let body: serde_json::Value = resp.json();
        assert_eq!(
            body["scopes_supported"],
            serde_json::json!(["read", "write"])
        );
    }

    #[tokio::test]
    async fn test_register_client_cap_of_two_accepts_two_then_rejects() {
        // Configure a 2-client cap and verify the first two registrations
        // succeed while the third is rejected with `registration_locked`.
        let dir = tempfile::tempdir().unwrap();
        let config = OAuthConfig::builder(
            "https://mcp.example.com".into(),
            "test-client-id".into(),
            "test-client-secret".into(),
            "Test App".into(),
            dir.path().join("passkeys.json"),
        )
        .setup_token("setup-token-123")
        .max_registered_clients(Some(2))
        .build()
        .unwrap();
        let server = TestServer::new(build_test_app_with_config(config));

        for _ in 0..2 {
            let resp = server
                .post("/register")
                .json(&serde_json::json!({
                    "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"]
                }))
                .await;
            resp.assert_status_ok();
        }

        let resp = server
            .post("/register")
            .json(&serde_json::json!({
                "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"]
            }))
            .await;
        resp.assert_status(StatusCode::FORBIDDEN);
        let body: serde_json::Value = resp.json();
        assert_eq!(body["error"], "registration_locked");
    }
}
