//! # mcp-oauth
//!
//! A reusable OAuth 2.1 layer for [MCP](https://modelcontextprotocol.io)
//! (Model Context Protocol) servers, designed for compatibility with Claude.ai.
//!
//! This crate is not a standalone binary — consumers import it and call
//! [`build_oauth_router`] to wrap their [axum](https://docs.rs/axum) `Router`
//! with a complete OAuth 2.1 implementation.
//!
//! ## Features
//!
//! - **OAuth 2.1 with PKCE** (S256) — authorization code flow with proof key
//! - **Dynamic client registration** ([RFC 7591](https://www.rfc-editor.org/rfc/rfc7591))
//! - **`WebAuthn` / passkey authentication** — passwordless approval via hardware keys or biometrics
//! - **Token refresh** — long-lived sessions via refresh tokens
//! - **Per-IP rate limiting** — three tiers (auth, registration, general)
//! - **In-memory state** with TTL-based cleanup — no database required
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use axum::Router;
//! use mcp_oauth::{OAuthConfig, build_oauth_router};
//! use std::path::PathBuf;
//!
//! let mcp_routes = Router::new(); // your protected MCP routes
//!
//! let config = OAuthConfig::with_defaults(
//!     "https://my-mcp.example.com".into(),
//!     "my-client-id".into(),
//!     "my-client-secret".into(),
//!     "My MCP Server".into(),
//!     PathBuf::from("passkeys.json"),
//!     Some("initial-setup-token".into()),
//! );
//!
//! let app = build_oauth_router(mcp_routes, config);
//! // Serve `app` with axum / hyper as usual.
//! ```

use std::collections::HashMap;
use std::io::Write as _;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
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
#[allow(clippy::expect_used)]
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

        Self {
            server_url,
            client_id,
            client_secret,
            app_name,
            passkey_store_path,
            setup_token,
            token_lifetime_secs: 86400,
            code_lifetime_secs: 300,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal stores
// ---------------------------------------------------------------------------

struct AuthCode {
    client_id: String,
    redirect_uri: String,
    code_challenge: String,
    created_at: u64, // epoch seconds
}

#[derive(Serialize, Deserialize, Clone)]
struct AccessTokenEntry {
    client_id: String,
    created_at: u64, // epoch seconds
    expires_in_secs: u64,
    refresh_token: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct RefreshTokenEntry {
    client_id: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct RegisteredClient {
    client_secret: String,
    redirect_uris: Vec<String>,
}

// Persisted state for tokens and registered clients
#[derive(Serialize, Deserialize, Default)]
struct PersistedTokens {
    access_tokens: HashMap<String, AccessTokenEntry>,
    refresh_tokens: HashMap<String, RefreshTokenEntry>,
    registered_clients: HashMap<String, RegisteredClient>,
}

#[derive(Clone)]
struct PendingAuthApproval {
    client_id: String,
    redirect_uri: String,
    state: Option<String>,
    code_challenge: String,
    #[allow(dead_code)]
    code_challenge_method: String,
}

// H2: Capacity limits to prevent memory exhaustion DoS
const MAX_AUTH_CODES: usize = 10_000;
const MAX_ACCESS_TOKENS: usize = 10_000;
const MAX_REFRESH_TOKENS: usize = 10_000;
const MAX_REGISTRATION_STATES: usize = 10_000;
const MAX_AUTHENTICATION_STATES: usize = 10_000;
const TRANSIENT_STATE_TTL_SECS: u64 = 300;

struct OAuthStore {
    config: OAuthConfig,
    auth_codes: Mutex<HashMap<String, AuthCode>>,
    access_tokens: Mutex<HashMap<String, AccessTokenEntry>>,
    refresh_tokens: Mutex<HashMap<String, RefreshTokenEntry>>,
    registered_clients: Mutex<HashMap<String, RegisteredClient>>,
    // Passkey / WebAuthn state
    webauthn: Webauthn,
    passkeys: Mutex<Vec<Passkey>>,
    passkey_store_path: PathBuf,
    // H2: Timestamps added for TTL-based cleanup
    registration_states: Mutex<HashMap<String, (PasskeyRegistration, u64)>>,
    authentication_states:
        Mutex<HashMap<String, (PasskeyAuthentication, PendingAuthApproval, u64)>>,
    // Session cookie for auto-approving /authorize after first passkey auth
    auth_session_token: Mutex<Option<(String, u64)>>, // (token, created_at_epoch)
}

const ALLOWED_REDIRECT_URIS: &[&str] = &[
    "https://claude.ai/api/mcp/auth_callback",
    "https://claude.com/api/mcp/auth_callback",
];

type AppState = Arc<OAuthStore>;

// L3: Return Result instead of silently falling back to "localhost"
fn extract_domain(server_url: &str) -> Result<String, String> {
    Url::parse(server_url)
        .ok()
        .and_then(|u| u.host_str().map(ToString::to_string))
        .ok_or_else(|| format!("cannot extract domain from URL: {server_url}"))
}

fn load_passkeys(path: &Path) -> Vec<Passkey> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

// M3: Atomic file write with restrictive permissions (0o600 on Unix).
//
// SECURITY NOTE: Persisted token files (tokens.json, passkeys.json) contain
// plaintext secrets. Ensure the data directory is owned by the service user
// and not world-readable. On a public-facing deployment, consider mounting
// the data directory on a tmpfs or encrypted filesystem.
fn atomic_write(path: &Path, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp_path = path.with_extension("tmp");
    {
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut file = opts.open(&temp_path)?;
        file.write_all(data)?;
        file.flush()?;
    }
    std::fs::rename(&temp_path, path)?;
    Ok(())
}

fn save_passkeys(path: &Path, passkeys: &[Passkey]) -> Result<(), Box<dyn std::error::Error>> {
    atomic_write(path, serde_json::to_string_pretty(passkeys)?.as_bytes())
}

fn tokens_path(passkey_path: &Path) -> PathBuf {
    passkey_path.with_file_name("tokens.json")
}

fn load_tokens(path: &Path) -> PersistedTokens {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_tokens(path: &Path, tokens: &PersistedTokens) -> Result<(), Box<dyn std::error::Error>> {
    atomic_write(path, serde_json::to_string_pretty(tokens)?.as_bytes())
}

impl OAuthStore {
    #[allow(clippy::expect_used)]
    fn new(config: OAuthConfig) -> Self {
        let rp_id =
            extract_domain(&config.server_url).expect("invalid server_url: cannot extract domain");
        let rp_origin = Url::parse(&config.server_url).expect("invalid server_url");
        let webauthn = WebauthnBuilder::new(&rp_id, &rp_origin)
            .expect("Failed to build WebAuthn")
            .rp_name(&config.app_name)
            .build()
            .expect("Failed to build WebAuthn");

        let passkeys = load_passkeys(&config.passkey_store_path);
        let passkey_store_path = config.passkey_store_path.clone();
        let tp = tokens_path(&passkey_store_path);
        let persisted = load_tokens(&tp);

        tracing::info!(
            "OAuth store loaded: {} passkeys, {} access_tokens, {} refresh_tokens, {} registered_clients from {:?}",
            passkeys.len(),
            persisted.access_tokens.len(),
            persisted.refresh_tokens.len(),
            persisted.registered_clients.len(),
            tp,
        );
        tracing::info!(
            "Token/passkey files are stored at {:?}. Ensure this directory is owned by the service user with 0o700 permissions.",
            passkey_store_path
                .parent()
                .unwrap_or_else(|| Path::new(".")),
        );

        if passkeys.is_empty() {
            tracing::warn!("=== FIRST-TIME SETUP REQUIRED ===");
            tracing::warn!(
                "No passkeys registered. Visit {}/passkey/register?setup_token=<SETUP_TOKEN> to register one.",
                config.server_url
            );
            tracing::warn!("Built-in OAuth client credentials (provide these to your MCP client):");
            tracing::warn!("  Client ID:     {}", config.client_id);
            tracing::warn!("  Client Secret: (set via OAUTH_CLIENT_SECRET env var)");
            tracing::warn!("After registering a passkey, registration will be permanently locked.");
        } else {
            tracing::info!(
                "Server is locked: {} passkey(s) registered, {} dynamic client(s).",
                passkeys.len(),
                persisted.registered_clients.len()
            );
        }

        Self {
            config,
            auth_codes: Mutex::new(HashMap::new()),
            access_tokens: Mutex::new(persisted.access_tokens),
            refresh_tokens: Mutex::new(persisted.refresh_tokens),
            registered_clients: Mutex::new(persisted.registered_clients),
            webauthn,
            passkeys: Mutex::new(passkeys),
            passkey_store_path,
            registration_states: Mutex::new(HashMap::new()),
            authentication_states: Mutex::new(HashMap::new()),
            auth_session_token: Mutex::new(None),
        }
    }

    // H1: Constant-time secret comparison to prevent timing side-channels
    async fn validate_client(&self, client_id: &str, client_secret: &str) -> bool {
        let id_match = constant_time_eq(client_id, &self.config.client_id);
        let secret_match = constant_time_eq(client_secret, &self.config.client_secret);
        if id_match && secret_match {
            return true;
        }
        let clients = self.registered_clients.lock().await;
        clients
            .get(client_id)
            .is_some_and(|c| constant_time_eq(client_secret, &c.client_secret))
    }

    async fn is_known_client(&self, client_id: &str) -> bool {
        if client_id == self.config.client_id {
            return true;
        }
        self.registered_clients.lock().await.contains_key(client_id)
    }

    async fn is_redirect_uri_allowed(&self, client_id: &str, redirect_uri: &str) -> bool {
        if ALLOWED_REDIRECT_URIS.contains(&redirect_uri) {
            return true;
        }
        let clients = self.registered_clients.lock().await;
        clients
            .get(client_id)
            .is_some_and(|c| c.redirect_uris.iter().any(|u| u == redirect_uri))
    }

    async fn has_passkeys(&self) -> bool {
        !self.passkeys.lock().await.is_empty()
    }

    // M2: Acquire all locks atomically for consistent snapshot
    async fn persist_tokens(&self) {
        let access = self.access_tokens.lock().await;
        let refresh = self.refresh_tokens.lock().await;
        let clients = self.registered_clients.lock().await;
        let persisted = PersistedTokens {
            access_tokens: access.clone(),
            refresh_tokens: refresh.clone(),
            registered_clients: clients.clone(),
        };
        drop(clients);
        drop(refresh);
        drop(access);
        if let Err(e) = save_tokens(&tokens_path(&self.passkey_store_path), &persisted) {
            tracing::error!("Failed to persist tokens: {e}");
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

#[allow(clippy::unwrap_used)]
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
// Public API: build_oauth_router
// ---------------------------------------------------------------------------

/// Wraps `protected_router` with OAuth 2.1 endpoints and Bearer-token middleware.
///
/// # Rate Limiting
///
/// Three tiers of per-IP rate limiting (keyed by `CF-Connecting-IP` header):
/// - **Strict (10 req/min):** `/token`, `/register`, `/passkey/*` — auth brute-force protection
/// - **Moderate (30 req/min):** `/authorize`, `/.well-known/*`, `/health` — OAuth flow, metadata
/// - **Lenient (60 req/min):** `/mcp` (protected routes) — already behind Bearer auth
pub fn build_oauth_router(protected_router: Router, config: OAuthConfig) -> Router {
    let store: AppState = Arc::new(OAuthStore::new(config));

    let strict_limiter = create_rate_limiter(10);
    let moderate_limiter = create_rate_limiter(30);
    let lenient_limiter = create_rate_limiter(60);

    // Auth routes: strict rate limiting (10 req/min per IP)
    let auth_routes = Router::new()
        .route("/register", post(register_client))
        .route("/token", post(token))
        .route("/passkey/register", get(passkey_register_page))
        .route("/passkey/register/start", post(passkey_register_start))
        .route("/passkey/register/finish", post(passkey_register_finish))
        .route("/passkey/auth/start", post(passkey_auth_start))
        .route("/passkey/auth/finish", post(passkey_auth_finish))
        .with_state(store.clone())
        .layer(middleware::from_fn_with_state(
            strict_limiter,
            rate_limit_middleware,
        ));

    // Other public routes: moderate rate limiting (30 req/min per IP)
    let other_public = Router::new()
        .route(
            "/.well-known/oauth-protected-resource",
            get(protected_resource_metadata),
        )
        .route(
            "/.well-known/oauth-authorization-server",
            get(authorization_server_metadata),
        )
        .route("/authorize", get(authorize_get))
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
        .layer(middleware::from_fn_with_state(store, auth_middleware))
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
#[allow(clippy::unwrap_used)]
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

async fn protected_resource_metadata(State(store): State<AppState>) -> impl IntoResponse {
    let url = &store.config.server_url;
    Json(serde_json::json!({
        "resource": url,
        "authorization_servers": [url],
        "bearer_methods_supported": ["header"]
    }))
}

async fn authorization_server_metadata(State(store): State<AppState>) -> impl IntoResponse {
    let url = &store.config.server_url;
    let has_clients = !store.registered_clients.lock().await.is_empty();
    let mut metadata = serde_json::json!({
        "issuer": url,
        "authorization_endpoint": format!("{url}/authorize"),
        "token_endpoint": format!("{url}/token"),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_post"],
        "scopes_supported": ["mcp:tools"]
    });
    // Only advertise registration endpoint when no client has been registered yet
    if !has_clients {
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
    #[allow(dead_code)]
    grant_types: Option<Vec<String>>,
    #[allow(dead_code)]
    response_types: Option<Vec<String>>,
    #[allow(dead_code)]
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

#[allow(clippy::needless_pass_by_value)]
async fn register_client(
    State(store): State<AppState>,
    Json(body): Json<RegisterClientRequest>,
) -> Result<Json<RegisterClientResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Hold the lock for the entire function to prevent TOCTOU race conditions.
    // Lock dynamic client registration after first client is registered.
    // To reset: delete tokens.json and restart the server.
    let mut clients = store.registered_clients.lock().await;
    if !clients.is_empty() {
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "registration_locked".into(),
                error_description: Some(
                    "Client registration is locked. A client already exists. Delete tokens.json and restart to reset."
                        .into(),
                ),
            }),
        ));
    }

    for uri in &body.redirect_uris {
        if !ALLOWED_REDIRECT_URIS.contains(&uri.as_str()) {
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

    clients.insert(
        client_id.clone(),
        RegisteredClient {
            client_secret: client_secret.clone(),
            redirect_uris: body.redirect_uris.clone(),
        },
    );
    drop(clients);
    store.persist_tokens().await;

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
    #[allow(dead_code)]
    resource: Option<String>,
}

#[allow(
    clippy::similar_names,
    clippy::cognitive_complexity,
    clippy::needless_pass_by_value,
    clippy::too_many_lines
)]
async fn authorize_get(
    State(store): State<AppState>,
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
        let code_ttl = store.config.code_lifetime_secs;
        let mut codes = store.auth_codes.lock().await;
        codes.retain(|_, v| now.saturating_sub(v.created_at) <= code_ttl);
        codes.insert(
            code.clone(),
            AuthCode {
                client_id: client_id.to_owned(),
                redirect_uri: redirect_uri.to_owned(),
                code_challenge: code_challenge.to_owned(),
                created_at: now,
            },
        );
        drop(codes);

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

#[allow(clippy::needless_pass_by_value)]
async fn token(
    State(store): State<AppState>,
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

#[allow(clippy::too_many_lines)]
async fn handle_authorization_code(
    store: &OAuthStore,
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

    let Some(auth_code) = store.auth_codes.lock().await.remove(code) else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_grant".into(),
                error_description: Some("Invalid or expired authorization code".into()),
            }),
        ));
    };

    if now_epoch() - auth_code.created_at > store.config.code_lifetime_secs {
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

    // H2: Cleanup expired tokens and enforce capacity before inserting
    {
        let now = now_epoch();
        let mut tokens = store.access_tokens.lock().await;
        tokens.retain(|_, v| now - v.created_at < v.expires_in_secs);
        if tokens.len() >= MAX_ACCESS_TOKENS {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: "capacity_exceeded".into(),
                    error_description: Some("Too many active tokens".into()),
                }),
            ));
        }
        tokens.insert(
            access_token.clone(),
            AccessTokenEntry {
                client_id: client_id.to_owned(),
                created_at: now,
                expires_in_secs: store.config.token_lifetime_secs,
                refresh_token: refresh_token.clone(),
            },
        );
    }
    {
        let mut tokens = store.refresh_tokens.lock().await;
        if tokens.len() >= MAX_REFRESH_TOKENS {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: "capacity_exceeded".into(),
                    error_description: Some("Too many active refresh tokens".into()),
                }),
            ));
        }
        tokens.insert(
            refresh_token.clone(),
            RefreshTokenEntry {
                client_id: client_id.to_owned(),
            },
        );
    }
    store.persist_tokens().await;

    Ok(Json(TokenResponse {
        access_token,
        token_type: "Bearer".into(),
        expires_in: store.config.token_lifetime_secs,
        refresh_token,
        scope: "mcp:tools".into(),
    }))
}

async fn handle_refresh_token(
    store: &OAuthStore,
    client_id: &str,
    params: &TokenRequest,
) -> Result<Json<TokenResponse>, (StatusCode, Json<ErrorResponse>)> {
    let refresh_token_val = params.refresh_token.as_deref().unwrap_or("");

    let Some(entry) = store.refresh_tokens.lock().await.remove(refresh_token_val) else {
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

    tracing::info!(
        "Refresh token valid, issuing new tokens for client_id={}...",
        &client_id[..client_id.len().min(8)]
    );

    // M1: Only revoke the specific access token associated with the consumed refresh token,
    // not all tokens for the client
    store
        .access_tokens
        .lock()
        .await
        .retain(|_, v| v.refresh_token != refresh_token_val);

    // L1: Use cryptographically strong tokens
    let new_access_token = generate_token();
    let new_refresh_token = generate_token();

    store.access_tokens.lock().await.insert(
        new_access_token.clone(),
        AccessTokenEntry {
            client_id: client_id.to_owned(),
            created_at: now_epoch(),
            expires_in_secs: store.config.token_lifetime_secs,
            refresh_token: new_refresh_token.clone(),
        },
    );
    store.refresh_tokens.lock().await.insert(
        new_refresh_token.clone(),
        RefreshTokenEntry {
            client_id: client_id.to_owned(),
        },
    );
    store.persist_tokens().await;

    Ok(Json(TokenResponse {
        access_token: new_access_token,
        token_type: "Bearer".into(),
        expires_in: store.config.token_lifetime_secs,
        refresh_token: new_refresh_token,
        scope: "mcp:tools".into(),
    }))
}

// ---------------------------------------------------------------------------
// Auth middleware for protected routes
// ---------------------------------------------------------------------------

async fn auth_middleware(
    State(store): State<AppState>,
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
    let tokens = store.access_tokens.lock().await;
    let now = now_epoch();
    match tokens.get(token) {
        Some(entry) if now - entry.created_at < entry.expires_in_secs => {
            tracing::info!(
                "Auth middleware: token {}... valid (age={}s)",
                token_prefix,
                now - entry.created_at
            );
            drop(tokens);
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
        Some(entry) => {
            tracing::warn!(
                "Auth middleware: token {}... EXPIRED (age={}s, max={}s)",
                token_prefix,
                now - entry.created_at,
                entry.expires_in_secs
            );
            drop(tokens);
            Err(unauthorized_response(&store.config.server_url))
        }
        None => {
            tracing::warn!(
                "Auth middleware: token {}... NOT FOUND ({} tokens in store)",
                token_prefix,
                tokens.len()
            );
            drop(tokens);
            Err(unauthorized_response(&store.config.server_url))
        }
    }
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

async fn passkey_register_page(
    State(store): State<AppState>,
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

#[allow(clippy::needless_pass_by_value)]
async fn passkey_register_start(
    State(store): State<AppState>,
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
    let existing = store.passkeys.lock().await;
    let exclude: Vec<CredentialID> = existing.iter().map(|pk| pk.cred_id().clone()).collect();
    drop(existing);

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
        states.retain(|_, (_, created_at)| now - *created_at <= TRANSIENT_STATE_TTL_SECS);
        if states.len() >= MAX_REGISTRATION_STATES {
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

#[allow(clippy::needless_pass_by_value, clippy::significant_drop_tightening)]
async fn passkey_register_finish(
    State(store): State<AppState>,
    Json(body): Json<PasskeyRegisterFinishRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    // Reject if a passkey already exists (prevents TOCTOU race where multiple
    // registrations are started concurrently before the first one completes).
    if store.has_passkeys().await {
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

    {
        let mut passkeys = store.passkeys.lock().await;
        passkeys.push(passkey);
        if let Err(e) = save_passkeys(&store.passkey_store_path, &passkeys) {
            tracing::error!("Failed to save passkeys: {e}");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "storage_error".into(),
                    error_description: Some("Failed to persist passkey.".into()),
                }),
            ));
        }
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

#[allow(clippy::needless_pass_by_value, clippy::significant_drop_tightening)]
async fn passkey_auth_start(
    State(store): State<AppState>,
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

    let passkeys = store.passkeys.lock().await;
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
        states.retain(|_, (_, _, created_at)| now - *created_at <= TRANSIENT_STATE_TTL_SECS);
        if states.len() >= MAX_AUTHENTICATION_STATES {
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

#[allow(clippy::needless_pass_by_value)]
async fn passkey_auth_finish(
    State(store): State<AppState>,
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
    let mut passkeys = store.passkeys.lock().await;
    for pk in passkeys.iter_mut() {
        pk.update_credential(&auth_result);
    }
    if let Err(e) = save_passkeys(&store.passkey_store_path, &passkeys) {
        tracing::error!("Failed to save updated passkey counters: {e}");
    }
    drop(passkeys);

    // L1: Use cryptographically strong token for auth code
    let code = generate_token();

    // H2: Cleanup expired auth codes and enforce capacity
    {
        let now = now_epoch();
        let code_ttl = store.config.code_lifetime_secs;
        let mut codes = store.auth_codes.lock().await;
        codes.retain(|_, v| now - v.created_at <= code_ttl);
        if codes.len() >= MAX_AUTH_CODES {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: "capacity_exceeded".into(),
                    error_description: Some("Too many pending authorization codes".into()),
                }),
            ));
        }
        codes.insert(
            code.clone(),
            AuthCode {
                client_id: pending.client_id.clone(),
                redirect_uri: pending.redirect_uri.clone(),
                code_challenge: pending.code_challenge,
                created_at: now,
            },
        );
    }

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

#[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::expect_used)]
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
#[allow(clippy::unwrap_used)]
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
        let protected = Router::new().route("/mcp", get_route(|| async { "protected content" }));
        build_oauth_router(protected, test_config(dir))
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
        atomic_write(&path, b"hello").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    #[test]
    fn test_atomic_write_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("dir").join("test.json");
        atomic_write(&path, b"nested").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "nested");
    }

    #[test]
    fn test_load_passkeys_missing_file() {
        let passkeys = load_passkeys(Path::new("/nonexistent/passkeys.json"));
        assert!(passkeys.is_empty());
    }

    #[test]
    fn test_load_tokens_missing_file() {
        let tokens = load_tokens(Path::new("/nonexistent/tokens.json"));
        assert!(tokens.access_tokens.is_empty());
        assert!(tokens.refresh_tokens.is_empty());
        assert!(tokens.registered_clients.is_empty());
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
}
