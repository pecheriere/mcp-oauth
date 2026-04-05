//! Pluggable storage traits for the OAuth layer.
//!
//! The library ships with JSON-file-backed implementations
//! ([`json_file`] module) that replicate the original in-memory +
//! file-persistence behaviour.  Consumers can implement these traits
//! to back the OAuth layer with `SQLite`, an encrypted store, or any
//! other backend.

pub mod json_file;

use std::fmt;
use std::future::Future;

use serde::{Deserialize, Serialize};
use webauthn_rs::prelude::{AuthenticationResult, Passkey};

// ---------------------------------------------------------------------------
// Public data types (used in trait signatures)
// ---------------------------------------------------------------------------

/// An authorization code awaiting exchange for tokens.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[non_exhaustive]
pub struct AuthCode {
    pub client_id: String,
    pub redirect_uri: String,
    pub code_challenge: String,
    pub created_at: u64,
}

impl AuthCode {
    #[must_use]
    pub const fn new(
        client_id: String,
        redirect_uri: String,
        code_challenge: String,
        created_at: u64,
    ) -> Self {
        Self {
            client_id,
            redirect_uri,
            code_challenge,
            created_at,
        }
    }
}

/// A stored access token.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[non_exhaustive]
pub struct AccessTokenEntry {
    pub client_id: String,
    pub created_at: u64,
    pub expires_in_secs: u64,
    pub refresh_token: String,
}

impl AccessTokenEntry {
    #[must_use]
    pub const fn new(
        client_id: String,
        created_at: u64,
        expires_in_secs: u64,
        refresh_token: String,
    ) -> Self {
        Self {
            client_id,
            created_at,
            expires_in_secs,
            refresh_token,
        }
    }
}

/// A stored refresh token.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[non_exhaustive]
pub struct RefreshTokenEntry {
    pub client_id: String,
}

impl RefreshTokenEntry {
    #[must_use]
    pub const fn new(client_id: String) -> Self {
        Self { client_id }
    }
}

/// A dynamically registered OAuth client.
#[derive(Serialize, Deserialize, Clone)]
#[non_exhaustive]
pub struct RegisteredClient {
    pub client_secret: String,
    pub redirect_uris: Vec<String>,
}

impl RegisteredClient {
    #[must_use]
    pub const fn new(client_secret: String, redirect_uris: Vec<String>) -> Self {
        Self {
            client_secret,
            redirect_uris,
        }
    }
}

// Manual Debug impl to redact client_secret
impl fmt::Debug for RegisteredClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredClient")
            .field("client_secret", &"[REDACTED]")
            .field("redirect_uris", &self.redirect_uris)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// TTL for transient state entries (auth codes, registration/authentication sessions).
pub const TRANSIENT_STATE_TTL_SECS: u64 = 300;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by store operations.
#[derive(Debug)]
pub enum StoreError {
    /// The store has reached its capacity limit.
    CapacityExceeded,
    /// A backend-specific error (I/O, serialization, …).
    Backend(Box<dyn std::error::Error + Send + Sync>),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityExceeded => write!(f, "store capacity exceeded"),
            Self::Backend(e) => write!(f, "store backend error: {e}"),
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(e) => Some(&**e),
            Self::CapacityExceeded => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Storage traits
// ---------------------------------------------------------------------------

/// Token storage: authorization codes, access tokens, refresh tokens.
pub trait TokenStore: Send + Sync + 'static {
    /// Store an authorization code.
    fn store_auth_code(
        &self,
        code: String,
        entry: AuthCode,
    ) -> impl Future<Output = Result<(), StoreError>> + Send;

    /// Remove and return an authorization code (single-use).
    fn consume_auth_code(
        &self,
        code: &str,
    ) -> impl Future<Output = Result<Option<AuthCode>, StoreError>> + Send;

    /// Store an access token.
    fn store_access_token(
        &self,
        token: String,
        entry: AccessTokenEntry,
    ) -> impl Future<Output = Result<(), StoreError>> + Send;

    /// Retrieve an access token without removing it.
    fn get_access_token(
        &self,
        token: &str,
    ) -> impl Future<Output = Result<Option<AccessTokenEntry>, StoreError>> + Send;

    /// Revoke all access tokens associated with the given refresh token.
    fn revoke_access_tokens_by_refresh(
        &self,
        refresh_token: &str,
    ) -> impl Future<Output = Result<(), StoreError>> + Send;

    /// Store a refresh token.
    fn store_refresh_token(
        &self,
        token: String,
        entry: RefreshTokenEntry,
    ) -> impl Future<Output = Result<(), StoreError>> + Send;

    /// Look up a refresh token without removing it.
    fn get_refresh_token(
        &self,
        token: &str,
    ) -> impl Future<Output = Result<Option<RefreshTokenEntry>, StoreError>> + Send;

    /// Consume (remove and return) a refresh token.
    fn consume_refresh_token(
        &self,
        token: &str,
    ) -> impl Future<Output = Result<Option<RefreshTokenEntry>, StoreError>> + Send;

    /// Remove tokens whose `created_at + expires_in_secs < now`.
    fn cleanup_expired_tokens(
        &self,
        now: u64,
    ) -> impl Future<Output = Result<(), StoreError>> + Send;
}

/// Client registration storage.
pub trait ClientStore: Send + Sync + 'static {
    /// Register a new dynamic client.
    fn register_client(
        &self,
        id: String,
        client: RegisteredClient,
    ) -> impl Future<Output = Result<(), StoreError>> + Send;

    /// Atomically register a client if the store is under its configured
    /// client cap.
    ///
    /// Returns `Ok(true)` if the client was registered, `Ok(false)` if the
    /// cap has been reached (registration locked). Implementations **must**
    /// check the count and insert under the same lock to prevent TOCTOU
    /// races.
    ///
    /// A cap of `Some(1)` (the default in [`crate::CapacityConfig`]) preserves
    /// the historical single-client lock. `None` means unlimited dynamic
    /// client registrations.
    fn try_register_client(
        &self,
        id: String,
        client: RegisteredClient,
    ) -> impl Future<Output = Result<bool, StoreError>> + Send;

    /// Look up a registered client by ID.
    fn get_client(
        &self,
        id: &str,
    ) -> impl Future<Output = Result<Option<RegisteredClient>, StoreError>> + Send;

    /// Return the number of registered clients.
    fn client_count(&self) -> impl Future<Output = Result<usize, StoreError>> + Send;
}

/// Passkey (`WebAuthn` credential) storage.
pub trait PasskeyStore: Send + Sync + 'static {
    /// Return all registered passkeys.
    fn list_passkeys(&self) -> impl Future<Output = Result<Vec<Passkey>, StoreError>> + Send;

    /// Atomically add a passkey only if no passkeys exist yet.
    ///
    /// Returns `Ok(true)` if the passkey was added, `Ok(false)` if
    /// passkeys already exist (registration locked).  Implementations
    /// **must** check emptiness and insert under the same lock.
    fn add_passkey_if_none(
        &self,
        passkey: Passkey,
    ) -> impl Future<Output = Result<bool, StoreError>> + Send;

    /// Persist a newly registered passkey.
    fn add_passkey(&self, passkey: Passkey) -> impl Future<Output = Result<(), StoreError>> + Send;

    /// Update credential counters after a successful authentication.
    fn update_passkey(
        &self,
        auth_result: &AuthenticationResult,
    ) -> impl Future<Output = Result<(), StoreError>> + Send;

    /// Check whether any passkeys are registered.
    fn has_passkeys(&self) -> impl Future<Output = Result<bool, StoreError>> + Send;
}
