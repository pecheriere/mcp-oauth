//! JSON-file-backed implementations of the storage traits.
//!
//! This is the default backend, replicating the original behaviour:
//! in-memory `HashMap`s with atomic JSON file persistence on every
//! mutation that affects durable state.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use webauthn_rs::prelude::{AuthenticationResult, Passkey};

use super::{
    AccessTokenEntry, AuthCode, ClientStore, PasskeyStore, RefreshTokenEntry, RegisteredClient,
    StoreError, TokenStore,
};

// ---------------------------------------------------------------------------
// Capacity limits (caller-provided via StoreCaps)
// ---------------------------------------------------------------------------

/// Capacity limits injected into the JSON-file stores at construction.
///
/// Mirrors the relevant fields from [`crate::CapacityConfig`]; kept as a
/// separate crate-private type so `create_json_file_stores` has a stable
/// signature independent of future additions to the public config.
#[derive(Debug, Clone, Copy)]
#[expect(
    clippy::struct_field_names,
    reason = "`max_` prefix carries meaning (cap semantics) and mirrors field names on the public CapacityConfig"
)]
pub(crate) struct StoreCaps {
    pub(crate) max_access_tokens: usize,
    pub(crate) max_refresh_tokens: usize,
    pub(crate) max_auth_codes: usize,
    pub(crate) max_registered_clients: Option<usize>,
}

use super::TRANSIENT_STATE_TTL_SECS;

// ---------------------------------------------------------------------------
// Shared persistence state (tokens + clients live in the same file)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Default)]
struct PersistedTokens {
    access_tokens: HashMap<String, AccessTokenEntry>,
    refresh_tokens: HashMap<String, RefreshTokenEntry>,
    registered_clients: HashMap<String, RegisteredClient>,
}

struct SharedState {
    access_tokens: HashMap<String, AccessTokenEntry>,
    refresh_tokens: HashMap<String, RefreshTokenEntry>,
    registered_clients: HashMap<String, RegisteredClient>,
    /// Auth codes are *not* persisted (short TTL, in-memory only).
    auth_codes: HashMap<String, AuthCode>,
    tokens_path: PathBuf,
    caps: StoreCaps,
}

impl SharedState {
    /// Persist the three durable maps atomically.
    fn persist(&self) -> Result<(), StoreError> {
        let persisted = PersistedTokens {
            access_tokens: self.access_tokens.clone(),
            refresh_tokens: self.refresh_tokens.clone(),
            registered_clients: self.registered_clients.clone(),
        };
        save_tokens(&self.tokens_path, &persisted).map_err(StoreError::Backend)
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

/// Create the default JSON-file-backed token and client stores.
///
/// Both stores share the same underlying state so that persistence
/// (to `tokens.json`) remains atomic. Returns `(token_store, client_store)`
/// plus a summary of what was loaded for logging.
#[must_use]
pub(crate) fn create_json_file_stores(
    passkey_store_path: &Path,
    caps: StoreCaps,
) -> (impl TokenStore, impl ClientStore, StoreSummary) {
    let tp = tokens_path(passkey_store_path);
    let persisted = load_tokens(&tp);

    let summary = StoreSummary {
        access_tokens: persisted.access_tokens.len(),
        refresh_tokens: persisted.refresh_tokens.len(),
        registered_clients: persisted.registered_clients.len(),
        tokens_path: tp.clone(),
    };

    let shared = Arc::new(Mutex::new(SharedState {
        access_tokens: persisted.access_tokens,
        refresh_tokens: persisted.refresh_tokens,
        registered_clients: persisted.registered_clients,
        auth_codes: HashMap::new(),
        tokens_path: tp,
        caps,
    }));

    let token_store = JsonFileTokenStore {
        state: Arc::clone(&shared),
    };
    let client_store = JsonFileClientStore { state: shared };

    (token_store, client_store, summary)
}

/// Summary of data loaded from disk (for startup logging).
pub(crate) struct StoreSummary {
    pub access_tokens: usize,
    pub refresh_tokens: usize,
    pub registered_clients: usize,
    pub tokens_path: PathBuf,
}

// ---------------------------------------------------------------------------
// JsonFileTokenStore
// ---------------------------------------------------------------------------

/// JSON-file-backed [`TokenStore`].
pub struct JsonFileTokenStore {
    state: Arc<Mutex<SharedState>>,
}

impl TokenStore for JsonFileTokenStore {
    async fn store_auth_code(&self, code: String, entry: AuthCode) -> Result<(), StoreError> {
        let mut s = self.state.lock().await;
        // Clean up expired codes before inserting
        let now = crate::now_epoch();
        s.auth_codes
            .retain(|_, v| now.saturating_sub(v.created_at) <= TRANSIENT_STATE_TTL_SECS);
        if s.auth_codes.len() >= s.caps.max_auth_codes {
            return Err(StoreError::CapacityExceeded);
        }
        s.auth_codes.insert(code, entry);
        drop(s);
        Ok(())
    }

    async fn consume_auth_code(&self, code: &str) -> Result<Option<AuthCode>, StoreError> {
        let mut s = self.state.lock().await;
        Ok(s.auth_codes.remove(code))
    }

    async fn store_access_token(
        &self,
        token: String,
        entry: AccessTokenEntry,
    ) -> Result<(), StoreError> {
        let mut s = self.state.lock().await;
        // Clean up expired tokens
        let now = crate::now_epoch();
        s.access_tokens
            .retain(|_, v| now.saturating_sub(v.created_at) < v.expires_in_secs);
        if s.access_tokens.len() >= s.caps.max_access_tokens {
            return Err(StoreError::CapacityExceeded);
        }
        s.access_tokens.insert(token, entry);
        s.persist()
    }

    async fn get_access_token(&self, token: &str) -> Result<Option<AccessTokenEntry>, StoreError> {
        let s = self.state.lock().await;
        Ok(s.access_tokens.get(token).cloned())
    }

    async fn revoke_access_tokens_by_refresh(&self, refresh_token: &str) -> Result<(), StoreError> {
        let mut s = self.state.lock().await;
        s.access_tokens
            .retain(|_, v| v.refresh_token != refresh_token);
        s.persist()
    }

    async fn store_refresh_token(
        &self,
        token: String,
        entry: RefreshTokenEntry,
    ) -> Result<(), StoreError> {
        let mut s = self.state.lock().await;
        if s.refresh_tokens.len() >= s.caps.max_refresh_tokens {
            return Err(StoreError::CapacityExceeded);
        }
        s.refresh_tokens.insert(token, entry);
        s.persist()
    }

    async fn get_refresh_token(
        &self,
        token: &str,
    ) -> Result<Option<RefreshTokenEntry>, StoreError> {
        let s = self.state.lock().await;
        Ok(s.refresh_tokens.get(token).cloned())
    }

    async fn consume_refresh_token(
        &self,
        token: &str,
    ) -> Result<Option<RefreshTokenEntry>, StoreError> {
        let mut s = self.state.lock().await;
        let entry = s.refresh_tokens.remove(token);
        if entry.is_some() {
            s.persist()?;
            drop(s);
        }
        Ok(entry)
    }

    async fn cleanup_expired_tokens(&self, now: u64) -> Result<(), StoreError> {
        let mut s = self.state.lock().await;
        let before = s.access_tokens.len();
        s.access_tokens
            .retain(|_, v| now.saturating_sub(v.created_at) < v.expires_in_secs);
        if s.access_tokens.len() != before {
            s.persist()?;
            drop(s);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// JsonFileClientStore
// ---------------------------------------------------------------------------

/// JSON-file-backed [`ClientStore`].
pub struct JsonFileClientStore {
    state: Arc<Mutex<SharedState>>,
}

impl ClientStore for JsonFileClientStore {
    async fn register_client(
        &self,
        id: String,
        client: RegisteredClient,
    ) -> Result<(), StoreError> {
        let mut s = self.state.lock().await;
        s.registered_clients.insert(id, client);
        s.persist()
    }

    async fn try_register_client(
        &self,
        id: String,
        client: RegisteredClient,
    ) -> Result<bool, StoreError> {
        use std::collections::hash_map::Entry;

        let mut s = self.state.lock().await;
        if let Some(cap) = s.caps.max_registered_clients
            && s.registered_clients.len() >= cap
        {
            return Ok(false);
        }
        // Defensive: never clobber an existing client. Callers that supply
        // their own id (custom stores, future handlers) must not be able to
        // silently overwrite a registered client's secret or redirect URIs.
        match s.registered_clients.entry(id) {
            Entry::Occupied(_) => Ok(false),
            Entry::Vacant(slot) => {
                slot.insert(client);
                s.persist()?;
                drop(s);
                Ok(true)
            }
        }
    }

    async fn get_client(&self, id: &str) -> Result<Option<RegisteredClient>, StoreError> {
        let s = self.state.lock().await;
        Ok(s.registered_clients.get(id).cloned())
    }

    async fn client_count(&self) -> Result<usize, StoreError> {
        let s = self.state.lock().await;
        Ok(s.registered_clients.len())
    }
}

// ---------------------------------------------------------------------------
// JsonFilePasskeyStore
// ---------------------------------------------------------------------------

/// JSON-file-backed [`PasskeyStore`].
pub struct JsonFilePasskeyStore {
    passkeys: Mutex<Vec<Passkey>>,
    path: PathBuf,
}

impl JsonFilePasskeyStore {
    /// Create a new passkey store, loading existing passkeys from `path`.
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        let passkeys = load_passkeys(&path);
        Self {
            passkeys: Mutex::new(passkeys),
            path,
        }
    }

    /// Return the number of passkeys loaded at construction time.
    ///
    /// This is intended for startup logging only.
    pub async fn passkey_count(&self) -> usize {
        self.passkeys.lock().await.len()
    }
}

impl PasskeyStore for JsonFilePasskeyStore {
    async fn list_passkeys(&self) -> Result<Vec<Passkey>, StoreError> {
        Ok(self.passkeys.lock().await.clone())
    }

    async fn add_passkey_if_none(&self, passkey: Passkey) -> Result<bool, StoreError> {
        let mut pks = self.passkeys.lock().await;
        if !pks.is_empty() {
            return Ok(false);
        }
        pks.push(passkey);
        save_passkeys(&self.path, &pks).map_err(StoreError::Backend)?;
        drop(pks);
        Ok(true)
    }

    async fn add_passkey(&self, passkey: Passkey) -> Result<(), StoreError> {
        let mut pks = self.passkeys.lock().await;
        pks.push(passkey);
        let result = save_passkeys(&self.path, &pks).map_err(StoreError::Backend);
        drop(pks);
        result
    }

    async fn update_passkey(&self, auth_result: &AuthenticationResult) -> Result<(), StoreError> {
        let mut pks = self.passkeys.lock().await;
        for pk in pks.iter_mut() {
            pk.update_credential(auth_result);
        }
        let result = save_passkeys(&self.path, &pks).map_err(StoreError::Backend);
        drop(pks);
        result
    }

    async fn has_passkeys(&self) -> Result<bool, StoreError> {
        Ok(!self.passkeys.lock().await.is_empty())
    }
}

// ---------------------------------------------------------------------------
// File I/O helpers (moved from lib.rs)
// ---------------------------------------------------------------------------

// M3: Atomic file write with restrictive permissions (0o600 on Unix).
//
// SECURITY NOTE: Persisted token files (tokens.json, passkeys.json) contain
// plaintext secrets. Ensure the data directory is owned by the service user
// and not world-readable. On a public-facing deployment, consider mounting
// the data directory on a tmpfs or encrypted filesystem.
pub(crate) fn atomic_write(
    path: &Path,
    data: &[u8],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

pub(crate) fn load_passkeys(path: &Path) -> Vec<Passkey> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_passkeys(
    path: &Path,
    passkeys: &[Passkey],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

fn save_tokens(
    path: &Path,
    tokens: &PersistedTokens,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    atomic_write(path, serde_json::to_string_pretty(tokens)?.as_bytes())
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: invariants are established by the test fixtures themselves, so .unwrap() is idiomatic and a panic on violation is the desired test failure mode"
)]
mod tests {
    use super::*;
    use crate::store::{ClientStore, PasskeyStore, TokenStore};

    /// Generous caps for tests that don't care about capacity enforcement.
    fn large_caps() -> StoreCaps {
        StoreCaps {
            max_access_tokens: 10_000,
            max_refresh_tokens: 10_000,
            max_auth_codes: 10_000,
            max_registered_clients: Some(1),
        }
    }

    // -- TokenStore tests --

    #[tokio::test]
    async fn test_store_and_consume_auth_code() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _, _) =
            create_json_file_stores(&dir.path().join("passkeys.json"), large_caps());

        let code = AuthCode::new("cid".into(), "uri".into(), "ch".into(), 1000);
        store.store_auth_code("code1".into(), code).await.unwrap();

        // First consume returns the entry
        let entry = store.consume_auth_code("code1").await.unwrap();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().client_id, "cid");

        // Second consume returns None (single-use)
        let entry = store.consume_auth_code("code1").await.unwrap();
        assert!(entry.is_none());
    }

    #[tokio::test]
    async fn test_store_and_get_access_token() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _, _) =
            create_json_file_stores(&dir.path().join("passkeys.json"), large_caps());

        let entry = AccessTokenEntry::new("cid".into(), 1000, 3600, "rt1".into());
        store.store_access_token("at1".into(), entry).await.unwrap();

        let got = store.get_access_token("at1").await.unwrap();
        assert!(got.is_some());
        let got = got.unwrap();
        assert_eq!(got.client_id, "cid");
        assert_eq!(got.expires_in_secs, 3600);
        assert_eq!(got.refresh_token, "rt1");

        // Non-existent token returns None
        assert!(store.get_access_token("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_store_and_consume_refresh_token() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _, _) =
            create_json_file_stores(&dir.path().join("passkeys.json"), large_caps());

        let entry = RefreshTokenEntry::new("cid".into());
        store
            .store_refresh_token("rt1".into(), entry)
            .await
            .unwrap();

        // Peek (non-destructive)
        let got = store.get_refresh_token("rt1").await.unwrap();
        assert!(got.is_some());

        // Consume
        let got = store.consume_refresh_token("rt1").await.unwrap();
        assert!(got.is_some());
        assert_eq!(got.unwrap().client_id, "cid");

        // Second consume returns None
        assert!(store.consume_refresh_token("rt1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_revoke_access_tokens_by_refresh() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _, _) =
            create_json_file_stores(&dir.path().join("passkeys.json"), large_caps());

        // Store two access tokens with different refresh tokens
        let now = crate::now_epoch();
        store
            .store_access_token(
                "at1".into(),
                AccessTokenEntry::new("cid".into(), now, 3600, "rt-a".into()),
            )
            .await
            .unwrap();
        store
            .store_access_token(
                "at2".into(),
                AccessTokenEntry::new("cid".into(), now, 3600, "rt-b".into()),
            )
            .await
            .unwrap();

        // Revoke only tokens associated with rt-a
        store.revoke_access_tokens_by_refresh("rt-a").await.unwrap();

        // at1 should be gone, at2 should remain
        assert!(store.get_access_token("at1").await.unwrap().is_none());
        assert!(store.get_access_token("at2").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_cleanup_expired_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _, _) =
            create_json_file_stores(&dir.path().join("passkeys.json"), large_caps());

        let now = crate::now_epoch();
        // Expired token (created 10000s ago, expires in 3600s)
        store
            .store_access_token(
                "expired".into(),
                AccessTokenEntry::new("cid".into(), now - 10000, 3600, "rt1".into()),
            )
            .await
            .unwrap();
        // Fresh token
        store
            .store_access_token(
                "fresh".into(),
                AccessTokenEntry::new("cid".into(), now, 3600, "rt2".into()),
            )
            .await
            .unwrap();

        store.cleanup_expired_tokens(now).await.unwrap();

        assert!(store.get_access_token("expired").await.unwrap().is_none());
        assert!(store.get_access_token("fresh").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_auth_code_capacity_exceeded() {
        let dir = tempfile::tempdir().unwrap();
        let caps = StoreCaps {
            max_auth_codes: 3,
            ..large_caps()
        };
        let (store, _, _) = create_json_file_stores(&dir.path().join("passkeys.json"), caps);

        let now = crate::now_epoch();
        for i in 0..3 {
            store
                .store_auth_code(
                    format!("code-{i}"),
                    AuthCode::new("cid".into(), "uri".into(), "ch".into(), now),
                )
                .await
                .unwrap();
        }

        let result = store
            .store_auth_code(
                "overflow".into(),
                AuthCode::new("cid".into(), "uri".into(), "ch".into(), now),
            )
            .await;
        assert!(matches!(result, Err(StoreError::CapacityExceeded)));
    }

    #[tokio::test]
    async fn test_access_token_capacity_exceeded() {
        let dir = tempfile::tempdir().unwrap();
        let caps = StoreCaps {
            max_access_tokens: 3,
            ..large_caps()
        };
        let (store, _, _) = create_json_file_stores(&dir.path().join("passkeys.json"), caps);

        let now = crate::now_epoch();
        for i in 0..3 {
            store
                .store_access_token(
                    format!("at-{i}"),
                    AccessTokenEntry::new("cid".into(), now, 3600, format!("rt-{i}")),
                )
                .await
                .unwrap();
        }

        let result = store
            .store_access_token(
                "overflow".into(),
                AccessTokenEntry::new("cid".into(), now, 3600, "rt-overflow".into()),
            )
            .await;
        assert!(matches!(result, Err(StoreError::CapacityExceeded)));
    }

    #[tokio::test]
    async fn test_refresh_token_capacity_exceeded() {
        let dir = tempfile::tempdir().unwrap();
        let caps = StoreCaps {
            max_refresh_tokens: 3,
            ..large_caps()
        };
        let (store, _, _) = create_json_file_stores(&dir.path().join("passkeys.json"), caps);

        for i in 0..3 {
            store
                .store_refresh_token(format!("rt-{i}"), RefreshTokenEntry::new("cid".into()))
                .await
                .unwrap();
        }

        let result = store
            .store_refresh_token("overflow".into(), RefreshTokenEntry::new("cid".into()))
            .await;
        assert!(matches!(result, Err(StoreError::CapacityExceeded)));
    }

    #[tokio::test]
    async fn test_token_persistence_across_reload() {
        let dir = tempfile::tempdir().unwrap();
        let passkey_path = dir.path().join("passkeys.json");

        // Store tokens and a client
        {
            let (token_store, client_store, _) =
                create_json_file_stores(&passkey_path, large_caps());
            let now = crate::now_epoch();
            token_store
                .store_access_token(
                    "at1".into(),
                    AccessTokenEntry::new("cid".into(), now, 86400, "rt1".into()),
                )
                .await
                .unwrap();
            token_store
                .store_refresh_token("rt1".into(), RefreshTokenEntry::new("cid".into()))
                .await
                .unwrap();
            client_store
                .register_client(
                    "client1".into(),
                    RegisteredClient::new("secret".into(), vec!["uri".into()]),
                )
                .await
                .unwrap();
        }

        // Reload from same path
        let (token_store, client_store, summary) =
            create_json_file_stores(&passkey_path, large_caps());
        assert_eq!(summary.access_tokens, 1);
        assert_eq!(summary.refresh_tokens, 1);
        assert_eq!(summary.registered_clients, 1);

        // Verify data survived
        assert!(token_store.get_access_token("at1").await.unwrap().is_some());
        assert!(
            token_store
                .get_refresh_token("rt1")
                .await
                .unwrap()
                .is_some()
        );
        assert!(client_store.get_client("client1").await.unwrap().is_some());
    }

    // -- ClientStore tests --

    #[tokio::test]
    async fn test_try_register_client_default_single_client_cap() {
        // Default cap is Some(1) — the historical single-client lock.
        let dir = tempfile::tempdir().unwrap();
        let (_, client_store, _) =
            create_json_file_stores(&dir.path().join("passkeys.json"), large_caps());

        let ok = client_store
            .try_register_client(
                "c1".into(),
                RegisteredClient::new("secret1".into(), vec!["uri".into()]),
            )
            .await
            .unwrap();
        assert!(ok);

        let ok = client_store
            .try_register_client(
                "c2".into(),
                RegisteredClient::new("secret2".into(), vec!["uri".into()]),
            )
            .await
            .unwrap();
        assert!(!ok);

        assert!(client_store.get_client("c1").await.unwrap().is_some());
        assert!(client_store.get_client("c2").await.unwrap().is_none());
        assert_eq!(client_store.client_count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_try_register_client_respects_configurable_cap() {
        let dir = tempfile::tempdir().unwrap();
        let caps = StoreCaps {
            max_registered_clients: Some(2),
            ..large_caps()
        };
        let (_, client_store, _) = create_json_file_stores(&dir.path().join("passkeys.json"), caps);

        for i in 0..2 {
            let ok = client_store
                .try_register_client(
                    format!("c{i}"),
                    RegisteredClient::new(format!("s{i}"), vec!["uri".into()]),
                )
                .await
                .unwrap();
            assert!(ok, "registration {i} should succeed");
        }

        let ok = client_store
            .try_register_client(
                "c2".into(),
                RegisteredClient::new("s2".into(), vec!["uri".into()]),
            )
            .await
            .unwrap();
        assert!(!ok, "third registration should be rejected by cap");
        assert_eq!(client_store.client_count().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_try_register_client_does_not_clobber_existing_id() {
        // Defence-in-depth: even when under the cap, re-registering with an
        // existing id must return Ok(false) and leave the original client
        // intact. The real handler uses CSPRNG ids so collision is
        // astronomically unlikely, but custom stores or future callers may
        // pass known ids.
        let dir = tempfile::tempdir().unwrap();
        let caps = StoreCaps {
            max_registered_clients: Some(5),
            ..large_caps()
        };
        let (_, client_store, _) = create_json_file_stores(&dir.path().join("passkeys.json"), caps);

        let ok = client_store
            .try_register_client(
                "dup".into(),
                RegisteredClient::new("original-secret".into(), vec!["uri-1".into()]),
            )
            .await
            .unwrap();
        assert!(ok);

        let ok = client_store
            .try_register_client(
                "dup".into(),
                RegisteredClient::new("attacker-secret".into(), vec!["uri-evil".into()]),
            )
            .await
            .unwrap();
        assert!(!ok, "duplicate id should be refused");

        let existing = client_store.get_client("dup").await.unwrap().unwrap();
        assert_eq!(existing.client_secret, "original-secret");
        assert_eq!(existing.redirect_uris, vec!["uri-1"]);
        assert_eq!(client_store.client_count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_try_register_client_unlimited_when_cap_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let caps = StoreCaps {
            max_registered_clients: None,
            ..large_caps()
        };
        let (_, client_store, _) = create_json_file_stores(&dir.path().join("passkeys.json"), caps);

        for i in 0..5 {
            let ok = client_store
                .try_register_client(
                    format!("c{i}"),
                    RegisteredClient::new(format!("s{i}"), vec!["uri".into()]),
                )
                .await
                .unwrap();
            assert!(ok, "registration {i} should succeed with unlimited cap");
        }
        assert_eq!(client_store.client_count().await.unwrap(), 5);
    }

    #[tokio::test]
    async fn test_client_persistence_across_reload() {
        let dir = tempfile::tempdir().unwrap();
        let passkey_path = dir.path().join("passkeys.json");

        {
            let (_, client_store, _) = create_json_file_stores(&passkey_path, large_caps());
            client_store
                .register_client(
                    "c1".into(),
                    RegisteredClient::new("secret".into(), vec!["u".into()]),
                )
                .await
                .unwrap();
        }

        let (_, client_store, _) = create_json_file_stores(&passkey_path, large_caps());
        let client = client_store.get_client("c1").await.unwrap();
        assert!(client.is_some());
        assert_eq!(client.unwrap().redirect_uris, vec!["u"]);
    }

    // -- Regression: get_refresh_token is non-destructive --

    #[tokio::test]
    async fn test_get_refresh_token_does_not_consume() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _, _) =
            create_json_file_stores(&dir.path().join("passkeys.json"), large_caps());

        store
            .store_refresh_token("rt1".into(), RefreshTokenEntry::new("cid".into()))
            .await
            .unwrap();

        // get_refresh_token should not remove it
        let entry = store.get_refresh_token("rt1").await.unwrap();
        assert!(entry.is_some());

        // Should still be there
        let entry = store.get_refresh_token("rt1").await.unwrap();
        assert!(entry.is_some());

        // consume_refresh_token removes it
        let entry = store.consume_refresh_token("rt1").await.unwrap();
        assert!(entry.is_some());

        // Now it's gone
        assert!(store.get_refresh_token("rt1").await.unwrap().is_none());
    }

    // -- PasskeyStore tests --

    #[tokio::test]
    async fn test_passkey_store_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonFilePasskeyStore::new(dir.path().join("passkeys.json"));

        assert!(!store.has_passkeys().await.unwrap());
        assert!(store.list_passkeys().await.unwrap().is_empty());
    }
}
