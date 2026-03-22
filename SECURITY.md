# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in mcp-oauth, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email: **security@pecheriere.com** (or open a [private security advisory](https://github.com/pecheriere/mcp-oauth/security/advisories/new) on GitHub).

You should receive an acknowledgment within 48 hours. We will work with you to understand the issue and coordinate a fix before any public disclosure.

## Scope

This crate handles security-sensitive operations including:

- OAuth 2.1 authorization code flow with PKCE
- Bearer token generation, validation, and refresh
- WebAuthn / passkey registration and authentication
- Dynamic client registration (RFC 7591)
- Per-IP rate limiting

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
