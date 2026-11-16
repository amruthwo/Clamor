//! TLS for Clamor's connections, without certificate authorities.
//!
//! Every process makes a throwaway self-signed certificate at start ([`Identity`]). Peers are
//! recognised by the SHA-256 fingerprint of their certificate, which they learn over a channel
//! that is already authenticated (the coordinator hands out each stage's fingerprint), or, for
//! the control connection, not at all: there the join handshake signs the TLS session's
//! *exporter* value instead (see clamor-node's auth.rs), which ties the device keys to this one
//! TLS session, so a machine in the middle cannot relay the handshake.
//!
//! TLS 1.3 only, ring crypto.
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{verify_tls12_signature, verify_tls13_signature, CryptoProvider};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::{ClientConfig, DigitallySignedStruct, DistinguishedName, ServerConfig, SignatureScheme};
use sha2::{Digest, Sha256};
use std::io;
use std::sync::Arc;

/// SHA-256 of a certificate (DER).
pub type Fingerprint = [u8; 32];

pub fn fingerprint(cert_der: &[u8]) -> Fingerprint {
    Sha256::digest(cert_der).into()
}

pub fn fingerprint_hex(fp: &Fingerprint) -> String {
    fp.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn parse_fingerprint(s: &str) -> io::Result<Fingerprint> {
    let bad = || io::Error::new(io::ErrorKind::InvalidData, format!("bad certificate fingerprint {s:?}"));
    if s.len() != 64 {
        return Err(bad());
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| bad())?;
    }
    Ok(out)
}

/// This process's certificate and key. (Debug shows only the fingerprint.)
pub struct Identity {
    cert: CertificateDer<'static>,
    key: Vec<u8>,
    pub fingerprint: Fingerprint,
}

impl std::fmt::Debug for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Identity({})", self.fingerprint_hex())
    }
}

impl Identity {
    pub fn generate() -> io::Result<Arc<Identity>> {
        let ck = rcgen::generate_simple_self_signed(vec!["clamor".to_string()]).map_err(io::Error::other)?;
        let cert = ck.cert.der().clone();
        let fingerprint = fingerprint(&cert);
        Ok(Arc::new(Identity { cert, key: ck.signing_key.serialize_der(), fingerprint }))
    }

    pub fn fingerprint_hex(&self) -> String {
        fingerprint_hex(&self.fingerprint)
    }

    fn chain(&self) -> (Vec<CertificateDer<'static>>, PrivateKeyDer<'static>) {
        (vec![self.cert.clone()], PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(self.key.clone())))
    }
}

fn provider() -> Arc<CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

/// Accepts exactly the certificate with fingerprint `want`, or any certificate if `want` is
/// None (only where the session is authenticated another way: the control connection).
#[derive(Debug)]
struct Pinned {
    want: Option<Fingerprint>,
    provider: Arc<CryptoProvider>,
}

impl Pinned {
    fn check(&self, cert: &CertificateDer<'_>) -> Result<(), rustls::Error> {
        match self.want {
            Some(w) if fingerprint(cert) != w => Err(rustls::Error::General("certificate does not match the expected fingerprint".into())),
            _ => Ok(()),
        }
    }
}

impl ServerCertVerifier for Pinned {
    fn verify_server_cert(&self, end_entity: &CertificateDer<'_>, _: &[CertificateDer<'_>], _: &ServerName<'_>, _: &[u8], _: UnixTime) -> Result<ServerCertVerified, rustls::Error> {
        self.check(end_entity).map(|_| ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(message, cert, dss, &self.provider.signature_verification_algorithms)
    }
    fn verify_tls13_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(message, cert, dss, &self.provider.signature_verification_algorithms)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider.signature_verification_algorithms.supported_schemes()
    }
}

impl ClientCertVerifier for Pinned {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }
    fn verify_client_cert(&self, end_entity: &CertificateDer<'_>, _: &[CertificateDer<'_>], _: UnixTime) -> Result<ClientCertVerified, rustls::Error> {
        self.check(end_entity).map(|_| ClientCertVerified::assertion())
    }
    fn verify_tls12_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(message, cert, dss, &self.provider.signature_verification_algorithms)
    }
    fn verify_tls13_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(message, cert, dss, &self.provider.signature_verification_algorithms)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider.signature_verification_algorithms.supported_schemes()
    }
}

/// Server side. `client`: require a client certificate with this fingerprint (data links), or
/// None for no client certificate (the control connection).
pub fn server_config(id: &Identity, client: Option<Fingerprint>) -> io::Result<Arc<ServerConfig>> {
    let p = provider();
    let b = ServerConfig::builder_with_provider(p.clone()).with_protocol_versions(&[&rustls::version::TLS13]).map_err(io::Error::other)?;
    let b = match client {
        Some(fp) => b.with_client_cert_verifier(Arc::new(Pinned { want: Some(fp), provider: p })),
        None => b.with_no_client_auth(),
    };
    let (chain, key) = id.chain();
    Ok(Arc::new(b.with_single_cert(chain, key).map_err(io::Error::other)?))
}

/// Client side. `server`: the server certificate's fingerprint, or None to accept any (only
/// for the control connection, which authenticates through the exporter). `id`: present a
/// client certificate.
pub fn client_config(id: Option<&Identity>, server: Option<Fingerprint>) -> io::Result<Arc<ClientConfig>> {
    let p = provider();
    let b = ClientConfig::builder_with_provider(p.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(io::Error::other)?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(Pinned { want: server, provider: p }));
    Ok(Arc::new(match id {
        Some(id) => {
            let (chain, key) = id.chain();
            b.with_client_auth_cert(chain, key).map_err(io::Error::other)?
        }
        None => b.with_no_client_auth(),
    }))
}

pub fn server_name() -> ServerName<'static> {
    ServerName::try_from("clamor").expect("valid name")
}

/// Label for the exporter value the join handshake signs.
pub const EXPORTER_LABEL: &[u8] = b"EXPORTER-clamor-auth-v1";

/// The TLS session's exporter value (RFC 8446 7.5): equal on both ends of one session, and
/// different for every session, so signing it binds a handshake to this connection.
pub fn exporter_client(conn: &rustls::ClientConnection) -> io::Result<[u8; 32]> {
    conn.export_keying_material([0u8; 32], EXPORTER_LABEL, None).map_err(io::Error::other)
}

pub fn exporter_server(conn: &rustls::ServerConnection) -> io::Result<[u8; 32]> {
    conn.export_keying_material([0u8; 32], EXPORTER_LABEL, None).map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio_rustls::{TlsAcceptor, TlsConnector};

    async fn handshake(server_id: &Identity, client_id: &Identity, server_pin: Option<Fingerprint>, client_pin: Option<Fingerprint>) -> (io::Result<[u8; 32]>, io::Result<[u8; 32]>) {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        let acceptor = TlsAcceptor::from(server_config(server_id, client_pin).unwrap());
        let connector = TlsConnector::from(client_config(Some(client_id), server_pin).unwrap());
        let server = async move {
            let (s, _) = l.accept().await?;
            let mut t = acceptor.accept(s).await?;
            let x = exporter_server(t.get_ref().1)?;
            let mut b = [0u8; 2];
            t.read_exact(&mut b).await?;
            t.write_all(b"ok").await?;
            t.flush().await?;
            Ok(x)
        };
        let client = async move {
            let s = TcpStream::connect(addr).await?;
            let mut t = connector.connect(server_name(), s).await?;
            t.write_all(b"hi").await?;
            t.flush().await?;
            let mut b = [0u8; 2];
            t.read_exact(&mut b).await?;
            exporter_client(t.get_ref().1)
        };
        tokio::join!(server, client)
    }

    #[tokio::test]
    async fn pinned_certificates_connect_and_share_an_exporter() {
        let (a, b) = (Identity::generate().unwrap(), Identity::generate().unwrap());
        let (s, c) = handshake(&a, &b, Some(a.fingerprint), Some(b.fingerprint)).await;
        assert_eq!(s.unwrap(), c.unwrap(), "both ends see the same exporter");
        // Another session, another exporter.
        let (s2, _) = handshake(&a, &b, Some(a.fingerprint), Some(b.fingerprint)).await;
        assert_ne!(s2.unwrap(), handshake(&a, &b, None, None).await.0.unwrap());
    }

    #[tokio::test]
    async fn the_wrong_certificate_is_refused_both_ways() {
        let (a, b, x) = (Identity::generate().unwrap(), Identity::generate().unwrap(), Identity::generate().unwrap());
        let (_, c) = handshake(&a, &b, Some(x.fingerprint), None).await;
        assert!(c.is_err(), "client pinned another server");
        let (s, _) = handshake(&a, &b, None, Some(x.fingerprint)).await;
        assert!(s.is_err(), "server pinned another client");
    }

    #[test]
    fn fingerprints_round_trip() {
        let id = Identity::generate().unwrap();
        assert_eq!(parse_fingerprint(&id.fingerprint_hex()).unwrap(), id.fingerprint);
        assert!(parse_fingerprint("zz").is_err());
    }
}
