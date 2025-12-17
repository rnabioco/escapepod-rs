//! UUID parsing utilities.

use crate::{Error, Result, Uuid};

/// Parse a UUID from various formats.
///
/// Supports:
/// - Standard format with dashes: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`
/// - Compact format without dashes: `a1b2c3d4e5f67890abcdef1234567890`
///
/// # Examples
///
/// ```
/// use escapepod::parse_uuid_flexible;
///
/// // Standard format
/// let uuid = parse_uuid_flexible("a1b2c3d4-e5f6-7890-abcd-ef1234567890").unwrap();
///
/// // Compact format (32 hex characters)
/// let uuid2 = parse_uuid_flexible("a1b2c3d4e5f67890abcdef1234567890").unwrap();
///
/// assert_eq!(uuid, uuid2);
/// ```
pub fn parse_uuid_flexible(s: &str) -> Result<Uuid> {
    // Try standard format first
    if let Ok(uuid) = Uuid::parse_str(s) {
        return Ok(uuid);
    }

    // Try without dashes (32 hex characters)
    if s.len() == 32 && s.chars().all(|c| c.is_ascii_hexdigit()) {
        let with_dashes = format!(
            "{}-{}-{}-{}-{}",
            &s[0..8],
            &s[8..12],
            &s[12..16],
            &s[16..20],
            &s[20..32]
        );
        return Uuid::parse_str(&with_dashes).map_err(|e| Error::InvalidUuid(format!("{}", e)));
    }

    Err(Error::InvalidUuid(format!("Invalid format: '{}'", s)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_uuid_standard() {
        let uuid = parse_uuid_flexible("a1b2c3d4-e5f6-7890-abcd-ef1234567890").unwrap();
        assert_eq!(uuid.to_string(), "a1b2c3d4-e5f6-7890-abcd-ef1234567890");
    }

    #[test]
    fn test_parse_uuid_compact() {
        let uuid = parse_uuid_flexible("a1b2c3d4e5f67890abcdef1234567890").unwrap();
        assert_eq!(uuid.to_string(), "a1b2c3d4-e5f6-7890-abcd-ef1234567890");
    }

    #[test]
    fn test_parse_uuid_invalid() {
        assert!(parse_uuid_flexible("not-a-uuid").is_err());
        assert!(parse_uuid_flexible("").is_err());
        assert!(parse_uuid_flexible("12345").is_err());
    }
}
