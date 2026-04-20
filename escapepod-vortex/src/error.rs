use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("escapepod: {0}")]
    Pod5(#[from] escapepod::Error),
    #[error("vortex: {0}")]
    Vortex(#[from] vortex_error::VortexError),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
