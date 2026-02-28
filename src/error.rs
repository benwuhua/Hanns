//! 错误处理

/// 错误码
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorCode(pub u32);

impl ErrorCode {
    pub const SUCCESS: ErrorCode = ErrorCode(0);
    pub const NOT_FOUND: ErrorCode = ErrorCode(1);
    pub const INVALID_ARG: ErrorCode = ErrorCode(2);
    pub const OUT_OF_MEMORY: ErrorCode = ErrorCode(3);
    pub const IO_ERROR: ErrorCode = ErrorCode(4);
    pub const NOT_IMPLEMENTED: ErrorCode = ErrorCode(5);
    pub const INTERRUPTED: ErrorCode = ErrorCode(6);
}

/// 错误类型
#[derive(Debug)]
pub enum KnowhereError {
    /// Standard error with code and message
    Standard { code: ErrorCode, msg: String },
    /// Operation was interrupted
    Interrupted,
    /// Operation was interrupted with a custom message
    InterruptedWithMessage(String),
    /// Internal error (e.g., lock poisoning)
    InternalError(String),
    /// Invalid argument
    InvalidArg(String),
    /// Index not trained
    IndexNotTrained(String),
}

impl KnowhereError {
    pub fn new(code: ErrorCode, msg: &str) -> Self {
        Self::Standard { code, msg: msg.into() }
    }
    
    pub fn interrupted() -> Self {
        Self::Interrupted
    }
    
    pub fn interrupted_with_message(msg: String) -> Self {
        Self::InterruptedWithMessage(msg)
    }
    
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Standard { code, .. } => *code,
            Self::Interrupted | Self::InterruptedWithMessage(_) => ErrorCode::INTERRUPTED,
            Self::InternalError(_) => ErrorCode::IO_ERROR,
            Self::InvalidArg(_) => ErrorCode::INVALID_ARG,
            Self::IndexNotTrained(_) => ErrorCode::NOT_IMPLEMENTED,
        }
    }
    
    pub fn msg(&self) -> &str {
        match self {
            Self::Standard { msg, .. } => msg,
            Self::Interrupted => "Operation interrupted",
            Self::InterruptedWithMessage(msg) => msg,
            Self::InternalError(msg) => msg,
            Self::InvalidArg(msg) => msg,
            Self::IndexNotTrained(msg) => msg,
        }
    }
}

impl std::fmt::Display for KnowhereError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error {}: {}", self.code().0, self.msg())
    }
}

impl std::error::Error for KnowhereError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error() {
        let e = KnowhereError::new(ErrorCode::NOT_FOUND, "vector not found");
        assert_eq!(e.code(), ErrorCode::NOT_FOUND);
    }
}
