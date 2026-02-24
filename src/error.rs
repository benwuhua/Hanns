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
}

/// 错误类型
#[derive(Debug)]
pub struct KnowhereError {
    code: ErrorCode,
    msg: String,
}

impl KnowhereError {
    pub fn new(code: ErrorCode, msg: &str) -> Self {
        Self { code, msg: msg.into() }
    }
    
    pub fn code(&self) -> ErrorCode { self.code }
    pub fn msg(&self) -> &str { &self.msg }
}

impl std::fmt::Display for KnowhereError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error {}: {}", self.code.0, self.msg)
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
