//! 实用工具

/// 断言宏
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $msg:expr) => { if !($cond) { panic!($msg) } };
}

/// 调试断言（仅调试版本）
#[macro_export]
macro_rules! debug_assert_normal {
    ($($arg:tt)*) => { debug_assert!($($arg)*) };
}

/// 简单哈希
pub fn hash(data: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    data.hash(&mut h);
    h.finish()
}

/// 字节交换
pub fn byteswap(val: u32) -> u32 {
    val.reverse_bits()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hash() {
        let h = hash(b"test");
        assert!(h > 0);
    }
    
    #[test]
    fn test_byteswap() {
        assert_eq!(byteswap(0x12345678), 0x1E6A2C48);
    }
}
