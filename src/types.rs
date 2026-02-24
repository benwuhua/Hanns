//! 类型别名

use crate::error::KnowhereError;
use crate::version::Config;

pub type Result<T> = std::result::Result<T, KnowhereError>;

pub type IndexConfig = Config;
