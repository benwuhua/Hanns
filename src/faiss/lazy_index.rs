use std::sync::{Arc, Mutex};

/// 延迟加载状态
pub enum LazyState<T> {
    /// 未加载，持有加载函数
    Unloaded(Box<dyn FnOnce() -> anyhow::Result<T> + Send>),
    /// 已加载
    Loaded(Arc<T>),
    /// 加载失败（记录错误信息）
    Failed(String),
}

/// Lazy-loading wrapper for any index-like type.
/// The inner object is only constructed on first access.
pub struct LazyIndex<T> {
    state: Mutex<Option<LazyState<T>>>,
    /// 元数据（dim、n_total 等）在 load 前就可用
    pub dim: usize,
    pub description: String,
}

impl<T: Send + Sync + 'static> LazyIndex<T> {
    /// 创建一个延迟加载 index，loader 在第一次访问时调用
    pub fn new(
        dim: usize,
        description: impl Into<String>,
        loader: impl FnOnce() -> anyhow::Result<T> + Send + 'static,
    ) -> Self {
        Self {
            state: Mutex::new(Some(LazyState::Unloaded(Box::new(loader)))),
            dim,
            description: description.into(),
        }
    }

    /// 立即加载（预热用），幂等
    pub fn load_now(&self) -> anyhow::Result<()> {
        self.ensure_loaded().map(|_| ())
    }

    /// 获取内部 index 引用（Arc），触发加载（如未加载）
    pub fn get(&self) -> anyhow::Result<Arc<T>> {
        self.ensure_loaded()
    }

    fn ensure_loaded(&self) -> anyhow::Result<Arc<T>> {
        let mut guard = self.state.lock().expect("lazy state mutex poisoned");
        match guard.as_ref() {
            Some(LazyState::Loaded(v)) => return Ok(v.clone()),
            Some(LazyState::Failed(e)) => {
                return Err(anyhow::anyhow!("index load failed: {}", e));
            }
            Some(LazyState::Unloaded(_)) => {}
            None => return Err(anyhow::anyhow!("lazy state corrupted")),
        }

        // 取出 loader
        let state = guard.take().expect("state checked above");
        let loader = match state {
            LazyState::Unloaded(f) => f,
            _ => unreachable!("state checked above"),
        };
        drop(guard);

        // 在锁外执行 loader（防止长时间持锁）
        match loader() {
            Ok(index) => {
                let loaded = Arc::new(index);
                let mut guard = self.state.lock().expect("lazy state mutex poisoned");
                *guard = Some(LazyState::Loaded(loaded.clone()));
                Ok(loaded)
            }
            Err(e) => {
                let msg = e.to_string();
                let mut guard = self.state.lock().expect("lazy state mutex poisoned");
                *guard = Some(LazyState::Failed(msg.clone()));
                Err(anyhow::anyhow!("index load failed: {}", msg))
            }
        }
    }

    pub fn is_loaded(&self) -> bool {
        matches!(
            self.state
                .lock()
                .expect("lazy state mutex poisoned")
                .as_ref(),
            Some(LazyState::Loaded(_))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::LazyIndex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_lazy_index() {
        let load_count = Arc::new(AtomicUsize::new(0));
        let lc = load_count.clone();

        let lazy = LazyIndex::new(128, "test_index", move || {
            lc.fetch_add(1, Ordering::SeqCst);
            Ok(vec![1u8, 2, 3])
        });

        assert!(!lazy.is_loaded(), "should not be loaded yet");
        assert_eq!(
            load_count.load(Ordering::SeqCst),
            0,
            "loader not called yet"
        );

        let data = lazy.get().expect("first get should load");
        assert_eq!(*data, vec![1u8, 2, 3]);
        assert!(lazy.is_loaded());
        assert_eq!(
            load_count.load(Ordering::SeqCst),
            1,
            "loader called exactly once"
        );

        // 再次 get 不重复加载
        let _ = lazy.get().expect("second get should reuse loaded value");
        assert_eq!(
            load_count.load(Ordering::SeqCst),
            1,
            "loader called exactly once (idempotent)"
        );

        println!(
            "test_lazy_index: OK, loader called {} time(s)",
            load_count.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn test_lazy_index_failure() {
        let lazy: LazyIndex<Vec<u8>> = LazyIndex::new(128, "bad_index", || {
            Err(anyhow::anyhow!("simulated load failure"))
        });

        let result = lazy.get();
        assert!(result.is_err());
        let err_msg = result.err().expect("error expected").to_string();
        assert!(
            err_msg.contains("simulated load failure"),
            "err: {}",
            err_msg
        );
        println!("test_lazy_index_failure: OK, error={}", err_msg);
    }
}
