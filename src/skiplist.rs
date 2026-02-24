//! Skip List 实现

use std::cmp::Ordering;
use std::fmt;

/// 跳表节点
#[derive(Clone)]
struct SkipNode<T> {
    value: T,
    forward: Vec<Option<usize>>, // 跳跃指针
}

impl<T> SkipNode<T> {
    fn new(value: T, level: usize) -> Self {
        Self {
            value,
            forward: vec![None; level + 1],
        }
    }
}

/// Skip List
pub struct SkipList<T> {
    header: usize,      // 头节点索引
    nodes: Vec<SkipNode<T>>,
    level: usize,       // 当前最大层数
    len: usize,         // 元素数量
    max_level: usize,   // 最大层数
}

impl<T: Ord + Clone + Default> SkipList<T> {
    pub fn new(max_level: usize) -> Self {
        let header_node = SkipNode::new(T::default(), max_level);
        Self {
            header: 0,
            nodes: vec![header_node],
            level: 0,
            len: 0,
            max_level,
        }
    }
    
    /// 随机层数
    fn random_level(&self) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut h = DefaultHasher::new();
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .hash(&mut h);
        
        let level = (h.finish() as usize) % (self.max_level + 1);
        level.min(self.level + 1)
    }
    
    /// 搜索
    pub fn search(&self, target: &T) -> Option<&T> {
        let mut current = self.header;
        
        for i in (0..=self.level).rev() {
            while let Some(next) = self.nodes[current].forward[i] {
                match self.nodes[next].value.cmp(target) {
                    Ordering::Less => current = next,
                    Ordering::Equal => return Some(&self.nodes[next].value),
                    Ordering::Greater => break,
                }
            }
        }
        
        // 检查下一层
        if let Some(next) = self.nodes[current].forward[0] {
            if self.nodes[next].value == *target {
                return Some(&self.nodes[next].value);
            }
        }
        
        None
    }
    
    /// 插入
    pub fn insert(&mut self, value: T) -> bool {
        // 搜索插入位置
        let mut updates: Vec<usize> = vec![0; self.max_level + 1];
        let mut current = self.header;
        
        for i in (0..=self.level).rev() {
            while let Some(next) = self.nodes[current].forward[i] {
                if self.nodes[next].value < value {
                    current = next;
                } else {
                    break;
                }
            }
            updates[i] = current;
        }
        
        // 检查是否已存在
        if let Some(next) = self.nodes[updates[0]].forward[0] {
            if self.nodes[next].value == value {
                return false; // 已存在
            }
        }
        
        // 新节点
        let new_level = self.random_level();
        let mut new_node = SkipNode::new(value, self.max_level);
        let new_idx = self.nodes.len();
        
        // 更新指针
        for i in 0..=new_level {
            if i < updates.len() {
                let update_idx = updates[i];
                new_node.forward[i] = self.nodes[update_idx].forward[i];
                self.nodes[update_idx].forward[i] = Some(new_idx);
            }
        }
        
        self.nodes.push(new_node);
        self.len += 1;
        
        // 更新层级
        if new_level > self.level {
            self.level = new_level;
        }
        
        true
    }
    
    /// 删除
    pub fn delete(&mut self, value: &T) -> bool {
        let mut updates: Vec<Option<usize>> = vec![None; self.max_level + 1];
        let mut current = self.header;
        
        for i in (0..=self.level).rev() {
            while let Some(next) = self.nodes[current].forward[i] {
                if self.nodes[next].value < *value {
                    current = next;
                } else {
                    break;
                }
            }
            updates[i] = Some(current);
        }
        
        // 找到要删除的节点
        if let Some(next) = self.nodes[updates[0].unwrap()].forward[0] {
            if self.nodes[next].value == *value {
                // 删除节点
                for i in 0..=self.level {
                    if let Some(idx) = updates[i] {
                        if let Some(fwd) = self.nodes[idx].forward[i] {
                            if fwd == next {
                                self.nodes[idx].forward[i] = self.nodes[next].forward[i];
                            }
                        }
                    }
                }
                self.len -= 1;
                return true;
            }
        }
        
        false
    }
    
    pub fn len(&self) -> usize {
        self.len
    }
    
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: fmt::Debug> fmt::Debug for SkipList<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SkipList {{ ")?;
        let mut current = self.header;
        let mut first = true;
        
        while let Some(next) = self.nodes[current].forward[0] {
            if !first {
                write!(f, " -> ")?;
            }
            write!(f, "{:?}", self.nodes[next].value)?;
            first = false;
            current = next;
        }
        
        write!(f, " }}, len: {}", self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_skip_list_new() {
        let list: SkipList<i32> = SkipList::new(16);
        assert!(list.is_empty());
    }
    
    #[test]
    fn test_skip_list_insert() {
        let mut list: SkipList<i32> = SkipList::new(16);
        
        list.insert(3);
        list.insert(1);
        list.insert(2);
        
        assert_eq!(list.len(), 3);
    }
    
    #[test]
    fn test_skip_list_search() {
        let mut list: SkipList<i32> = SkipList::new(16);
        
        list.insert(3);
        list.insert(1);
        list.insert(2);
        
        assert!(list.search(&2).is_some());
        assert!(list.search(&99).is_none());
    }
    
    #[test]
    fn test_skip_list_delete() {
        let mut list: SkipList<i32> = SkipList::new(16);
        
        list.insert(3);
        list.insert(1);
        list.insert(2);
        
        assert!(list.delete(&2));
        assert_eq!(list.len(), 2);
        assert!(list.search(&2).is_none());
    }
}
