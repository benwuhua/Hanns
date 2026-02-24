//! 序列化模块 - 索引持久化

use std::io::{Read, Write, Result};

/// 可序列化索引 trait
pub trait Serializable {
    fn serialize(&self, writer: &mut dyn Write) -> Result<()>;
    fn deserialize(&mut self, reader: &mut dyn Read) -> Result<()>;
}

/// 通用序列化工具
pub struct Serializer;

impl Serializer {
    /// 写入向量
    pub fn write_vector(writer: &mut dyn Write, v: &[f32]) -> Result<()> {
        let len = v.len() as u64;
        writer.write_all(&len.to_le_bytes())?;
        for &x in v {
            writer.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    }
    
    /// 读取向量
    pub fn read_vector(reader: &mut dyn Read) -> Result<Vec<f32>> {
        let mut len_buf = [0u8; 8];
        reader.read_exact(&mut len_buf)?;
        let len = u64::from_le_bytes(len_buf) as usize;
        
        let mut v = vec![0.0f32; len];
        for x in &mut v {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            *x = f32::from_le_bytes(buf);
        }
        Ok(v)
    }
    
    /// 写入整数向量
    pub fn write_usize_vec(writer: &mut dyn Write, v: &[usize]) -> Result<()> {
        let len = v.len() as u64;
        writer.write_all(&len.to_le_bytes())?;
        for &x in v {
            writer.write_all(&(x as u64).to_le_bytes())?;
        }
        Ok(())
    }
    
    /// 读取整数向量
    pub fn read_usize_vec(reader: &mut dyn Read) -> Result<Vec<usize>> {
        let mut len_buf = [0u8; 8];
        reader.read_exact(&mut len_buf)?;
        let len = u64::from_le_bytes(len_buf) as usize;
        
        let mut v = vec![0usize; len];
        for x in &mut v {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            *x = u64::from_le_bytes(buf) as usize;
        }
        Ok(v)
    }
}

/// 二进制格式版本
pub const FORMAT_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    
    #[test]
    fn test_serialize_vector() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut buf = Vec::new();
        
        Serializer::write_vector(&mut buf, &v).unwrap();
        let mut cursor = Cursor::new(buf);
        let read_v = Serializer::read_vector(&mut cursor).unwrap();
        
        assert_eq!(v, read_v);
    }
    
    #[test]
    fn test_serialize_usize() {
        let v = vec![1, 2, 3, 4, 5];
        let mut buf = Vec::new();
        
        Serializer::write_usize_vec(&mut buf, &v).unwrap();
        let mut cursor = Cursor::new(buf);
        let read_v = Serializer::read_usize_vec(&mut cursor).unwrap();
        
        assert_eq!(v, read_v);
    }
}
