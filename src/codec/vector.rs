//! Vector serialization codec

use std::io::{Read, Write, Seek};

use crate::api::{KnowhereError, Result};

/// Vector codec for serialization
pub struct VectorCodec;

impl VectorCodec {
    /// Write vectors in batch
    /// Format:
    /// - num_vectors: u64
    /// - dim: u64
    /// - vectors: [f32]
    pub fn write_vectors<W: Write + Seek>(
        vectors: &[f32],
        num: usize,
        dim: usize,
        writer: &mut W,
    ) -> Result<()> {
        if vectors.len() != num * dim {
            return Err(KnowhereError::Codec(format!(
                "vector size mismatch: expected {}*{}={}, got {}",
                num, dim, num * dim,
                vectors.len()
            )));
        }
        
        // Magic number
        writer.write_all(b"VECS")?;
        
        // Version
        writer.write_all(&1u32.to_le_bytes())?;
        
        // Num vectors
        writer.write_all(&(num as u64).to_le_bytes())?;
        
        // Dimension
        writer.write_all(&(dim as u64).to_le_bytes())?;
        
        // Vectors (raw f32 bytes)
        let bytes: Vec<u8> = vectors.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        writer.write_all(&bytes)?;
        
        Ok(())
    }

    /// Read vectors
    pub fn read_vectors<R: Read + Seek>(reader: &mut R) -> Result<(usize, usize, Vec<f32>)> {
        // Magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"VECS" {
            return Err(KnowhereError::Codec("invalid vector magic".to_string()));
        }
        
        // Version
        let mut version = [0u8; 4];
        reader.read_exact(&mut version)?;
        
        // Num vectors
        let mut num_bytes = [0u8; 8];
        reader.read_exact(&mut num_bytes)?;
        let num = u64::from_le_bytes(num_bytes) as usize;
        
        // Dimension
        let mut dim_bytes = [0u8; 8];
        reader.read_exact(&mut dim_bytes)?;
        let dim = u64::from_le_bytes(dim_bytes) as usize;
        
        // Read vectors
        let total = num * dim;
        let mut buffer = vec![0u8; total * 4];
        reader.read_exact(&mut buffer)?;
        
        let vectors: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        Ok((num, dim, vectors))
    }

    /// Write ID mapping (vector ID -> internal ID)
    pub fn write_ids<W: Write>(ids: &[i64], writer: &mut W) -> Result<()> {
        writer.write_all(&(ids.len() as u64).to_le_bytes())?;
        for &id in ids {
            writer.write_all(&id.to_le_bytes())?;
        }
        Ok(())
    }

    /// Read ID mapping
    pub fn read_ids<R: Read>(reader: &mut R) -> Result<Vec<i64>> {
        let mut num_bytes = [0u8; 8];
        reader.read_exact(&mut num_bytes)?;
        let num = u64::from_le_bytes(num_bytes) as usize;
        
        let mut ids = Vec::with_capacity(num);
        for _ in 0..num {
            let mut id_bytes = [0u8; 8];
            reader.read_exact(&mut id_bytes)?;
            ids.push(i64::from_le_bytes(id_bytes));
        }
        
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use super::*;

    #[test]
    fn test_vector_codec() {
        let vectors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let num = 2;
        let dim = 4;
        
        let mut buffer = Cursor::new(Vec::new());
        VectorCodec::write_vectors(&vectors, num, dim, &mut buffer).unwrap();
        
        buffer.set_position(0);
        let (read_num, read_dim, read_vectors) = VectorCodec::read_vectors(&mut buffer).unwrap();
        
        assert_eq!(num, read_num);
        assert_eq!(dim, read_dim);
        assert_eq!(vectors, read_vectors);
    }
}
