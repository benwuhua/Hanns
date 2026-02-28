//! Index serialization codec

use std::io::{Read, Write, Seek};

use crate::api::{IndexConfig, IndexType, KnowhereError, MetricType, Result};

/// Index codec for serialization
pub struct IndexCodec;

impl IndexCodec {
    /// Serialize index config to binary
    pub fn write_config<W: Write + Seek>(config: &IndexConfig, writer: &mut W) -> Result<()> {
        // Magic number
        writer.write_all(b"KWIX")?; // KnowHere Index
        
        // Version
        let version: u32 = 1;
        writer.write_all(&version.to_le_bytes())?;
        
        // Index type
        let idx_type: u32 = match config.index_type {
            IndexType::Flat => 0,
            IndexType::IvfFlat => 1,
            IndexType::IvfPq => 2,
            IndexType::Hnsw => 3,
            IndexType::DiskAnn => 4,
            IndexType::Annoy => 5,
            IndexType::IvfSq8 => 6,
            IndexType::HnswPrq => 7,
            IndexType::IvfRabitq => 8,
            IndexType::HnswPq => 16,
            IndexType::IvfFlatCc => 9,
            IndexType::IvfSqCc => 10,
            IndexType::SparseInverted => 11,
            IndexType::SparseInvertedCc => 17,
            IndexType::SparseWand => 18,
            IndexType::SparseWandCc => 19,
            IndexType::BinaryHnsw => 12,
            IndexType::BinFlat => 13,
            IndexType::BinIvfFlat => 20,
            IndexType::HnswSq => 14,
            IndexType::Aisaq => 15,
            IndexType::HnswPq => 16,
            IndexType::MinHashLsh => 21,
        };
        writer.write_all(&idx_type.to_le_bytes())?;
        
        // Metric type
        let metric: u32 = match config.metric_type {
            MetricType::L2 => 0,
            MetricType::Ip => 1,
            MetricType::Cosine => 2,
            MetricType::Hamming => 3,
        };
        writer.write_all(&metric.to_le_bytes())?;
        
        // Dimension
        let dim: u32 = config.dim as u32;
        writer.write_all(&dim.to_le_bytes())?;
        
        Ok(())
    }

    /// Deserialize index config from binary
    pub fn read_config<R: Read + Seek>(reader: &mut R) -> Result<IndexConfig> {
        // Magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"KWIX" {
            return Err(KnowhereError::Codec("invalid magic number".to_string()));
        }
        
        // Version
        let mut version = [0u8; 4];
        reader.read_exact(&mut version)?;
        let version = u32::from_le_bytes(version);
        if version != 1 {
            return Err(KnowhereError::Codec(format!("unsupported version: {}", version)));
        }
        
        // Index type
        let mut idx_type_bytes = [0u8; 4];
        reader.read_exact(&mut idx_type_bytes)?;
        let idx_type = u32::from_le_bytes(idx_type_bytes);
        let index_type = match idx_type {
            0 => IndexType::Flat,
            1 => IndexType::IvfFlat,
            2 => IndexType::IvfPq,
            3 => IndexType::Hnsw,
            4 => IndexType::DiskAnn,
            5 => IndexType::Annoy,
            6 => IndexType::IvfSq8,
            7 => IndexType::HnswPrq,
            8 => IndexType::IvfRabitq,
            16 => IndexType::HnswPq,
            9 => IndexType::IvfFlatCc,
            10 => IndexType::IvfSqCc,
            11 => IndexType::SparseInverted,
            17 => IndexType::SparseInvertedCc,
            18 => IndexType::SparseWand,
            19 => IndexType::SparseWandCc,
            12 => IndexType::BinaryHnsw,
            13 => IndexType::BinFlat,
            20 => IndexType::BinIvfFlat,
            14 => IndexType::HnswSq,
            15 => IndexType::Aisaq,
            16 => IndexType::HnswPq,
            21 => IndexType::MinHashLsh,
            _ => return Err(KnowhereError::Codec(format!("unknown index type: {}", idx_type))),
        };
        
        // Metric type
        let mut metric_bytes = [0u8; 4];
        reader.read_exact(&mut metric_bytes)?;
        let metric = u32::from_le_bytes(metric_bytes);
        let metric_type = match metric {
            0 => MetricType::L2,
            1 => MetricType::Ip,
            2 => MetricType::Cosine,
            _ => return Err(KnowhereError::Codec(format!("unknown metric type: {}", metric))),
        };
        
        // Dimension
        let mut dim_bytes = [0u8; 4];
        reader.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        
        Ok(IndexConfig::new(index_type, metric_type, dim))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use super::*;

    #[test]
    fn test_codec_roundtrip() {
        let config = IndexConfig::new(IndexType::Hnsw, MetricType::L2, 128);
        
        let mut buffer = Cursor::new(Vec::new());
        IndexCodec::write_config(&config, &mut buffer).unwrap();
        
        buffer.set_position(0);
        let loaded = IndexCodec::read_config(&mut buffer).unwrap();
        
        assert_eq!(config.index_type, loaded.index_type);
        assert_eq!(config.metric_type, loaded.metric_type);
        assert_eq!(config.dim, loaded.dim);
    }
}
