use std::collections::HashMap;
use std::io::{Read, Write};

use crate::api::{KnowhereError, MetricType, Result};
use crate::half::{Bf16, Fp16};
use crate::quantization::sq::ScalarQuantizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RefineType {
    DataView = 0,
    Uint8Quant = 1,
    Float16Quant = 2,
    Bfloat16Quant = 3,
    Sq4Quant = 4,
    Sq6Quant = 5,
}

impl RefineType {
    pub fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::DataView),
            1 => Ok(Self::Uint8Quant),
            2 => Ok(Self::Float16Quant),
            3 => Ok(Self::Bfloat16Quant),
            4 => Ok(Self::Sq4Quant),
            5 => Ok(Self::Sq6Quant),
            _ => Err(KnowhereError::InvalidArg(format!(
                "unknown refine type: {}",
                value
            ))),
        }
    }
}

#[derive(Clone, Debug)]
enum RefineStorage {
    DataView(Vec<f32>),
    Uint8Quant {
        codes: Vec<u8>,
        quantizer: ScalarQuantizer,
    },
    Float16Quant(Vec<u16>),
    Bfloat16Quant(Vec<u16>),
    Sq4Quant {
        codes: Vec<u8>,
        min: f32,
        scale: f32,
    },
    Sq6Quant {
        codes: Vec<u8>,
        min: f32,
        scale: f32,
    },
}

#[derive(Clone, Debug)]
pub struct RefineIndex {
    dim: usize,
    metric_type: MetricType,
    refine_type: RefineType,
    ids: Vec<i64>,
    id_to_offset: HashMap<i64, usize>,
    storage: RefineStorage,
}

pub fn pick_refine_index(
    data: &[f32],
    dim: usize,
    ids: &[i64],
    metric_type: MetricType,
    refine_type: Option<RefineType>,
) -> Result<Option<RefineIndex>> {
    match refine_type {
        Some(refine_type) => Ok(Some(RefineIndex::build(
            data,
            dim,
            ids,
            metric_type,
            refine_type,
        )?)),
        None => Ok(None),
    }
}

impl RefineIndex {
    pub fn build(
        data: &[f32],
        dim: usize,
        ids: &[i64],
        metric_type: MetricType,
        refine_type: RefineType,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(KnowhereError::InvalidArg(
                "refine dim must be > 0".to_string(),
            ));
        }
        if data.len() != ids.len() * dim {
            return Err(KnowhereError::InvalidArg(format!(
                "invalid refine input: data len {} does not match ids {} * dim {}",
                data.len(),
                ids.len(),
                dim
            )));
        }

        let storage = match refine_type {
            RefineType::DataView => RefineStorage::DataView(data.to_vec()),
            RefineType::Uint8Quant => {
                let mut quantizer = ScalarQuantizer::new(dim, 8);
                quantizer.train(data);
                RefineStorage::Uint8Quant {
                    codes: encode_sq8(&quantizer, data, dim),
                    quantizer,
                }
            }
            RefineType::Float16Quant => RefineStorage::Float16Quant(
                data.iter().map(|&v| Fp16::from_f32(v).to_bits()).collect(),
            ),
            RefineType::Bfloat16Quant => RefineStorage::Bfloat16Quant(
                data.iter().map(|&v| Bf16::from_f32(v).to_bits()).collect(),
            ),
            RefineType::Sq4Quant => {
                let min = data.iter().copied().fold(f32::INFINITY, f32::min);
                let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut scale = (max - min) / 15.0;
                if scale.abs() < 1e-6 {
                    scale = 1e-6;
                }
                RefineStorage::Sq4Quant {
                    codes: encode_sq4(data, dim, min, scale),
                    min,
                    scale,
                }
            }
            RefineType::Sq6Quant => {
                let min = data.iter().copied().fold(f32::INFINITY, f32::min);
                let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut scale = (max - min) / 63.0;
                if scale.abs() < 1e-6 {
                    scale = 1e-6;
                }
                RefineStorage::Sq6Quant {
                    codes: encode_sq6(data, dim, min, scale),
                    min,
                    scale,
                }
            }
        };

        let mut id_to_offset = HashMap::with_capacity(ids.len());
        for (offset, &id) in ids.iter().enumerate() {
            id_to_offset.insert(id, offset);
        }

        Ok(Self {
            dim,
            metric_type,
            refine_type,
            ids: ids.to_vec(),
            id_to_offset,
            storage,
        })
    }

    pub fn refine_type(&self) -> RefineType {
        self.refine_type
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn append(&mut self, data: &[f32], ids: &[i64]) -> Result<()> {
        if data.len() != ids.len() * self.dim {
            return Err(KnowhereError::InvalidArg(format!(
                "invalid refine append: data len {} does not match ids {} * dim {}",
                data.len(),
                ids.len(),
                self.dim
            )));
        }

        let base = self.ids.len();
        self.ids.extend_from_slice(ids);
        for (i, &id) in ids.iter().enumerate() {
            self.id_to_offset.insert(id, base + i);
        }

        match &mut self.storage {
            RefineStorage::DataView(values) => values.extend_from_slice(data),
            RefineStorage::Uint8Quant { codes, quantizer } => {
                codes.extend(encode_sq8(quantizer, data, self.dim));
            }
            RefineStorage::Float16Quant(values) => {
                values.extend(data.iter().map(|&v| Fp16::from_f32(v).to_bits()));
            }
            RefineStorage::Bfloat16Quant(values) => {
                values.extend(data.iter().map(|&v| Bf16::from_f32(v).to_bits()));
            }
            RefineStorage::Sq4Quant { codes, min, scale } => {
                codes.extend(encode_sq4(data, self.dim, *min, *scale));
            }
            RefineStorage::Sq6Quant { codes, min, scale } => {
                codes.extend(encode_sq6(data, self.dim, *min, *scale));
            }
        }

        Ok(())
    }

    pub fn refine_distance(&self, query: &[f32], id: i64) -> Option<f32> {
        if query.len() != self.dim {
            return None;
        }
        let offset = *self.id_to_offset.get(&id)?;
        Some(match &self.storage {
            RefineStorage::DataView(data) => {
                let base = offset * self.dim;
                compute_distance(self.metric_type, query, &data[base..base + self.dim])
            }
            RefineStorage::Uint8Quant { codes, quantizer } => {
                let base = offset * self.dim;
                compute_sq8_distance(
                    self.metric_type,
                    query,
                    &codes[base..base + self.dim],
                    quantizer,
                )
            }
            RefineStorage::Float16Quant(data) => {
                let base = offset * self.dim;
                compute_fp16_distance(self.metric_type, query, &data[base..base + self.dim])
            }
            RefineStorage::Bfloat16Quant(data) => {
                let base = offset * self.dim;
                compute_bf16_distance(self.metric_type, query, &data[base..base + self.dim])
            }
            RefineStorage::Sq4Quant { codes, min, scale } => {
                let bytes_per_vec = sq4_bytes_per_vec(self.dim);
                let base = offset * bytes_per_vec;
                let decoded =
                    decode_sq4(&codes[base..base + bytes_per_vec], self.dim, *min, *scale);
                compute_distance(self.metric_type, query, &decoded)
            }
            RefineStorage::Sq6Quant { codes, min, scale } => {
                let bytes_per_vec = sq6_bytes_per_vec(self.dim);
                let base = offset * bytes_per_vec;
                let decoded =
                    decode_sq6(&codes[base..base + bytes_per_vec], self.dim, *min, *scale);
                compute_distance(self.metric_type, query, &decoded)
            }
        })
    }

    pub fn refine_distances(&self, query: &[f32], ids: &[i64]) -> Vec<(i64, f32)> {
        ids.iter()
            .filter_map(|&id| self.refine_distance(query, id).map(|dist| (id, dist)))
            .collect()
    }

    pub fn rerank(
        &self,
        query: &[f32],
        candidates: &[(i64, f32)],
        top_k: usize,
    ) -> Vec<(i64, f32)> {
        let ids: Vec<i64> = candidates.iter().map(|(id, _)| *id).collect();
        let mut refined = self.refine_distances(query, &ids);
        sort_candidates(&mut refined, self.metric_type);
        refined.truncate(top_k);
        refined
    }

    pub fn rerank_batch(
        &self,
        queries: &[f32],
        candidate_batches: &[Vec<(i64, f32)>],
        top_k: usize,
    ) -> Result<Vec<Vec<(i64, f32)>>> {
        if queries.len() != candidate_batches.len() * self.dim {
            return Err(KnowhereError::InvalidArg(format!(
                "invalid batch rerank: queries len {} does not match batches {} * dim {}",
                queries.len(),
                candidate_batches.len(),
                self.dim
            )));
        }

        Ok(candidate_batches
            .iter()
            .enumerate()
            .map(|(i, candidates)| {
                let query = &queries[i * self.dim..(i + 1) * self.dim];
                self.rerank(query, candidates, top_k)
            })
            .collect())
    }

    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&[1u8])?;
        writer.write_all(&[self.refine_type as u8])?;
        writer.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        for &id in &self.ids {
            writer.write_all(&id.to_le_bytes())?;
        }

        match &self.storage {
            RefineStorage::DataView(values) => {
                for &value in values {
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
            RefineStorage::Uint8Quant { codes, quantizer } => {
                writer.write_all(&(quantizer.bit as u64).to_le_bytes())?;
                writer.write_all(&quantizer.min_val.to_le_bytes())?;
                writer.write_all(&quantizer.max_val.to_le_bytes())?;
                writer.write_all(&quantizer.scale.to_le_bytes())?;
                writer.write_all(&quantizer.offset.to_le_bytes())?;
                writer.write_all(codes)?;
            }
            RefineStorage::Float16Quant(values) | RefineStorage::Bfloat16Quant(values) => {
                for &value in values {
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
            RefineStorage::Sq4Quant { codes, min, scale } => {
                writer.write_all(&min.to_le_bytes())?;
                writer.write_all(&scale.to_le_bytes())?;
                writer.write_all(codes)?;
            }
            RefineStorage::Sq6Quant { codes, min, scale } => {
                writer.write_all(&min.to_le_bytes())?;
                writer.write_all(&scale.to_le_bytes())?;
                writer.write_all(codes)?;
            }
        }
        Ok(())
    }

    pub fn read_from<R: Read>(
        reader: &mut R,
        dim: usize,
        metric_type: MetricType,
    ) -> Result<Option<Self>> {
        let mut flag = [0u8; 1];
        reader.read_exact(&mut flag)?;
        if flag[0] == 0 {
            return Ok(None);
        }

        let mut refine_type_buf = [0u8; 1];
        reader.read_exact(&mut refine_type_buf)?;
        let refine_type = RefineType::from_u8(refine_type_buf[0])?;

        let len = read_u64(reader)? as usize;
        let mut ids = Vec::with_capacity(len);
        let mut id_to_offset = HashMap::with_capacity(len);
        for offset in 0..len {
            let id = read_i64(reader)?;
            ids.push(id);
            id_to_offset.insert(id, offset);
        }

        let storage = match refine_type {
            RefineType::DataView => {
                let mut values = vec![0.0f32; len * dim];
                for value in &mut values {
                    *value = read_f32(reader)?;
                }
                RefineStorage::DataView(values)
            }
            RefineType::Uint8Quant => {
                let bit = read_u64(reader)? as usize;
                let min_val = read_f32(reader)?;
                let max_val = read_f32(reader)?;
                let scale = read_f32(reader)?;
                let offset = read_f32(reader)?;
                let mut codes = vec![0u8; len * dim];
                reader.read_exact(&mut codes)?;
                let quantizer = ScalarQuantizer {
                    dim,
                    bit,
                    quantizer_type: crate::quantization::sq::QuantizerType::Uniform,
                    min_val,
                    max_val,
                    scale,
                    offset,
                };
                RefineStorage::Uint8Quant { codes, quantizer }
            }
            RefineType::Float16Quant => {
                let mut values = vec![0u16; len * dim];
                for value in &mut values {
                    *value = read_u16(reader)?;
                }
                RefineStorage::Float16Quant(values)
            }
            RefineType::Bfloat16Quant => {
                let mut values = vec![0u16; len * dim];
                for value in &mut values {
                    *value = read_u16(reader)?;
                }
                RefineStorage::Bfloat16Quant(values)
            }
            RefineType::Sq4Quant => {
                let min = read_f32(reader)?;
                let scale = read_f32(reader)?;
                let mut codes = vec![0u8; len * sq4_bytes_per_vec(dim)];
                reader.read_exact(&mut codes)?;
                RefineStorage::Sq4Quant { codes, min, scale }
            }
            RefineType::Sq6Quant => {
                let min = read_f32(reader)?;
                let scale = read_f32(reader)?;
                let mut codes = vec![0u8; len * sq6_bytes_per_vec(dim)];
                reader.read_exact(&mut codes)?;
                RefineStorage::Sq6Quant { codes, min, scale }
            }
        };

        Ok(Some(Self {
            dim,
            metric_type,
            refine_type,
            ids,
            id_to_offset,
            storage,
        }))
    }
}

fn sort_candidates(candidates: &mut [(i64, f32)], metric_type: MetricType) {
    match metric_type {
        MetricType::Ip | MetricType::Cosine => {
            candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        }
        MetricType::L2 | MetricType::Hamming => {
            candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        }
    }
}

fn encode_sq8(quantizer: &ScalarQuantizer, data: &[f32], dim: usize) -> Vec<u8> {
    data.chunks(dim)
        .flat_map(|vector| quantizer.encode(vector))
        .collect()
}

fn sq4_bytes_per_vec(dim: usize) -> usize {
    (dim + 1) / 2
}

fn encode_sq4(data: &[f32], dim: usize, min: f32, scale: f32) -> Vec<u8> {
    let n = data.len() / dim;
    let bytes_per_vec = sq4_bytes_per_vec(dim);
    let mut codes = vec![0u8; n * bytes_per_vec];
    let inv = 1.0 / scale.max(1e-6);
    for (vi, chunk) in data.chunks(dim).enumerate() {
        let base = vi * bytes_per_vec;
        for (i, &v) in chunk.iter().enumerate() {
            let code = ((v - min) * inv).round().clamp(0.0, 15.0) as u8;
            if i % 2 == 0 {
                codes[base + i / 2] = code;
            } else {
                codes[base + i / 2] |= code << 4;
            }
        }
    }
    codes
}

fn decode_sq4(codes: &[u8], dim: usize, min: f32, scale: f32) -> Vec<f32> {
    let bytes_per_vec = sq4_bytes_per_vec(dim);
    let n = codes.len() / bytes_per_vec;
    let mut out = Vec::with_capacity(n * dim);
    for vi in 0..n {
        let base = vi * bytes_per_vec;
        for i in 0..dim {
            let byte = codes[base + i / 2];
            let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out.push(nibble as f32 * scale + min);
        }
    }
    out
}

fn sq6_bytes_per_vec(dim: usize) -> usize {
    (dim * 6 + 7) / 8
}

fn encode_sq6(data: &[f32], dim: usize, min: f32, scale: f32) -> Vec<u8> {
    let n = data.len() / dim;
    let bytes_per_vec = sq6_bytes_per_vec(dim);
    let inv = 1.0 / scale.max(1e-6);
    let mut codes = vec![0u8; n * bytes_per_vec];
    for (vi, chunk) in data.chunks(dim).enumerate() {
        let base = vi * bytes_per_vec;
        let mut bitbuf = 0u32;
        let mut bitcnt = 0usize;
        let mut byte_idx = 0usize;
        for &v in chunk {
            let q = ((v - min) * inv).round().clamp(0.0, 63.0) as u32;
            bitbuf |= q << bitcnt;
            bitcnt += 6;
            while bitcnt >= 8 {
                codes[base + byte_idx] = (bitbuf & 0xFF) as u8;
                bitbuf >>= 8;
                bitcnt -= 8;
                byte_idx += 1;
            }
        }
        if bitcnt > 0 && byte_idx < bytes_per_vec {
            codes[base + byte_idx] = (bitbuf & 0xFF) as u8;
        }
    }
    codes
}

fn decode_sq6(codes: &[u8], dim: usize, min: f32, scale: f32) -> Vec<f32> {
    let bytes_per_vec = sq6_bytes_per_vec(dim);
    let n = codes.len() / bytes_per_vec;
    let mut out = Vec::with_capacity(n * dim);
    for vi in 0..n {
        let base = vi * bytes_per_vec;
        let mut bitbuf = 0u32;
        let mut bitcnt = 0usize;
        let mut byte_idx = 0usize;
        for _ in 0..dim {
            while bitcnt < 6 && byte_idx < bytes_per_vec {
                bitbuf |= (codes[base + byte_idx] as u32) << bitcnt;
                bitcnt += 8;
                byte_idx += 1;
            }
            let q = (bitbuf & 0x3F) as f32;
            out.push(q * scale + min);
            bitbuf >>= 6;
            bitcnt -= 6;
        }
    }
    out
}

fn compute_distance(metric_type: MetricType, query: &[f32], vector: &[f32]) -> f32 {
    match metric_type {
        MetricType::L2 | MetricType::Hamming => query
            .iter()
            .zip(vector.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum(),
        MetricType::Ip | MetricType::Cosine => -query
            .iter()
            .zip(vector.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>(),
    }
}

fn compute_sq8_distance(
    metric_type: MetricType,
    query: &[f32],
    code: &[u8],
    quantizer: &ScalarQuantizer,
) -> f32 {
    match metric_type {
        MetricType::L2 | MetricType::Hamming => query
            .iter()
            .zip(code.iter())
            .map(|(q, &c)| {
                let decoded = c as f32 / quantizer.scale + quantizer.offset;
                let diff = q - decoded;
                diff * diff
            })
            .sum(),
        MetricType::Ip | MetricType::Cosine => -query
            .iter()
            .zip(code.iter())
            .map(|(q, &c)| q * (c as f32 / quantizer.scale + quantizer.offset))
            .sum::<f32>(),
    }
}

fn compute_fp16_distance(metric_type: MetricType, query: &[f32], vector: &[u16]) -> f32 {
    match metric_type {
        MetricType::L2 | MetricType::Hamming => query
            .iter()
            .zip(vector.iter())
            .map(|(q, &v)| {
                let decoded = Fp16::from_bits(v).to_f32();
                let diff = q - decoded;
                diff * diff
            })
            .sum(),
        MetricType::Ip | MetricType::Cosine => -query
            .iter()
            .zip(vector.iter())
            .map(|(q, &v)| q * Fp16::from_bits(v).to_f32())
            .sum::<f32>(),
    }
}

fn compute_bf16_distance(metric_type: MetricType, query: &[f32], vector: &[u16]) -> f32 {
    match metric_type {
        MetricType::L2 | MetricType::Hamming => query
            .iter()
            .zip(vector.iter())
            .map(|(q, &v)| {
                let decoded = Bf16::from_bits(v).to_f32();
                let diff = q - decoded;
                diff * diff
            })
            .sum(),
        MetricType::Ip | MetricType::Cosine => -query
            .iter()
            .zip(vector.iter())
            .map(|(q, &v)| q * Bf16::from_bits(v).to_f32())
            .sum::<f32>(),
    }
}

fn read_u16<R: Read>(reader: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(reader: &mut R) -> Result<i64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> (Vec<f32>, Vec<i64>) {
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.08, 0.08, 0.08, 0.08, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0,
        ];
        let ids = vec![10, 11, 12, 13];
        (data, ids)
    }

    #[test]
    fn test_refine_rerank_dataview() {
        let (data, ids) = sample_data();
        let refine =
            RefineIndex::build(&data, 4, &ids, MetricType::L2, RefineType::DataView).unwrap();
        let query = vec![0.02; 4];
        let candidates = vec![(12, 10.0), (11, 11.0), (10, 12.0)];
        let reranked = refine.rerank(&query, &candidates, 2);
        assert_eq!(reranked[0].0, 10);
        assert_eq!(reranked[1].0, 11);
    }

    #[test]
    fn test_refine_rerank_float16() {
        let (data, ids) = sample_data();
        let refine =
            RefineIndex::build(&data, 4, &ids, MetricType::L2, RefineType::Float16Quant).unwrap();
        let query = vec![0.02; 4];
        let candidates = vec![(12, 10.0), (11, 11.0), (10, 12.0)];
        let reranked = refine.rerank(&query, &candidates, 2);
        assert_eq!(reranked[0].0, 10);
        assert_eq!(reranked[1].0, 11);
    }

    #[test]
    fn test_refine_rerank_batch() {
        let (data, ids) = sample_data();
        let refine =
            RefineIndex::build(&data, 4, &ids, MetricType::L2, RefineType::DataView).unwrap();
        let queries = vec![0.02; 4 * 2];
        let batches = vec![
            vec![(12, 10.0), (11, 11.0), (10, 12.0)],
            vec![(13, 10.0), (12, 11.0), (11, 12.0)],
        ];
        let reranked = refine.rerank_batch(&queries, &batches, 1).unwrap();
        assert_eq!(reranked.len(), 2);
        assert_eq!(reranked[0][0].0, 10);
        assert_eq!(reranked[1][0].0, 11);
    }

    #[test]
    fn test_sq4_refine_roundtrip() {
        let dim = 7usize;
        let data: Vec<f32> = (0..(16 * dim))
            .map(|i| (i % 31) as f32 * 0.3 - 2.0)
            .collect();
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        let scale = (range / 15.0).max(1e-6);
        let codes = encode_sq4(&data, dim, min, scale);
        let decoded = decode_sq4(&codes, dim, min, scale);
        let max_err = data
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("sq4_roundtrip max_err={:.6} range={:.6}", max_err, range);
        assert!(max_err <= range / 15.0 + 1e-5);
    }

    #[test]
    fn test_sq6_refine_roundtrip() {
        let dim = 11usize;
        let data: Vec<f32> = (0..(16 * dim))
            .map(|i| (i % 47) as f32 * 0.2 - 3.0)
            .collect();
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        let scale = (range / 63.0).max(1e-6);
        let codes = encode_sq6(&data, dim, min, scale);
        let decoded = decode_sq6(&codes, dim, min, scale);
        let max_err = data
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("sq6_roundtrip max_err={:.6} range={:.6}", max_err, range);
        assert!(max_err <= range / 63.0 + 1e-5);
    }
}
