//! MinHash-LSH Index Implementation
//! 
//! MinHash LSH (Locality Sensitive Hashing) for Jaccard similarity search.
//! This implementation is based on the C++ knowhere MinHash-LSH index.

use std::fs::File;
use std::io::{Read, Write, Seek, SeekFrom};
use std::cmp::{min, max};
use crate::error::KnowhereError;
use crate::bitset::BitsetView;
use crate::interrupt::Interrupt;

type Result<T> = std::result::Result<T, KnowhereError>;

/// Key type for hash values
pub type KeyType = u64;
/// Value type for vector IDs
pub type ValueType = i64;

/// Key-Value pair for hash table
#[derive(Debug, Clone, Copy)]
pub struct KVPair {
    pub key: KeyType,
    pub value: ValueType,
}

/// Band index for MinHash-LSH
#[derive(Debug)]
pub struct MinHashBandIndex {
    /// Minimum key for each block
    mins: Vec<KeyType>,
    /// Maximum key for each block
    maxs: Vec<KeyType>,
    /// Number of items in each block
    num_in_block: Vec<usize>,
    /// Block size in bytes
    block_size: usize,
    /// Number of blocks
    blocks_num: usize,
    /// Block data (keys and values)
    data: Vec<u8>,
}

impl MinHashBandIndex {
    /// Create a new band index
    pub fn new() -> Self {
        Self {
            mins: Vec::new(),
            maxs: Vec::new(),
            num_in_block: Vec::new(),
            block_size: 8192,
            blocks_num: 0,
            data: Vec::new(),
        }
    }

    /// Build the band index from sorted KV pairs
    pub fn build(&mut self, sorted_kv: &[KVPair], block_size: usize) -> Result<usize> {
        self.build_with_interrupt(sorted_kv, block_size, &Interrupt::new())
    }

    /// Build the band index from sorted KV pairs (with interrupt support)
    pub fn build_with_interrupt(&mut self, sorted_kv: &[KVPair], block_size: usize, interrupt: &Interrupt) -> Result<usize> {
        self.block_size = block_size;
        let max_num_per_block = block_size / (std::mem::size_of::<KeyType>() + std::mem::size_of::<ValueType>());
        self.blocks_num = (sorted_kv.len() + max_num_per_block - 1) / max_num_per_block;
        
        self.mins.resize(self.blocks_num, 0);
        self.maxs.resize(self.blocks_num, 0);
        self.num_in_block.resize(self.blocks_num, 0);
        
        let mut data_buf = Vec::new();
        
        for i in 0..self.blocks_num {
            // Check interrupt periodically (every 100 blocks)
            if i % 100 == 0 && interrupt.is_interrupted() {
                return Err(KnowhereError::interrupted());
            }
            
            let beg = i * max_num_per_block;
            let end = min((i + 1) * max_num_per_block, sorted_kv.len());
            let num = end - beg;
            
            self.num_in_block[i] = num;
            self.mins[i] = sorted_kv[beg].key;
            self.maxs[i] = sorted_kv[end - 1].key;
            
            // Write keys
            for j in beg..end {
                data_buf.extend_from_slice(&sorted_kv[j].key.to_le_bytes());
            }
            // Write values
            for j in beg..end {
                data_buf.extend_from_slice(&sorted_kv[j].value.to_le_bytes());
            }
        }
        
        self.data = data_buf;
        Ok(self.blocks_num)
    }

    /// Search for a key in the band index
    pub fn search(&self, key: KeyType, res: &mut MinHashLSHResultHandler, id_selector: Option<&BitsetView>) {
        // Binary search for the block
        let block_id = match self.maxs.binary_search(&key) {
            Ok(id) | Err(id) => {
                if id >= self.maxs.len() {
                    return;
                }
                id
            }
        };

        if block_id >= self.mins.len() || key < self.mins[block_id] {
            return;
        }

        let mut current_block = block_id;
        while current_block < self.mins.len() && key >= self.mins[current_block] {
            let rows = self.num_in_block[current_block];
            let block_offset = current_block * self.block_size;
            
            // Read keys from block
            let key_start = block_offset;
            let key_end = key_start + rows * std::mem::size_of::<KeyType>();
            
            if key_end > self.data.len() {
                break;
            }

            // Binary search within the block
            let mut left = 0;
            let mut right = rows;
            while left < right {
                let mid = (left + right) / 2;
                let key_offset = key_start + mid * std::mem::size_of::<KeyType>();
                let block_key = KeyType::from_le_bytes(
                    self.data[key_offset..key_offset + std::mem::size_of::<KeyType>()].try_into().unwrap()
                );
                if block_key < key {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }

            // Collect all matching entries
            let mut inner_id = left;
            while inner_id < rows {
                let key_offset = key_start + inner_id * std::mem::size_of::<KeyType>();
                let block_key = KeyType::from_le_bytes(
                    self.data[key_offset..key_offset + std::mem::size_of::<KeyType>()].try_into().unwrap()
                );
                
                if block_key != key {
                    break;
                }

                let val_offset = key_end + inner_id * std::mem::size_of::<ValueType>();
                let block_val = ValueType::from_le_bytes(
                    self.data[val_offset..val_offset + std::mem::size_of::<ValueType>()].try_into().unwrap()
                );

                // Check ID selector (bitset filter)
                let should_include = match id_selector {
                    Some(selector) => {
                        let idx = block_val as usize;
                        idx < selector.len() && !selector.get(idx)
                    }
                    None => true,
                };

                if should_include {
                    res.push(block_val, 1.0);
                }

                if res.is_full() {
                    return;
                }

                inner_id += 1;
            }

            current_block += 1;
        }
    }

    /// Save band index to file
    pub fn save<W: Write + Seek>(&self, writer: &mut W) -> Result<u64> {
        // Write metadata first (fixed size)
        writer.write_all(&self.blocks_num.to_le_bytes())
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        writer.write_all(&self.block_size.to_le_bytes())
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        
        // Write data length and data
        let data_len = self.data.len() as u64;
        writer.write_all(&data_len.to_le_bytes())
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        writer.write_all(&self.data)
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        
        // Write mins, maxs, and num_in_block
        for &min in &self.mins {
            writer.write_all(&min.to_le_bytes())
                .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        }
        for &max in &self.maxs {
            writer.write_all(&max.to_le_bytes())
                .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        }
        for &num in &self.num_in_block {
            writer.write_all(&num.to_le_bytes())
                .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        }
        
        Ok(0)
    }

    /// Load band index from file
    pub fn load<R: Read + Seek>(&mut self, reader: &mut R) -> Result<()> {
        // Read metadata
        let mut buf = [0u8; 8];
        
        reader.read_exact(&mut buf)
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        self.blocks_num = usize::from_le_bytes(buf);
        
        reader.read_exact(&mut buf)
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        self.block_size = usize::from_le_bytes(buf);
        
        // Read data length and data
        reader.read_exact(&mut buf)
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        let data_len = u64::from_le_bytes(buf) as usize;
        
        self.data.resize(data_len, 0);
        reader.read_exact(&mut self.data)
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        
        // Read mins
        self.mins.clear();
        for _ in 0..self.blocks_num {
            reader.read_exact(&mut buf)
                .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
            self.mins.push(KeyType::from_le_bytes(buf));
        }
        
        // Read maxs
        self.maxs.clear();
        for _ in 0..self.blocks_num {
            reader.read_exact(&mut buf)
                .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
            self.maxs.push(KeyType::from_le_bytes(buf));
        }
        
        // Read num_in_block
        self.num_in_block.clear();
        for _ in 0..self.blocks_num {
            reader.read_exact(&mut buf)
                .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
            self.num_in_block.push(usize::from_le_bytes(buf));
        }
        
        Ok(())
    }
}

/// Result handler for MinHash-LSH search
#[derive(Debug)]
pub struct MinHashLSHResultHandler {
    ids: Vec<i64>,
    distances: Vec<f32>,
    topk: usize,
    counter: usize,
}

impl MinHashLSHResultHandler {
    pub fn new(topk: usize) -> Self {
        Self {
            ids: vec![-1; topk],
            distances: vec![0.0; topk],
            topk,
            counter: 0,
        }
    }

    pub fn push(&mut self, id: i64, distance: f32) {
        if self.is_full() {
            return;
        }
        if id == -1 || distance < 0.000001 {
            return;
        }
        if self.topk > 1 && self.ids[..self.counter].contains(&id) {
            return;
        }
        self.ids[self.counter] = id;
        self.distances[self.counter] = distance;
        self.counter += 1;
    }

    pub fn is_full(&self) -> bool {
        self.counter == self.topk
    }

    pub fn count(&self) -> usize {
        self.counter
    }

    pub fn get_results(self) -> (Vec<i64>, Vec<f32>) {
        (self.ids, self.distances)
    }
}

/// MinHash-LSH Index
#[derive(Debug)]
pub struct MinHashLSHIndex {
    /// Number of bands
    pub bands: usize,
    /// Size of each band
    pub band_size: usize,
    /// MinHash vector length
    pub mh_vec_length: usize,
    /// MinHash element size in bytes
    pub mh_vec_element_size: usize,
    /// Whether raw data is stored
    with_raw_data: bool,
    /// Raw vector data
    raw_data: Vec<u8>,
    /// Band indexes
    band_indexes: Vec<MinHashBandIndex>,
    /// Bloom filters for each band
    bloom_filters: Vec<BloomFilter>,
    /// Total number of vectors
    ntotal: usize,
}

/// Simple Bloom filter implementation
#[derive(Debug)]
pub struct BloomFilter {
    bits: Vec<bool>,
    size: usize,
}

impl BloomFilter {
    pub fn new(expected_items: usize, false_positive_prob: f32) -> Self {
        // Simple sizing: 10 bits per item
        let size = expected_items * 10;
        Self {
            bits: vec![false; size],
            size,
        }
    }

    pub fn add(&mut self, key: KeyType) {
        let h1 = key as usize % self.size;
        let h2 = (key >> 32) as usize % self.size;
        self.bits[h1] = true;
        self.bits[h2] = true;
    }

    pub fn contains(&self, key: KeyType) -> bool {
        let h1 = key as usize % self.size;
        let h2 = (key >> 32) as usize % self.size;
        self.bits[h1] && self.bits[h2]
    }
}

impl MinHashLSHIndex {
    /// Create a new MinHash-LSH index
    pub fn new() -> Self {
        Self {
            bands: 0,
            band_size: 0,
            mh_vec_length: 0,
            mh_vec_element_size: 0,
            with_raw_data: false,
            raw_data: Vec::new(),
            band_indexes: Vec::new(),
            bloom_filters: Vec::new(),
            ntotal: 0,
        }
    }

    /// Optimize MinHash-LSH parameters (bands and band size)
    pub fn optimize_params(mh_vec_length: usize, target_bands: usize) -> (usize, usize) {
        // Simple optimization: divide vector length by target bands
        let band_size = mh_vec_length / target_bands;
        let bands = if band_size > 0 { mh_vec_length / band_size } else { 1 };
        (bands, band_size)
    }

    /// Calculate hash key for a band
    pub fn get_hash_key(data: &[u8], band: usize, band_i: usize) -> KeyType {
        let r = data.len() / band;
        let band_start = r * band_i;
        let band_end = min(band_start + r, data.len());
        
        if band_start >= data.len() {
            return 0;
        }

        // Simple hash: FNV-1a
        let mut hash: KeyType = 0xcbf29ce484222325;
        for &byte in &data[band_start..band_end] {
            hash ^= byte as KeyType;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Generate transposed hash KV pairs
    pub fn gen_transposed_hash_kv(
        data: &[u8],
        rows: usize,
        band_num: usize,
        band_size: usize,
    ) -> Vec<KVPair> {
        let vec_size = band_num * band_size;
        let mut kv_pairs = Vec::with_capacity(rows * band_num);

        for row in 0..rows {
            let row_start = row * vec_size;
            let row_end = min(row_start + vec_size, data.len());
            
            if row_start >= data.len() {
                continue;
            }

            for band_i in 0..band_num {
                let band_start = row_start + band_i * band_size;
                let band_end = min(band_start + band_size, row_end);
                
                if band_start >= row_end {
                    break;
                }

                let key = Self::get_hash_key(&data[band_start..band_end], 1, 0);
                kv_pairs.push(KVPair {
                    key,
                    value: row as ValueType,
                });
            }
        }

        kv_pairs
    }

    /// Sort KV pairs by key
    pub fn sort_hash_kv(kv_pairs: &mut [KVPair]) {
        kv_pairs.sort_by_key(|kv| kv.key);
    }

    /// Build the index from raw data
    pub fn build(
        &mut self,
        data: &[u8],
        mh_vec_length: usize,
        mh_vec_element_size: usize,
        bands: usize,
        with_raw_data: bool,
    ) -> Result<()> {
        self.build_with_interrupt(
            data,
            mh_vec_length,
            mh_vec_element_size,
            bands,
            with_raw_data,
            &Interrupt::new(),
        )
    }

    /// Build the index from raw data (with interrupt support)
    pub fn build_with_interrupt(
        &mut self,
        data: &[u8],
        mh_vec_length: usize,
        mh_vec_element_size: usize,
        bands: usize,
        with_raw_data: bool,
        interrupt: &Interrupt,
    ) -> Result<()> {
        self.mh_vec_length = mh_vec_length;
        self.mh_vec_element_size = mh_vec_element_size;
        self.with_raw_data = with_raw_data;
        
        let vec_size = mh_vec_length * mh_vec_element_size;
        self.ntotal = if vec_size > 0 { data.len() / vec_size } else { 0 };
        
        // Optimize parameters
        let (band_num, band_size) = Self::optimize_params(mh_vec_length, bands);
        self.bands = band_num;
        self.band_size = band_size;

        // Store raw data if requested
        if with_raw_data {
            self.raw_data = data.to_vec();
        }

        // Create bloom filters
        self.bloom_filters.clear();
        for _ in 0..band_num {
            self.bloom_filters.push(BloomFilter::new(self.ntotal.max(1), 0.01));
        }

        // Build band indexes
        self.band_indexes.clear();
        
        if self.ntotal == 0 {
            return Ok(());
        }

        // Generate hash KV pairs for all bands
        let all_kv = Self::gen_transposed_hash_kv(data, self.ntotal, band_num, band_size);
        
        for band_i in 0..band_num {
            // Check interrupt for each band
            if interrupt.is_interrupted() {
                return Err(KnowhereError::interrupted());
            }
            
            let band_start = band_i * self.ntotal;
            let band_end = band_start + self.ntotal;
            
            if band_end > all_kv.len() {
                break;
            }

            let mut band_kv = all_kv[band_start..band_end].to_vec();
            Self::sort_hash_kv(&mut band_kv);

            // Add to bloom filter
            for kv in &band_kv {
                self.bloom_filters[band_i].add(kv.key);
            }

            let mut band_index = MinHashBandIndex::new();
            band_index.build_with_interrupt(&band_kv, 8192, interrupt)?;
            self.band_indexes.push(band_index);
        }

        Ok(())
    }

    /// Search for nearest neighbors
    pub fn search(
        &self,
        query: &[u8],
        topk: usize,
        id_selector: Option<&BitsetView>,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        self.search_with_interrupt(query, topk, id_selector, &Interrupt::new())
    }

    /// Search for nearest neighbors (with interrupt support)
    pub fn search_with_interrupt(
        &self,
        query: &[u8],
        topk: usize,
        id_selector: Option<&BitsetView>,
        interrupt: &Interrupt,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let mut res = MinHashLSHResultHandler::new(topk);

        for band_i in 0..self.bands {
            // Check interrupt for each band
            if interrupt.is_interrupted() {
                return Err(KnowhereError::interrupted());
            }
            
            let hash = Self::get_hash_key(query, self.bands, band_i);
            
            if self.bloom_filters[band_i].contains(hash) {
                self.band_indexes[band_i].search(hash, &mut res, id_selector);
            }

            if res.is_full() {
                break;
            }
        }

        Ok(res.get_results())
    }

    /// Batch search for multiple queries
    pub fn batch_search(
        &self,
        queries: &[u8],
        nq: usize,
        topk: usize,
        id_selectors: Option<&[BitsetView]>,
    ) -> Result<(Vec<Vec<i64>>, Vec<Vec<f32>>)> {
        self.batch_search_with_interrupt(queries, nq, topk, id_selectors, &Interrupt::new())
    }

    /// Batch search for multiple queries (with interrupt support)
    pub fn batch_search_with_interrupt(
        &self,
        queries: &[u8],
        nq: usize,
        topk: usize,
        id_selectors: Option<&[BitsetView]>,
        interrupt: &Interrupt,
    ) -> Result<(Vec<Vec<i64>>, Vec<Vec<f32>>)> {
        let mut all_ids = Vec::with_capacity(nq);
        let mut all_distances = Vec::with_capacity(nq);

        let vec_size = self.mh_vec_length * self.mh_vec_element_size;

        for i in 0..nq {
            // Check interrupt for each query
            if interrupt.is_interrupted() {
                return Err(KnowhereError::interrupted());
            }
            
            let query_start = i * vec_size;
            let query_end = query_start + vec_size;
            
            if query_end > queries.len() {
                break;
            }

            let selector = id_selectors.and_then(|s| s.get(i));
            let (ids, dists) = self.search_with_interrupt(&queries[query_start..query_end], topk, selector, interrupt)?;
            
            all_ids.push(ids);
            all_distances.push(dists);
        }

        Ok((all_ids, all_distances))
    }

    /// Get vectors by IDs
    pub fn get_vector_by_ids(&self, ids: &[i64]) -> Result<Vec<u8>> {
        if !self.with_raw_data {
            return Err(KnowhereError::new(crate::error::ErrorCode::NOT_IMPLEMENTED, "raw data not stored"));
        }

        let vec_size = self.mh_vec_length * self.mh_vec_element_size;
        let mut result = Vec::with_capacity(ids.len() * vec_size);

        for &id in ids {
            if id < 0 || (id as usize) >= self.ntotal {
                continue;
            }

            let start = (id as usize) * vec_size;
            let end = start + vec_size;
            
            if end <= self.raw_data.len() {
                result.extend_from_slice(&self.raw_data[start..end]);
            }
        }

        Ok(result)
    }

    /// Check if index has raw data
    pub fn has_raw_data(&self) -> bool {
        self.with_raw_data
    }

    /// Get number of vectors
    pub fn count(&self) -> usize {
        self.ntotal
    }

    /// Get index size in bytes
    pub fn size(&self) -> usize {
        self.ntotal * self.bands * std::mem::size_of::<KeyType>()
    }

    /// Save index to file
    pub fn save(&self, path: &str) -> Result<()> {
        let mut file = File::create(path)
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("Failed to create file: {}", e)))?;

        // Write header
        file.write_all(&self.ntotal.to_le_bytes())
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        file.write_all(&self.mh_vec_length.to_le_bytes())
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        file.write_all(&self.mh_vec_element_size.to_le_bytes())
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        file.write_all(&self.bands.to_le_bytes())
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        file.write_all(&self.band_size.to_le_bytes())
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        file.write_all(&(self.with_raw_data as u8).to_le_bytes())
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;

        // Write raw data if present
        if self.with_raw_data {
            let raw_len = self.raw_data.len();
            file.write_all(&raw_len.to_le_bytes())
                .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
            file.write_all(&self.raw_data)
                .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        }

        // Write band indexes
        for band_index in &self.band_indexes {
            band_index.save(&mut file)?;
        }

        Ok(())
    }

    /// Load index from file
    pub fn load(&mut self, path: &str) -> Result<()> {
        use std::io::BufReader;
        
        let file = File::open(path)
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("Failed to open file: {}", e)))?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut buf = [0u8; 8];

        reader.read_exact(&mut buf)
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        self.ntotal = usize::from_le_bytes(buf);

        reader.read_exact(&mut buf)
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        self.mh_vec_length = usize::from_le_bytes(buf);

        reader.read_exact(&mut buf)
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        self.mh_vec_element_size = usize::from_le_bytes(buf);

        reader.read_exact(&mut buf)
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        self.bands = usize::from_le_bytes(buf);

        reader.read_exact(&mut buf)
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        self.band_size = usize::from_le_bytes(buf);

        let mut raw_flag = [0u8; 1];
        reader.read_exact(&mut raw_flag)
            .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        self.with_raw_data = raw_flag[0] != 0;

        // Read raw data if present
        if self.with_raw_data {
            reader.read_exact(&mut buf)
                .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
            let raw_len = usize::from_le_bytes(buf);
            self.raw_data.resize(raw_len, 0);
            reader.read_exact(&mut self.raw_data)
                .map_err(|e| KnowhereError::new(crate::error::ErrorCode::IO_ERROR, &format!("IO error: {}", e)))?;
        }

        // Load band indexes
        self.band_indexes.clear();
        self.bloom_filters.clear();
        
        for _ in 0..self.bands {
            let mut band_index = MinHashBandIndex::new();
            band_index.load(&mut reader)?;
            self.band_indexes.push(band_index);
            self.bloom_filters.push(BloomFilter::new(self.ntotal, 0.01));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minhash_lsh_build() {
        let mut index = MinHashLSHIndex::new();
        
        // Create test data: 10 vectors of 8 elements each (u64)
        let data: Vec<u8> = (0..10 * 8 * 8).map(|i| i as u8).collect();
        
        index.build(&data, 8, 8, 4, true).unwrap();
        
        assert_eq!(index.count(), 10);
        assert!(index.has_raw_data());
    }

    #[test]
    fn test_minhash_lsh_search() {
        let mut index = MinHashLSHIndex::new();
        
        // Create test data
        let data: Vec<u8> = (0..100 * 8 * 8).map(|i| i as u8).collect();
        
        index.build(&data, 8, 8, 4, true).unwrap();
        
        // Search with a query vector
        let query: Vec<u8> = (0..8 * 8).map(|i| i as u8).collect();
        let (ids, dists) = index.search(&query, 5, None).unwrap();
        
        assert!(ids.len() <= 5);
        assert_eq!(ids.len(), dists.len());
    }

    #[test]
    fn test_minhash_lsh_save_load() {
        let mut index = MinHashLSHIndex::new();
        
        let data: Vec<u8> = (0..50 * 8 * 8).map(|i| i as u8).collect();
        index.build(&data, 8, 8, 4, true).unwrap();
        
        // Save to temp file
        let temp_path = "/tmp/test_minhash_lsh.bin";
        index.save(temp_path).unwrap();
        
        // Load from file
        let mut loaded_index = MinHashLSHIndex::new();
        loaded_index.load(temp_path).unwrap();
        
        assert_eq!(loaded_index.count(), index.count());
        assert_eq!(loaded_index.bands, index.bands);
        
        // Cleanup
        std::fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_minhash_lsh_get_vector_by_ids() {
        let mut index = MinHashLSHIndex::new();
        
        let data: Vec<u8> = (0..10 * 8 * 8).map(|i| i as u8).collect();
        index.build(&data, 8, 8, 4, true).unwrap();
        
        let ids = vec![0i64, 5, 9];
        let vectors = index.get_vector_by_ids(&ids).unwrap();
        
        assert_eq!(vectors.len(), 3 * 8 * 8);
    }

    #[test]
    fn test_bloom_filter() {
        let mut bf = BloomFilter::new(100, 0.01);
        
        bf.add(42);
        bf.add(100);
        
        assert!(bf.contains(42));
        assert!(bf.contains(100));
        assert!(!bf.contains(999));
    }

    #[test]
    fn test_hash_key() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let hash = MinHashLSHIndex::get_hash_key(&data, 1, 0);
        
        assert!(hash > 0);
    }

    #[test]
    fn test_result_handler() {
        let mut res = MinHashLSHResultHandler::new(5);
        
        res.push(1, 0.9);
        res.push(2, 0.8);
        res.push(3, 0.7);
        
        assert_eq!(res.count(), 3);
        assert!(!res.is_full());
        
        res.push(4, 0.6);
        res.push(5, 0.5);
        
        assert!(res.is_full());
        
        // Should not add more
        res.push(6, 0.4);
        assert_eq!(res.count(), 5);
    }
}
