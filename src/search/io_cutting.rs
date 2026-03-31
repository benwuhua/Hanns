pub struct IoCuttingState {
    stable_count: u32,
    change_point_index: u32,
    search_list_size: u32,
    threshold: f32,
}

impl IoCuttingState {
    pub fn new(l: usize, threshold: f32) -> Self {
        Self {
            stable_count: 0,
            change_point_index: 0,
            search_list_size: l as u32,
            threshold,
        }
    }

    #[inline]
    pub fn record(&mut self, is_valid: bool, insert_pos: usize) {
        if is_valid {
            self.stable_count = 0;
            self.change_point_index = insert_pos as u32;
        } else {
            self.stable_count += 1;
        }
    }

    #[inline]
    pub fn should_stop(&self) -> bool {
        let remaining =
            (self.search_list_size.saturating_sub(1)).saturating_sub(self.change_point_index);
        if remaining == 0 {
            return true;
        }
        self.stable_count as f32 >= (remaining as f32 * self.threshold).ceil()
    }
}
