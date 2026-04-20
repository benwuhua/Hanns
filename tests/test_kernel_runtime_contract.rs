use hanns::api::SearchRequest;
use hanns::kernel::{AnnRuntime, IndexFamily};

struct EmptyRuntime;

impl AnnRuntime for EmptyRuntime {
    fn family(&self) -> IndexFamily {
        IndexFamily::Flat
    }

    fn dim(&self) -> usize {
        4
    }

    fn len(&self) -> usize {
        0
    }

    fn search_into(
        &self,
        _query: &[f32],
        _req: &SearchRequest,
        _ids: &mut [i64],
        _dists: &mut [f32],
    ) -> hanns::api::Result<usize> {
        Ok(0)
    }
}

#[test]
fn ann_runtime_trait_is_public_and_object_safe() {
    let runtime: Box<dyn AnnRuntime> = Box::new(EmptyRuntime);
    assert_eq!(runtime.family(), IndexFamily::Flat);
    assert_eq!(runtime.dim(), 4);
    assert_eq!(runtime.len(), 0);

    let req = SearchRequest {
        top_k: 4,
        nprobe: 8,
        filter: None,
        params: None,
        radius: None,
    };
    let mut ids = [-1_i64; 4];
    let mut dists = [f32::INFINITY; 4];
    assert_eq!(
        runtime
            .search_into(&[0.0; 4], &req, &mut ids, &mut dists)
            .unwrap(),
        0
    );
}
