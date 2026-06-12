//! Milvus search core for milvus-sdk-rust 0.1.0 (compile-ready)

use std::{str::FromStr, sync::Arc, time::Instant};

use anyhow::Result;
use milvus::client::Client;
use milvus::collection::{Collection, SearchOption, SearchResult};
use milvus::index::MetricType;
use milvus::value::{Value, ValueVec}; 

use heungguk_poc_rag::pb::{FilterField, SearchHit, SearchRequest, SearchResponse};

use crate::embedding::Embedder;

#[derive(Clone)]
pub struct Config {
    pub collection:      String,
    pub milvus_url:      String,
    pub embedding_model: String,
    pub device:          String,
    pub doc_field:       String,
    pub metric_type:     String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            //insert vector storage in order to use this
            collection: std::env::var("COLLECTION_NAME")
                .unwrap_or_else(|_| "VECTOR-STORAGE".into()),
            milvus_url: format!(
                "http://{}:{}",
                std::env::var("MILVUS_HOST").unwrap_or_else(|_| "localhost".into()),
                std::env::var("MILVUS_PORT").unwrap_or_else(|_| "19530".into()),
            ),
            // it is free to choose your embedding model. but be careful of shape
            embedding_model: std::env::var("EMBEDDING_MODEL")
                .unwrap_or_else(|_| "XIANXIE31-EMB-CHAMPON".into()),
            device:      std::env::var("DEVICE").unwrap_or_else(|_| "cuda".into()),
            doc_field:   "doc_id".into(),
            metric_type: "IP".into(),
        }
    }
}

pub struct SearchCore {
    embedder:      Arc<Embedder>,
    collection:    Collection,
    doc_names:     Vec<String>,
    filter_fields: Vec<FilterField>,
    cfg:           Config,
}

impl SearchCore {
    pub async fn new(cfg: &Config) -> Result<Self> {
        let embedder   = Arc::new(Embedder::new(&cfg.embedding_model, &cfg.device)?);
        let client     = Client::new(cfg.milvus_url.clone()).await?;
        let collection = client.get_collection(&cfg.collection).await?;
        // match filter option -> milvus
        let filter_fields = vec![
            FilterField { name: "doc_id".into(), dtype: "string".into() },
            FilterField { name: "page".into(),   dtype: "int".into()    },
        ];

        Ok(Self {
            embedder,
            collection,
            doc_names: Vec::new(),
            filter_fields,
            cfg: cfg.clone(),
        })
    }

    pub fn list_fields(&self) -> Vec<FilterField> { self.filter_fields.clone() }
    pub fn list_docs(&self)   -> Vec<String>      { self.doc_names.clone() }

    pub async fn search(&self, req: SearchRequest) -> Result<SearchResponse> {
        let t0 = Instant::now();

        let emb = self.embedder.encode(&req.query)?;

        let mut option = SearchOption::default();
        if !req.filter.trim().is_empty() {       
            option.set_expr(&req.filter);
        }
        option.add_param("ef", 100_i64.into());

        let output_fields = vec!["doc_id", "page", "text"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();

        let results: Vec<SearchResult> = self
            .collection
            .search(
                vec![Value::from(emb.clone())],
                "embedding",
                req.num_results as i32,
                MetricType::from_str(&self.cfg.metric_type).unwrap_or(MetricType::IP),
                &output_fields,
                &option,
            )
            .await?;

        if results.is_empty() {
            return Ok(SearchResponse { hits: vec![], latency_ms: t0.elapsed().as_secs_f64() * 1_000.0 });
        }

        let first = &results[0];            
        let mut hits_pb = Vec::with_capacity(first.size as usize);

        for rank in 0..first.size as usize {
            let doc_id = match &first.field[0].value {
                ValueVec::String(vs) => vs[rank].to_owned(),
                _                => "<unknown>".into(),
            };

            let page = match &first.field[1].value {
                ValueVec::Long(vs) => vs[rank] as i32,
                ValueVec::Int(vs) => vs[rank],
                _               => 0,
            };

            let mut text = match &first.field[2].value {
                ValueVec::String(vs) => vs[rank].to_owned(),
                _                => String::new(),
            };
            if req.max_chars > 0 && text.len() > req.max_chars as usize {
                text.truncate(req.max_chars as usize);
                text.push('…');
            }

            let score = first.score[rank] as f64;
            hits_pb.push(SearchHit { rank: (rank as i32) + 1, doc_id, page, text, score });
        }

        Ok(SearchResponse { hits: hits_pb, latency_ms: t0.elapsed().as_secs_f64() * 1_000.0 })
    }
}
