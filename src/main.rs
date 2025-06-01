// src/main.rs

pub mod google {
    pub mod protobuf {
        tonic::include_proto!("google.protobuf");
    }
}

pub mod evidence {
    tonic::include_proto!("search_list");
}

use anyhow::Result;
use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};
use tracing::info;

use heungguk_poc_rag::pb::{
    evidence_search_server::{EvidenceSearch, EvidenceSearchServer},
    ListDocNamesResponse,
    ListFilterFieldsResponse,
    SearchRequest,
    SearchResponse,
};

use heungguk_poc_rag::google::protobuf::Empty;

mod embedding;
mod search;
use search::{Config, SearchCore};

#[derive(Clone)]
struct EvidenceSvc {
    core: Arc<SearchCore>,
}

#[tonic::async_trait]
impl EvidenceSearch for EvidenceSvc {

    async fn search(
        &self,
        _req: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let resp = SearchResponse { hits: Vec::new(), latency_ms: 0.0 };
        Ok(Response::new(resp))
    }

    async fn list_filter_fields(
        &self,
        _req: Request<Empty>,
    ) -> Result<Response<ListFilterFieldsResponse>, Status> {
        let fields = self.core.list_fields();
        Ok(Response::new(ListFilterFieldsResponse { fields }))
    }


    async fn list_doc_names(
        &self,
        _req: Request<Empty>,
    ) -> Result<Response<ListDocNamesResponse>, Status> {
        let names = self.core.list_docs();
        Ok(Response::new(ListDocNamesResponse { doc_names: names }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let host = std::env::var("BIND_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port: u16 = std::env::var("BIND_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50055);

    let cfg  = Config::default();
    let core = SearchCore::new(&cfg).await?;
    let svc  = EvidenceSvc { core: Arc::new(core) };

    info!("서버가돌아갑니다~ {host}:{port}");
    Server::builder()
        .add_service(EvidenceSearchServer::new(svc))
        .serve(format!("{host}:{port}").parse()?)
        .await?;

    Ok(())
}