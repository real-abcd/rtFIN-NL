use anyhow::Result;
use clap::{arg};
use heungguk_poc_rag::pb::{
    evidence_search_client::EvidenceSearchClient,
    SearchRequest,
};
#[tokio::main]

async fn main() -> Result<()> {
    let matches = clap::Command::new("client")
        .arg(arg!(--host <HOST>).default_value("127.0.0.1"))
        .arg(arg!(--port <PORT>).default_value("50051"))
        .arg(arg!(--query <TEXT>).required(true))
        .arg(arg!(--k <INT>).default_value("5"))
        .arg(arg!(--filter <EXPR>).default_value(""))
        .arg(arg!(--max_chars <INT>).default_value("300"))
        .get_matches();

    let host = matches.get_one::<String>("host").expect("host is required");
    let port = matches.get_one::<String>("port").expect("port is required");
    let query = matches.get_one::<String>("query").expect("query is required");

    let k:  i32 = matches
        .get_one::<String>("k")
        .unwrap()
        .parse()
        .expect("k must be an integer");

    let filter = matches.get_one::<String>("filter").unwrap();

    let max_chars: i32 = matches
        .get_one::<String>("max_chars")
        .unwrap()
        .parse()
        .expect("max_chars must be an integer");

    let target = format!("http://{}:{}", host, port);
    
    let mut client = EvidenceSearchClient::connect(target).await?;

    let req = tonic::Request::new(SearchRequest {
        query: query.clone(),
        num_results: k,
        filter: filter.clone(),
        max_chars,
    });

    let resp = client.search(req).await?.into_inner();

    if resp.hits.is_empty() {
        println!("(no hits)");
    } else {
        for h in resp.hits {
            println!("[{}] {} p{}\n{}\n", h.rank, h.doc_id, h.page, h.text);
        }
        println!("elapsed {:.1} ms", resp.latency_ms);
    }
    Ok(())
}