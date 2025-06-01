#![allow(clippy::all, warnings)]
pub mod google {
    pub mod protobuf {
        tonic::include_proto!("google.protobuf");
    }
}

pub mod search_list {
    tonic::include_proto!("search_list");  
}

pub mod pb {
    tonic::include_proto!("search_list");  
}