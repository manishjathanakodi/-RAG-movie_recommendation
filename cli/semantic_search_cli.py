#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text, 
    SemanticSearch)
from lib.search_utils import load_movies

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")
    subparsers.add_parser("verify_embeddings", help="Verify that the embeddings are loaded")
    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for input text")
    embed_parser.add_argument("query", type=str, help="Text to generate embedding for")
    embed_query_parser = subparsers.add_parser("embedquery", help="Generate embedding for input query text")
    embed_query_parser.add_argument("query", type=str, help="Query text to generate embedding for")
    search_parser = subparsers.add_parser("search", help="Search for similar documents using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of top results to return"
    )
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.query)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_instance = SemanticSearch()
            movies = load_movies()
            search_instance.load_or_create_embeddings(movies)
            results = search_instance.search(args.query, args.limit)
            for score, doc in results:
                print(f"{doc['title']} (score: {score:.4f})\n{doc['description']}\n")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()