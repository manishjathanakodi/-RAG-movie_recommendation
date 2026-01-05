#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer
with open("./data/movies.json") as f:
    movies = json.load(f)
with open("./data/stopwords.txt") as sw:
    stop_words = sw.read().splitlines()

def main() -> None:
    stemmer = PorterStemmer()
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    movies["movies"]
    cleaner = str.maketrans('', '', string.punctuation)
    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}") 
        # 1. Clean the query ONCE (not inside the movie loop)
            split_input = args.query.translate(cleaner).lower().split()
            cleaned_input = [word for word in split_input if word not in stop_words]    
            i = 1       
            for movie in movies["movies"]:
            # 2. Clean the title
                cleaned_match = movie["title"].translate(cleaner).lower().split()
            # 3. Check if any query word is inside any title word
            # We use a flag or 'any' to ensure we only print ONCE per movie
                found = False
                for query_word in cleaned_input:
                    for title_word in cleaned_match:
                        if stemmer.stem(query_word) in title_word:
                            found = True
                            break # Stop looking at title words for THIS query word
                    if found:
                        break # Stop looking at other query words for THIS movie
            
                if found:
                    print(f"{i}. {movie['title']}")
                    i += 1
                if i > 5:
                    break
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
