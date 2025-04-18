import asyncio
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from config import RAG_CONFIG
from scraper import extract_section, save_markdown_and_mapping

# Import the web crawler
from crawl4ai import AsyncWebCrawler


# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_link(url: str, crawler: AsyncWebCrawler) -> bool:
    """
    Process a single URL: extract its content, extract the section, and save it.
    
    Args:
        url: The URL to process
        crawler: AsyncWebCrawler instance to use
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        logger.info(f"Processing URL: {url}")
        result = await crawler.arun(url=url)
        
        if result.success:
            markdown_data = result.markdown
            section_content, is_section = extract_section(markdown_data, url)

            if not is_section:
                logger.warning(f"No valid section header found in {url}. Saving full content.")

            save_markdown_and_mapping(url, section_content, is_section)
            return True
        else:
            logger.error(f"Failed to retrieve content from {url}. Status code: {result.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        return False

async def process_links_from_json(json_path: Path, max_concurrent: int = 3, batch_delay: float = 1.0) -> Dict:
    """
    Process links from a JSON file and generate markdown files.
    
    Args:
        json_path: Path to the JSON file containing links
        max_concurrent: Maximum number of concurrent requests
        batch_delay: Delay between batches in seconds
        
    Returns:
        Dictionary with processing statistics
    """
    # Load links from JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    links = data.get("links", [])
    parent_url = data.get("parent_url", "unknown")
    total_links = len(links)
    
    logger.info(f"Loaded {total_links} links from {json_path}")

    
    if total_links == 0:
        logger.warning("No links found to process")
        return {
            "parent_url": parent_url,
            "total_links": 0,
            "successful": 0,
            "failed": 0
        }
    
    # Process each link with limited concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    successful = 0
    failed = 0
    
    async def bounded_process(url):
        async with semaphore:
            return await process_link(url, crawler)
    
    # Create smaller batches to avoid connection issues
    batch_size = 5
    batches = [links[i:i + batch_size] for i in range(0, len(links), batch_size)]
    
    async with AsyncWebCrawler() as crawler:
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} links)")  
            tasks = [bounded_process(url) for url in batch]
            results = await asyncio.gather(*tasks)
            
            batch_success = sum(1 for result in results if result)
            batch_fail = len(batch) - batch_success
            
            successful += batch_success
            failed += batch_fail
            
            logger.info(f"Batch {i+1} complete: {batch_success} successful, {batch_fail} failed")
            
            if i < len(batches) - 1:
                await asyncio.sleep(batch_delay)
    
    # Save processing summary
    stats = {
        "parent_url": parent_url,
        "total_links": total_links,
        "successful": successful,
        "failed": failed,
        "json_source": str(json_path)
    }
    
    # Save summary to a file with a name related to the source JSON
    summary_filename = f"summary_{json_path.stem}.json"
    summary_path = RAG_CONFIG["URL_EXTRACT"]["markdown_files_path"] / "summaries" / summary_filename
    
    # Ensure the summaries directory exists
    summary_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Processing summary saved to {summary_path}")

    
    return stats

async def process_all_json_files(directory: Path, max_concurrent: int = 3, batch_delay: float = 1.0) -> List[Dict]:
    """
    Process all JSON files in a directory containing link data.
    
    Args:
        directory: Directory containing JSON files with links
        max_concurrent: Maximum number of concurrent requests
        batch_delay: Delay between batches in seconds
        
    Returns:
        List of dictionaries with processing statistics for each file
    """
    json_files = list(directory.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {directory}")
        return []
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    all_stats = []
    for json_file in json_files:
        if json_file.stem == "url_to_file" or json_file.parent.name == "summaries":
            # Skip the url mapping file and summary files
            continue
            
        stats = await process_links_from_json(json_file, max_concurrent, batch_delay)
        all_stats.append(stats)
    
    # Create overall summary
    overall_summary = {
        "total_files_processed": len(all_stats),
        "total_links": sum(s["total_links"] for s in all_stats),
        "total_successful": sum(s["successful"] for s in all_stats),
        "total_failed": sum(s["failed"] for s in all_stats),
        "file_summaries": all_stats
    }
    
    # Save overall summary
    overall_path = RAG_CONFIG["URL_EXTRACT"]["markdown_files_path"] / "summaries" / "overall_summary.json"
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)
    
    logger.info(f"Overall processing summary saved to {overall_path}")
    
    return all_stats

async def main():
    """Main function to handle command line arguments and process links"""
    parser = argparse.ArgumentParser(
        description="Process links from JSON files and generate markdown files"
    )
    parser.add_argument("--json-file", 
                      help="Path to a specific JSON file containing links (optional)")
    parser.add_argument("--all", action="store_true",
                      help="Process all JSON files in the raw_links_path directory")
    parser.add_argument("--concurrency", type=int, default=3,
                      help="Maximum number of concurrent requests (default: 3)")
    parser.add_argument("--batch-delay", type=float, default=1.0,
                      help="Delay between batches in seconds (default: 1.0)")
    
    args = parser.parse_args()
    
    # Get the output directory from config
    output_dir = RAG_CONFIG["URL_EXTRACT"]["markdown_files_path"]
    links_dir = RAG_CONFIG["URL_EXTRACT"]["raw_links_path"]
    
    # Ensure both directories exist
    output_dir.mkdir(exist_ok=True, parents=True)
    links_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Markdown files will be saved to: {output_dir}")    
    try:
        if args.json_file:
            # Process a specific JSON file
            json_path = Path(args.json_file)
            if not json_path.exists():
                print(f"Error: File not found: {json_path}")
                return
                
            stats = await process_links_from_json(json_path, args.concurrency, args.batch_delay)
            
            if stats['total_links'] > 0:
                success_rate = (stats['successful'] / stats['total_links']) * 100
                print(f"  - Success rate: {success_rate:.1f}%")
            
        elif args.all or not args.json_file:
            # Process all JSON files in the links directory
            all_stats = await process_all_json_files(links_dir, args.concurrency, args.batch_delay)
            
            if all_stats:
                # Print overall summary
                total_links = sum(s["total_links"] for s in all_stats)
                total_successful = sum(s["successful"] for s in all_stats)
                total_failed = sum(s["failed"] for s in all_stats)
                
                print(f"  - Total links: {total_links}")
                print(f"  - Successfully processed: {total_successful}")
                print(f"  - Failed: {total_failed}")
            
            else:
                print("No files were processed.")
    
    except Exception as e:
        logger.error(f"Failed to process links: {e}")
        print(f" Failed to process links: {e}")

if __name__ == "__main__":
    asyncio.run(main())