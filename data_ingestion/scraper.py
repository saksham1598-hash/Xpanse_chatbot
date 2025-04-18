import asyncio
import argparse
import os
import re
import json
from pathlib import Path
from crawl4ai import AsyncWebCrawler
import sys

current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = current_dir.parent

sys.path.append(str(root_dir))

# Import the configuration
from config import RAG_CONFIG

# Configure logging
from utils.logger import get_logger
logger = get_logger()


def extract_section(markdown: str, url: str) -> tuple[str, bool]:
    """
    Extract a section from markdown content, or fallback to full content.

    Returns:
        Tuple of (extracted content, whether section header was found)
    """
    start_index = markdown.find("# §")

    if start_index != -1:
        marker = f"[ Back to top ]({url})"
        end_index = markdown.find(marker, start_index)

        if end_index == -1:
            extracted = markdown[start_index:].strip()
            logger.warning(f"No end marker found. Extracted from section header to end.")
        else:
            extracted = markdown[start_index:end_index].strip()
            logger.info(f"Extracted section from '{url}' based on header and marker.")
        return extracted, True

    # Case 2: Fallback to full content
    logger.warning(f"No section header ('# §') found in {url}. Falling back to full page content.")
    return markdown.strip(), False

def safe_filename_from_url(url: str) -> str:
    """
    Create a safe filename from a URL.
    
    Args:
        url: The URL to convert to a filename
        
    Returns:
        A filename-safe string derived from the URL
    """
    path = re.sub(r"https?://", "", url)
    path = re.sub(r"[^\w\-_/]", "_", path)
    return path.strip("/").replace("/", "_") + ".md"

def save_markdown_and_mapping(url: str, markdown_data: str, is_section: bool = True):
    """
    Save markdown content to a file and update the URL mapping.
    
    Args:
        url: The source URL
        markdown_data: The markdown content to save
        is_section: Whether the content is a section (True) or full content (False)
    """
    # Get the output directory from the config file
    output_dir = RAG_CONFIG["URL_EXTRACT"]["markdown_files_path"]
    
    # Convert to Path object if it's not already
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    
    # Ensure directory exists
    output_dir.mkdir(exist_ok=True, parents=True)

    filename = safe_filename_from_url(url)
    full_path = output_dir / filename

    # Check if file already exists
    if full_path.exists():
        logger.info(f"File already exists for {url}. Overwriting.")
    
    # Save markdown
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(markdown_data)

    mapping_path = output_dir / "url_to_file.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            url_map = json.load(f)
    else:
        url_map = {}

    # Update mapping with additional metadata
    url_map[url] = {
        "file_path": str(full_path),
        "is_section": is_section,
        "timestamp": str(Path(full_path).stat().st_mtime),
        "size_bytes": Path(full_path).stat().st_size
    }
    
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(url_map, f, indent=2)

    logger.info(f"Markdown saved to: {full_path}")
    logger.info(f"Mapping updated in: {mapping_path}")
    

async def process_url(url: str) -> bool:
    """
    Process a URL: crawl it, extract content, and save to markdown.
    
    Args:
        url: The URL to process
        
    Returns:
        True if processing succeeded, False otherwise
    """
    logger.info(f"Processing URL: {url}")
    
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)

            if result.success:
                markdown_data = result.markdown
                section_content, is_section = extract_section(markdown_data, url)

                # Log content info
                content_preview = section_content[:200] + "..." if len(section_content) > 200 else section_content
                logger.info(f"Extracted content preview: {content_preview}")

                # Save content
                save_markdown_and_mapping(url, section_content, is_section)
                return True
            else:
                logger.error(f"Failed to retrieve content from {url}. Status code: {result.status_code}")
                return False
    except Exception as e:
        logger.error(f"Error processing {url}: {str(e)}")
        return False

async def process_all_urls():
    """Process all URLs defined in the configuration"""
    if "urls" not in RAG_CONFIG["URL_EXTRACT"]:
        logger.error("No URLs defined in configuration. Add URLs to RAG_CONFIG['URL_EXTRACT']['urls']")

        return

    urls = RAG_CONFIG["URL_EXTRACT"]["urls"]
    logger.info(f"Processing {len(urls)} URLs from configuration")
 
    
    success_count = 0
    for url in urls:
        if await process_url(url):
            success_count += 1
    
    logger.info(f"URL processing completed. Success: {success_count}/{len(urls)}")
  

async def main():
    """Main function to handle command line arguments and process URLs"""
    parser = argparse.ArgumentParser(description="Extract content from URLs and save as markdown.")
    parser.add_argument("--url", help="Process a single URL instead of using the config")
    parser.add_argument("--force", action="store_true", help="Force processing even if file exists")
    args = parser.parse_args()
    
    if args.url:
        # Process a single URL provided via command line
        success = await process_url(args.url)
        if success:
            print("URL processing completed successfully.")
        else:
            print(" URL processing failed.")
    else:
        # Process all URLs from the configuration
        await process_all_urls()

if __name__ == "__main__":
    asyncio.run(main())