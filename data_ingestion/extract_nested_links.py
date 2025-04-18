import re
import sys
import os
import asyncio
import argparse
import logging
import json
from datetime import datetime
from urllib.parse import urldefrag, urlparse, urlunparse
from pathlib import Path
from scraper import extract_section
from crawl4ai import AsyncWebCrawler
from config import RAG_CONFIG

current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = current_dir.parent
sys.path.append(str(root_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def extract_links_from_markdown(markdown: str) -> list:
    """
    Find all HTTP/HTTPS links in the markdown.
    """
    links = re.findall(r'\[.*?\]\((https?://.*?)\)', markdown)
    logger.info("Found %d raw links in markdown", len(links))
    return links


def modify_link(link: str) -> str:
    """
    Apply custom modifications to a link: remove fragments, adjust regulation paths, etc.
    """
    link, _ = urldefrag(link)
    parsed = urlparse(link)
    path_parts = [part for part in parsed.path.split('/') if part]
    logger.debug("Processing link: %s", link)

    # Get the exclude_regulations and default_regulation from config
    exclude_regulations = RAG_CONFIG["URL_EXTRACT"].get("link_filters", {}).get("exclude_regulations", ["17"])
    default_regulation = RAG_CONFIG["URL_EXTRACT"].get("link_filters", {}).get("default_regulation", "1024")

    try:
        reg_index = path_parts.index("regulations")
    except ValueError:
        logger.debug("No 'regulations' in path, keeping unchanged: %s", link)
        return link

    if reg_index + 1 < len(path_parts):
        segment = path_parts[reg_index + 1]
        if segment in exclude_regulations:
            logger.debug(f"Dropping link to excluded regulation {segment}: %s", link)
            return None
        if segment.isdigit() and segment != default_regulation:
            logger.debug(f"Detected regulation root is not {default_regulation} ({segment}), not modifying", segment)
            return link
    else:
        logger.debug(f"Adding {default_regulation} to empty regulation path: %s", link)
        path_parts.append(default_regulation)

    new_path = "/" + "/".join(path_parts) + "/"
    new_parsed = parsed._replace(path=new_path)
    modified = urlunparse(new_parsed)
    logger.debug("Modified link: %s", modified)
    return modified


async def extract_links(url: str) -> set[str]:
    """
    Crawl the provided URL, extract markdown content and links,
    then return a set of unique, modified links.
    """
    logger.info("Processing URL: %s", url)
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)

            if not result.success:
                raise ValueError(f"Failed to retrieve content from {url}. Status code: {result.status_code}")

            markdown_data = result.markdown
            section_content, is_section = extract_section(markdown_data, url)
            logger.info("Extracted %s content (%d chars) from %s",
                        "section" if is_section else "full", len(section_content), url)

            links = extract_links_from_markdown(section_content)
            modified_links = []

            for link in links:
                modified = modify_link(link)
                if modified:
                    modified_links.append(modified)
                    logger.info("Link modified: %s -> %s", link, modified)
                else:
                    logger.info("Link filtered out: %s", link)

            return set(modified_links)
    except Exception as e:
        logger.error("Error processing %s: %s", url, str(e))
        raise


async def process_url_links(url: str, save: bool = True) -> bool:
    """
    Process a single URL to extract links.
    """
    try:
        links = await extract_links(url)
        logger.info("Found %d unique valid modified links.", len(links))
        for i, link in enumerate(sorted(links), 1):
            logger.info("%d. %s", i, link)

        if save:
            section_id = "_".join(url.strip("/").split("/")[-2:])
            extraction_time = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_data = {
                "parent_url": url,
                "extraction_time": extraction_time,
                "total_links": len(links),
                "links": sorted(links),
            }

            output_dir = RAG_CONFIG["URL_EXTRACT"]["raw_links_path"]
            # Ensure output directory exists
            if not isinstance(output_dir, Path):
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{section_id}_{extraction_time}.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)

            logger.info("Output saved to: %s", output_file)

        return True
    except Exception as e:
        logger.error("Error: %s", str(e))
        return False


async def process_all_urls(save: bool = True):
    """Process all URLs defined in the configuration"""
    if "urls" not in RAG_CONFIG["URL_EXTRACT"]:
        logger.error("No URLs defined in configuration. Add URLs to RAG_CONFIG['URL_EXTRACT']['urls']")
        return

    urls = RAG_CONFIG["URL_EXTRACT"]["urls"]
    logger.info(f"Processing {len(urls)} URLs from configuration")
    
    success_count = 0
    for url in urls:
        if await process_url_links(url, save=save):
            success_count += 1
    
    logger.info(f"Link extraction completed. Success: {success_count}/{len(urls)}")

async def main():
    """Main function to handle command line arguments and process URLs"""
    parser = argparse.ArgumentParser(
        description="Extract links from URLs based on custom rules."
    )
    parser.add_argument("--url", help="Process a single URL instead of using the config")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Don't save output JSON")
    parser.set_defaults(save=True)
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logging.root.handlers:
            handler.setLevel(logging.DEBUG)

    if args.url:
        # Process a single URL provided via command line
        success = await process_url_links(args.url, save=args.save)
        if success:
            print(" Link extraction completed successfully.")
        else:
            print(" Link extraction failed.")
    else:
        # Process all URLs from the configuration
        await process_all_urls(save=args.save)


if __name__ == "__main__":
    asyncio.run(main())