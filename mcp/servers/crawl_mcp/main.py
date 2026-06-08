import json

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from pyapplib.crawler import CrawlConfig, WebCrawler

# Load environment variables from .env file
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("crawl-mcp")

@mcp.tool()
async def crawl(url: str) -> str:
    """Crawl a website and return the results.
    Args:
        url: The URL of the website to crawl.
    Returns:
        A JSON string containing the crawled pages.
    """
    config = CrawlConfig(
        max_depth=1,
        max_pages=10,
        describe_page=True,
        describe_images=True,
    )
    try:
        async with WebCrawler(config) as crawler:
            pages = await crawler.crawl(url)
            return json.dumps(pages, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

def main():
    """Initialize and run the MCP server."""
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        print(f"Error initializing server: {e}")
        exit(1)

if __name__ == "__main__":
    main()
