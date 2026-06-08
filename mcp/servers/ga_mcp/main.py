import os
import json
from typing import Any, List, Optional, Dict
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env file
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("google-analytics-mcp")

# Constants
SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']

def init_ga4_client():
    """Initialize Google Analytics 4 client using service account."""
    global analytics_service
    
    # Get service account file path from environment variable
    service_account_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not service_account_file:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required")
    
    if not os.path.exists(service_account_file):
        raise FileNotFoundError(f"Service account file not found: {service_account_file}")
    
    # Authenticate using service account credentials
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES)
    
    # Build the Google Analytics Data API client (for GA4)
    return build('analyticsdata', 'v1beta', credentials=credentials)

def format_analytics_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Format the analytics response to match the JS version structure."""
    return {
        "rows": response.get('rows', []),
        "totals": response.get('totals', []),
        "rowCount": response.get('rowCount', 0)
    }

@mcp.tool()
async def query_analytics(property_id: str, start_date: str, end_date: str, metrics: List[str], 
                         dimensions: List[str], filters: Optional[Dict] = None) -> str:
    """Query Google Analytics data with custom metrics, dimensions, and filters.
    
    Args:
        property_id: Google Analytics 4 property ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        metrics: List of metric names to query
        dimensions: List of dimension names to query
        filters: Optional dimension filters
    """
    try:
        client = init_ga4_client()
        request_body = {
            'dateRanges': [{'startDate': start_date, 'endDate': end_date}],
            'metrics': [{'name': metric} for metric in metrics],
            'dimensions': [{'name': dimension} for dimension in dimensions]
        }
        
        if filters:
            request_body['dimensionFilter'] = filters
        
        response = client.properties().runReport(
            property=f'properties/{property_id}',
            body=request_body
        ).execute()
        
        result = format_analytics_response(response)
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@mcp.tool()
async def get_realtime_data(property_id: str, dimensions: Optional[List[str]] = None, 
                           metrics: Optional[List[str]] = None) -> str:
    """Get real-time Google Analytics data.
    
    Args:
        property_id: Google Analytics 4 property ID
        dimensions: List of dimension names (default: ['country'])
        metrics: List of metric names (default: ['activeUsers'])
    """
    try:
        client = init_ga4_client()
        if dimensions is None:
            dimensions = ['country']
        if metrics is None:
            metrics = ['activeUsers']
        
        request_body = {
            'dimensions': [{'name': dimension} for dimension in dimensions],
            'metrics': [{'name': metric} for metric in metrics]
        }
        
        response = client.properties().runRealtimeReport(
            property=f'properties/{property_id}',
            body=request_body
        ).execute()
        
        result = format_analytics_response(response)
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@mcp.tool()
async def get_traffic_sources(property_id: str, start_date: str, end_date: str) -> str:
    """Get traffic source analysis data.
    
    Args:
        property_id: Google Analytics 4 property ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    try:
        client = init_ga4_client()
        request_body = {
            'dateRanges': [{'startDate': start_date, 'endDate': end_date}],
            'dimensions': [
                {'name': 'sessionSource'},
                {'name': 'sessionMedium'},
                {'name': 'sessionCampaign'}
            ],
            'metrics': [
                {'name': 'sessions'},
                {'name': 'users'},
                {'name': 'newUsers'},
                {'name': 'bounceRate'}
            ]
        }
        
        response = client.properties().runReport(
            property=f'properties/{property_id}',
            body=request_body
        ).execute()
        
        result = format_analytics_response(response)
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@mcp.tool()
async def get_user_demographics(property_id: str,start_date: str, end_date: str) -> str:
    """Get user demographics data.
    
    Args:
        property_id: Google Analytics 4 property ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    try:
        client = init_ga4_client()
        request_body = {
            'dateRanges': [{'startDate': start_date, 'endDate': end_date}],
            'dimensions': [
                {'name': 'userAgeBracket'},
                {'name': 'userGender'},
                {'name': 'country'},
                {'name': 'city'}
            ],
            'metrics': [
                {'name': 'users'},
                {'name': 'newUsers'},
                {'name': 'sessions'},
                {'name': 'averageSessionDuration'}
            ]
        }
        
        response = client.properties().runReport(
            property=f'properties/{property_id}',
            body=request_body
        ).execute()
        
        result = format_analytics_response(response)
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@mcp.tool()
async def get_page_performance(property_id: str, start_date: str, end_date: str) -> str:
    """Get page performance metrics.
    
    Args:
        property_id: Google Analytics 4 property ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    try:
        client = init_ga4_client()
        request_body = {
            'dateRanges': [{'startDate': start_date, 'endDate': end_date}],
            'dimensions': [
                {'name': 'pagePath'},
                {'name': 'pageTitle'}
            ],
            'metrics': [
                {'name': 'screenPageViews'},
                {'name': 'uniquePageviews'},
                {'name': 'averageSessionDuration'},
                {'name': 'bounceRate'},
                {'name': 'exitRate'}
            ]
        }
        
        response = client.properties().runReport(
            property=f'properties/{property_id}',
            body=request_body
        ).execute()
        
        result = format_analytics_response(response)
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@mcp.tool()
async def get_conversion_data(property_id: str, start_date: str, end_date: str) -> str:
    """Get conversion and event data.
    
    Args:
        property_id: Google Analytics 4 property ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    try:
        client = init_ga4_client()
        request_body = {
            'dateRanges': [{'startDate': start_date, 'endDate': end_date}],
            'dimensions': [
                {'name': 'eventName'},
                {'name': 'conversionEventName'}
            ],
            'metrics': [
                {'name': 'eventCount'},
                {'name': 'conversions'},
                {'name': 'conversionRate'},
                {'name': 'totalRevenue'}
            ]
        }
        
        response = client.properties().runReport(
            property=f'properties/{property_id}',
            body=request_body
        ).execute()
        
        result = format_analytics_response(response)
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@mcp.tool()
async def get_custom_report(property_id: str, start_date: str, end_date: str, metrics: List[str], 
                           dimensions: List[str], filters: Optional[Dict] = None,
                           order_bys: Optional[List] = None, limit: Optional[int] = None) -> str:
    """Get custom analytics report with flexible parameters.
    
    Args:
        property_id: Google Analytics 4 property ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        metrics: List of metric names to query
        dimensions: List of dimension names to query
        filters: Optional dimension filters
        order_bys: Optional ordering configuration
        limit: Optional limit on number of rows returned
    """
    try:
        client = init_ga4_client()
        request_body = {
            'dateRanges': [{'startDate': start_date, 'endDate': end_date}],
            'metrics': [{'name': metric} for metric in metrics],
            'dimensions': [{'name': dimension} for dimension in dimensions]
        }
        
        if filters:
            request_body['dimensionFilter'] = filters
        if order_bys:
            request_body['orderBys'] = order_bys
        if limit:
            request_body['limit'] = limit
        
        response = client.properties().runReport(
            property=f'properties/{property_id}',
            body=request_body
        ).execute()
        
        result = format_analytics_response(response)
        return json.dumps(result, indent=2)
        
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
