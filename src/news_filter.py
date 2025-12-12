"""
News Filter Integration Module
Provides hooks for ForexFactory economic calendar API
Avoids trading during high-impact news events
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class NewsFilter:
    """
    News filter to avoid trading during high-impact economic events
    
    Usage:
        filter = NewsFilter()
        if filter.is_safe_to_trade(datetime.now()):
            # Execute trade
    """
    
    def __init__(self, buffer_minutes=30, impact_levels=['high', 'medium']):
        """
        Initialize news filter
        
        Args:
            buffer_minutes: Minutes before/after news to avoid trading (default: 30)
            impact_levels: List of impact levels to filter (default: ['high', 'medium'])
        """
        self.buffer_minutes = buffer_minutes
        self.impact_levels = impact_levels
        self.news_calendar = []
        self.last_update = None
        
    def load_news_calendar(self, filepath=None):
        """
        Load news calendar from file or API
        
        Args:
            filepath: Path to news calendar JSON file
            
        Returns:
            bool: True if loaded successfully
        """
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.news_calendar = data.get('events', [])
                    self.last_update = datetime.fromisoformat(data.get('last_update'))
                    print(f"✓ Loaded {len(self.news_calendar)} news events")
                    return True
            except Exception as e:
                print(f"❌ Error loading news calendar: {e}")
                return False
        else:
            print("⚠️ News calendar file not found. Using empty calendar.")
            print("   To enable news filtering:")
            print("   1. Sign up for ForexFactory API or similar service")
            print("   2. Download economic calendar data")
            print("   3. Save as JSON in data/news_calendar.json")
            return False
    
    def fetch_from_api(self, api_key=None, days_ahead=7):
        """
        Fetch news calendar from API (placeholder for future implementation)
        
        Args:
            api_key: API key for news service
            days_ahead: Number of days to fetch ahead
            
        Returns:
            bool: True if fetched successfully
        """
        # TODO: Implement ForexFactory or similar API integration
        # Example API endpoints:
        # - ForexFactory: https://www.forexfactory.com/calendar
        # - Investing.com Economic Calendar API
        # - Trading Economics API
        
        print("⚠️ API integration not yet implemented")
        print("   This is a placeholder for future ForexFactory integration")
        print()
        print("   Implementation steps:")
        print("   1. Choose news API provider (ForexFactory, Investing.com, etc.)")
        print("   2. Obtain API key")
        print("   3. Implement HTTP requests to fetch calendar data")
        print("   4. Parse response and filter for gold-related events")
        print("   5. Store in self.news_calendar with proper datetime parsing")
        
        return False
    
    def is_safe_to_trade(self, timestamp):
        """
        Check if it's safe to trade at given timestamp
        
        Args:
            timestamp: datetime object to check
            
        Returns:
            bool: True if safe to trade, False if near news event
        
        Warning:
            If no calendar loaded, returns True (trading allowed).
            This is intentional to avoid blocking trades, but means
            news filtering is NOT active. Load calendar to enable.
        """
        if not self.news_calendar:
            # No news calendar loaded - allow trading
            # This is intentional: don't block trades if user hasn't set up news filtering
            return True
        
        for event in self.news_calendar:
            event_time = datetime.fromisoformat(event['datetime'])
            
            # Check if event impact matches filter criteria
            if event.get('impact', '').lower() not in self.impact_levels:
                continue
            
            # Check if within buffer window
            time_diff = abs((timestamp - event_time).total_seconds() / 60)
            
            if time_diff <= self.buffer_minutes:
                return False
        
        return True
    
    def get_next_news_event(self, timestamp):
        """
        Get next high-impact news event after timestamp
        
        Args:
            timestamp: datetime object
            
        Returns:
            dict: Next news event or None
        """
        upcoming = [
            event for event in self.news_calendar
            if datetime.fromisoformat(event['datetime']) > timestamp
            and event.get('impact', '').lower() in self.impact_levels
        ]
        
        if upcoming:
            upcoming.sort(key=lambda x: datetime.fromisoformat(x['datetime']))
            return upcoming[0]
        
        return None
    
    def filter_trade_signals(self, df, time_column='time'):
        """
        Filter DataFrame to remove signals near news events
        
        Args:
            df: DataFrame with trade signals
            time_column: Name of datetime column
            
        Returns:
            DataFrame: Filtered signals
        """
        if not self.news_calendar:
            print("⚠️ No news calendar loaded - returning unfiltered signals")
            return df
        
        df_filtered = df.copy()
        df_filtered['safe_to_trade'] = df_filtered[time_column].apply(self.is_safe_to_trade)
        
        filtered_out = (~df_filtered['safe_to_trade']).sum()
        print(f"   News filter removed {filtered_out} signals ({filtered_out/len(df)*100:.1f}%)")
        
        return df_filtered[df_filtered['safe_to_trade']].drop('safe_to_trade', axis=1)


# Example news calendar format
EXAMPLE_NEWS_CALENDAR = {
    "last_update": datetime.now().isoformat(),
    "events": [
        {
            "datetime": "2025-12-15T14:30:00",
            "title": "US FOMC Interest Rate Decision",
            "currency": "USD",
            "impact": "high",
            "previous": "5.25%",
            "forecast": "5.50%",
            "actual": None
        },
        {
            "datetime": "2025-12-15T15:00:00",
            "title": "FOMC Press Conference",
            "currency": "USD",
            "impact": "high",
            "previous": None,
            "forecast": None,
            "actual": None
        },
        {
            "datetime": "2025-12-16T13:30:00",
            "title": "US Initial Jobless Claims",
            "currency": "USD",
            "impact": "medium",
            "previous": "220K",
            "forecast": "225K",
            "actual": None
        }
    ]
}


def create_example_calendar():
    """Create example news calendar file for testing"""
    os.makedirs('data/news', exist_ok=True)
    
    with open('data/news/example_calendar.json', 'w') as f:
        json.dump(EXAMPLE_NEWS_CALENDAR, f, indent=2)
    
    print("✓ Created example news calendar at: data/news/example_calendar.json")
    print("  Modify this file to add your own news events")


if __name__ == '__main__':
    print("=" * 70)
    print("NEWS FILTER MODULE - DEMO")
    print("=" * 70)
    print()
    
    # Create example calendar
    create_example_calendar()
    print()
    
    # Initialize filter
    filter = NewsFilter(buffer_minutes=30, impact_levels=['high'])
    
    # Try to load calendar
    filter.load_news_calendar('data/news/example_calendar.json')
    print()
    
    # Test with example timestamps
    print("Testing news filter with example timestamps:")
    print()
    
    test_times = [
        datetime(2025, 12, 15, 13, 30),  # 1 hour before FOMC
        datetime(2025, 12, 15, 14, 15),  # 15 min before FOMC
        datetime(2025, 12, 15, 14, 30),  # During FOMC
        datetime(2025, 12, 15, 15, 30),  # 1 hour after FOMC
        datetime(2025, 12, 16, 10, 0),   # Safe time
    ]
    
    for ts in test_times:
        safe = filter.is_safe_to_trade(ts)
        status = "✅ SAFE" if safe else "❌ BLOCKED"
        print(f"   {ts.strftime('%Y-%m-%d %H:%M')} - {status}")
    
    print()
    print("=" * 70)
    print("INTEGRATION INSTRUCTIONS")
    print("=" * 70)
    print()
    print("To integrate news filtering into your trading bot:")
    print()
    print("1. In backtest_simple.py or EA:")
    print("   from news_filter import NewsFilter")
    print("   news_filter = NewsFilter(buffer_minutes=30)")
    print("   news_filter.load_news_calendar('data/news/calendar.json')")
    print()
    print("2. Before executing trade:")
    print("   if news_filter.is_safe_to_trade(current_time):")
    print("       # Execute trade")
    print()
    print("3. Or filter entire signal DataFrame:")
    print("   df_signals = news_filter.filter_trade_signals(df_signals)")
    print()
    print("4. To get live news data:")
    print("   - Use ForexFactory API (https://www.forexfactory.com/calendar)")
    print("   - Or scrape from Investing.com economic calendar")
    print("   - Or use Trading Economics API")
    print("   - Update calendar.json daily via cron job")
    print()
