import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import talib
import math
import os
import requests
from datetime import datetime, timezone, timedelta
import configparser
import json
import traceback
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import threading
import queue
import sys
import random

# ======================================================
# CONFIGURATION AND CONSTANTS
# ======================================================

# Create config parser
config = configparser.ConfigParser()

# Default configuration
DEFAULT_CONFIG = {
    "General": {
        # "symbols": "BTCUSD,ETHUSD,EURUSD,GBPUSD,USDJPY",
        "symbols": "BTCUSD",
        "default_symbol": "BTCUSD",
        "risk_percentage": "0.5",
        "max_open_positions": "5",
        "min_balance": "10000",
        "log_level": "INFO"
    },
    "TimeFrames": {
        "primary_timeframe": "M5",
        "secondary_timeframes": "M15,H1,H4",
        "trend_timeframe": "H1"
    },
    "TechnicalIndicators": {
        "atr_period": "14",
        "rsi_period": "14",
        "bb_period": "20",
        "bb_std_dev": "2",
        "ema_short_period": "50",
        "ema_long_period": "200",
        "adx_period": "14",
        "adx_threshold": "20",
        "macd_fast_period": "12",
        "macd_slow_period": "26",
        "macd_signal_period": "9",
        "vwma_period": "14"
    },
    "RiskManagement": {
        "max_drawdown_per_trade": "0.02",
        "max_drawdown_per_day": "0.05",
        "max_risk_per_trade": "0.01",
        "min_risk_reward_ratio": "1.3"
    },
    "TrailingStop": {
        "tsl_activation_threshold": "0.003",
        "tsl_step_percent": "0.0015",
        "tsl_profit_lock_levels": "0.01,0.02,0.03,0.05,0.08,0.13,0.21",
        "tsl_profit_lock_percents": "0.5,0.6,0.65,0.7,0.75,0.8,0.9",
        "remove_take_profit_after_activation": "False"  # New parameter
    },
    "ExternalData": {
        "news_api_key": "08b5ee4c929d47c0aedbd5426084a7c2",
        "news_check_timeframe": "24",
        "use_news_filter": "False"
    },
    "Account": {
        "account_number": "Add_Account_Number",
        "password": "Add_Password",
        "server": "Exness-MT5Trial8",
        "use_env_vars": "True"
    },
    "VolatilityThresholds": {
        "low_volatility_threshold": "0.0020",
        "normal_volatility_threshold": "0.0050",
        "high_volatility_threshold": "0.0100",
        "extreme_volatility_threshold": "0.0200"
    },
    "Performance": {
        "update_interval": "15",
        "tsl_update_interval": "1",
        "max_retries": "3",
        "retry_delay": "5",
        "min_balance": "1000",
    },
    "AssetSpecific": {
        "crypto_movement_threshold": "1.5",
        "forex_movement_threshold": "0.5",
        "indices_movement_threshold": "0.8"
    }
}

# Define options that should always be updated from DEFAULT_CONFIG
force_update_options = [
    ('General', 'symbols')  # Always update symbols from DEFAULT_CONFIG
]

# Read existing config first (if it exists)
if os.path.exists('config.ini'):
    config.read('config.ini')

# Ensure all sections and options from DEFAULT_CONFIG exist in config
for section, options in DEFAULT_CONFIG.items():
    # Add section if it doesn't exist
    if not config.has_section(section):
        config.add_section(section)
    
    # Update all options in this section
    for option, value in options.items():
        # Always update forced options or add missing options
        if (section, option) in force_update_options or not config.has_option(section, option):
            config.set(section, option, value)

# Write the updated config
with open('config.ini', 'w') as f:
    config.write(f)

# Load configuration settings
def load_config():
    global SYMBOLS, DEFAULT_SYMBOL, RISK_PERCENTAGE, MAX_OPEN_POSITIONS, MIN_BALANCE, LOG_LEVEL
    global PRIMARY_TIMEFRAME, SECONDARY_TIMEFRAMES, TREND_TIMEFRAME
    global ATR_PERIOD, RSI_PERIOD, BB_PERIOD, BB_STD_DEV, EMA_SHORT_PERIOD, EMA_LONG_PERIOD
    global ADX_PERIOD, ADX_THRESHOLD, MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD, VWMA_PERIOD
    global MAX_DRAWDOWN_PER_TRADE, MAX_DRAWDOWN_PER_DAY, MAX_RISK_PER_TRADE, MIN_RISK_REWARD_RATIO
    global TSL_ACTIVATION_THRESHOLD, TSL_STEP_PERCENT, TSL_PROFIT_LOCK_LEVELS, TSL_PROFIT_LOCK_PERCENTS
    global NEWS_API_KEY, NEWS_CHECK_TIMEFRAME, USE_NEWS_FILTER
    global ACCOUNT_NUMBER, PASSWORD, SERVER, USE_ENV_VARS
    global LOW_VOLATILITY_THRESHOLD, NORMAL_VOLATILITY_THRESHOLD, HIGH_VOLATILITY_THRESHOLD, EXTREME_VOLATILITY_THRESHOLD
    global UPDATE_INTERVAL, TSL_UPDATE_INTERVAL, MAX_RETRIES, RETRY_DELAY
    global CRYPTO_MOVEMENT_THRESHOLD, FOREX_MOVEMENT_THRESHOLD, INDICES_MOVEMENT_THRESHOLD
    
    # General settings
    SYMBOLS = config.get('General', 'symbols').split(',')
    DEFAULT_SYMBOL = config.get('General', 'default_symbol')
    RISK_PERCENTAGE = config.getfloat('General', 'risk_percentage')
    MAX_OPEN_POSITIONS = config.getint('General', 'max_open_positions')
    MIN_BALANCE = config.getfloat('General', 'min_balance')
    LOG_LEVEL = config.get('General', 'log_level')
    
    # TimeFrames
    PRIMARY_TIMEFRAME = config.get('TimeFrames', 'primary_timeframe')
    SECONDARY_TIMEFRAMES = config.get('TimeFrames', 'secondary_timeframes').split(',')
    TREND_TIMEFRAME = config.get('TimeFrames', 'trend_timeframe')
    
    # Technical Indicators
    ATR_PERIOD = config.getint('TechnicalIndicators', 'atr_period')
    RSI_PERIOD = config.getint('TechnicalIndicators', 'rsi_period')
    BB_PERIOD = config.getint('TechnicalIndicators', 'bb_period')
    BB_STD_DEV = config.getfloat('TechnicalIndicators', 'bb_std_dev')
    EMA_SHORT_PERIOD = config.getint('TechnicalIndicators', 'ema_short_period')
    EMA_LONG_PERIOD = config.getint('TechnicalIndicators', 'ema_long_period')
    ADX_PERIOD = config.getint('TechnicalIndicators', 'adx_period')
    ADX_THRESHOLD = config.getfloat('TechnicalIndicators', 'adx_threshold')
    MACD_FAST_PERIOD = config.getint('TechnicalIndicators', 'macd_fast_period')
    MACD_SLOW_PERIOD = config.getint('TechnicalIndicators', 'macd_slow_period')
    MACD_SIGNAL_PERIOD = config.getint('TechnicalIndicators', 'macd_signal_period')
    VWMA_PERIOD = config.getint('TechnicalIndicators', 'vwma_period')
    
    # Risk Management
    MAX_DRAWDOWN_PER_TRADE = config.getfloat('RiskManagement', 'max_drawdown_per_trade')
    MAX_DRAWDOWN_PER_DAY = config.getfloat('RiskManagement', 'max_drawdown_per_day')
    MAX_RISK_PER_TRADE = config.getfloat('RiskManagement', 'max_risk_per_trade')
    MIN_RISK_REWARD_RATIO = config.getfloat('RiskManagement', 'min_risk_reward_ratio')
    
    # Trailing Stop
    TSL_ACTIVATION_THRESHOLD = config.getfloat('TrailingStop', 'tsl_activation_threshold')
    TSL_STEP_PERCENT = config.getfloat('TrailingStop', 'tsl_step_percent')
    TSL_PROFIT_LOCK_LEVELS = [float(x) for x in config.get('TrailingStop', 'tsl_profit_lock_levels').split(',')]
    TSL_PROFIT_LOCK_PERCENTS = [float(x) for x in config.get('TrailingStop', 'tsl_profit_lock_percents').split(',')]
    
    # External Data
    NEWS_API_KEY = config.get('ExternalData', 'news_api_key')
    NEWS_CHECK_TIMEFRAME = config.getint('ExternalData', 'news_check_timeframe')
    USE_NEWS_FILTER = config.getboolean('ExternalData', 'use_news_filter')
    
    # Account
    ACCOUNT_NUMBER = config.get('Account', 'account_number')
    PASSWORD = config.get('Account', 'password')
    SERVER = config.get('Account', 'server')
    USE_ENV_VARS = config.getboolean('Account', 'use_env_vars')
    
    # Volatility Thresholds
    LOW_VOLATILITY_THRESHOLD = config.getfloat('VolatilityThresholds', 'low_volatility_threshold')
    NORMAL_VOLATILITY_THRESHOLD = config.getfloat('VolatilityThresholds', 'normal_volatility_threshold')
    HIGH_VOLATILITY_THRESHOLD = config.getfloat('VolatilityThresholds', 'high_volatility_threshold')
    EXTREME_VOLATILITY_THRESHOLD = config.getfloat('VolatilityThresholds', 'extreme_volatility_threshold')
    
    # Performance
    UPDATE_INTERVAL = config.getfloat('Performance', 'update_interval')
    TSL_UPDATE_INTERVAL = config.getfloat('Performance', 'tsl_update_interval')
    MAX_RETRIES = config.getint('Performance', 'max_retries')
    RETRY_DELAY = config.getint('Performance', 'retry_delay')

    # Asset Specific
    CRYPTO_MOVEMENT_THRESHOLD = config.getfloat('AssetSpecific', 'crypto_movement_threshold')
    FOREX_MOVEMENT_THRESHOLD = config.getfloat('AssetSpecific', 'forex_movement_threshold')
    INDICES_MOVEMENT_THRESHOLD = config.getfloat('AssetSpecific', 'indices_movement_threshold')

    # Override with environment variables if specified
    if USE_ENV_VARS:
        if os.getenv("MT5_ACCOUNT_NUMBER"):
            ACCOUNT_NUMBER = os.getenv("MT5_ACCOUNT_NUMBER")
        if os.getenv("MT5_PASSWORD"):
            PASSWORD = os.getenv("MT5_PASSWORD")
        if os.getenv("MT5_SERVER"):
            SERVER = os.getenv("MT5_SERVER")
        if os.getenv("NEWS_API_KEY"):
            NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Load configuration
load_config()

# Convert timeframe strings to MT5 constants
def get_mt5_timeframe(timeframe_str):
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1
    }
    return timeframe_map.get(timeframe_str, mt5.TIMEFRAME_M5)

# Mapping of timeframe strings to MT5 constants
MT5_TIMEFRAMES = {
    "PRIMARY": get_mt5_timeframe(PRIMARY_TIMEFRAME),
    "TREND": get_mt5_timeframe(TREND_TIMEFRAME)
}

for i, tf in enumerate(SECONDARY_TIMEFRAMES):
    MT5_TIMEFRAMES[f"SECONDARY_{i}"] = get_mt5_timeframe(tf)

# Paths for files
TRADE_NOTES_FILE = "trade_notes.txt"
TRADE_STATS_FILE = "trade_stats.json"
LOG_FILE = "trading_bot.log"

# ======================================================
# LOGGING SETUP
# ======================================================

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_cached_volatility(symbol, timeframe, max_age_seconds=60):
    """
    Get volatility level with caching to avoid redundant calculations.
    """
    current_time = time.time()
    cache_key = f"{symbol}_{timeframe}"
    
    # Check cache
    if cache_key in state.volatility_cache:
        cache_entry = state.volatility_cache[cache_key]
        # Use cached value if not expired
        if current_time - cache_entry['timestamp'] < max_age_seconds:
            return cache_entry['level']
    
    # Calculate if not in cache or expired
    volatility_level, atr_percent = calculate_market_volatility(symbol, timeframe)
    
    # Update cache
    state.volatility_cache[cache_key] = {
        'level': volatility_level,
        'atr_percent': atr_percent,
        'timestamp': current_time
    }
    
    return volatility_level

def throttled_log(key, message, level='info', min_interval=60):
    """
    Log messages with throttling to reduce log spam.
    """
    current_time = time.time()
    
    # Initialize throttle tracking if needed
    if key not in state.log_throttle:
        state.log_throttle[key] = 0
    
    # Check if enough time has passed since last log
    if current_time - state.log_throttle[key] >= min_interval:
        # Log message based on level
        if level == 'debug':
            logging.debug(message)
        elif level == 'info':
            logging.info(message)
        elif level == 'warning':
            logging.warning(message)
        elif level == 'error':
            logging.error(message)
        
        # Update last log time
        state.log_throttle[key] = current_time
        return True
    
    return False


# Initialize global variables
class State:
    """Class to store global state variables"""
    volume_warning_logged = set()  # Track which symbols have logged volume warnings
    symbols_data = {}  # Store data for each symbol
    trade_stats = {}   # Trade statistics
    initial_balance = 0.0  # Initial account balance
    daily_loss = 0.0   # Track daily loss
    profit_streak = 0  # Track consecutive profitable trades
    last_trade_time = {} # Last trade time for each symbol
    exit_requested = False  # Flag to request graceful shutdown
    market_state = {}  # Store market state for each symbol (ranging, trending, volatile)
    trade_queue = queue.Queue()  # Queue for trade operations
    data_cache = {}  # Cache for indicator data
    current_signals = {}  # Current trade signals for each symbol
    custom_symbols = {}  # Custom symbols list support (if needed)
    performance_stats = {}  # Performance statistics    
    risk_modifier = 1.0  # Global risk modifier
    max_drawdown_seen = 0.0  # Track maximum drawdown
    last_risk_update = 0  # Last time risk was updated
    volatility_cache = {}  # Cache for volatility calculations
    log_throttle = {}

# Initialize state
state = State()

# ======================================================
# UTILITY FUNCTIONS
# ======================================================

def ensure_required_packages():
    """Check and install required packages."""
    try:
        # Try importing scikit-learn
        try:
            import sklearn
            logging.info("scikit-learn is already installed.")
        except ImportError:
            logging.warning("scikit-learn is not installed. Attempting to install...")
            import sys
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
            logging.info("scikit-learn successfully installed.")
        
        # You can add more package checks here if needed
        
    except Exception as e:
        logging.error(f"Error checking/installing required packages: {e}")
        logging.error("Please manually install required packages: pip install scikit-learn")

def save_trade_stats():
    """Save trade statistics to a JSON file"""
    with open(TRADE_STATS_FILE, 'w') as f:
        json.dump(state.trade_stats, f, indent=4)

def load_trade_stats():
    """Load trade statistics from a JSON file"""
    if os.path.exists(TRADE_STATS_FILE):
        try:
            with open(TRADE_STATS_FILE, 'r') as f:
                state.trade_stats = json.load(f)
            logging.info(f"Loaded trade stats: {state.trade_stats}")
        except Exception as e:
            logging.error(f"Error loading trade stats: {e}")
            state.trade_stats = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "breakeven_trades": 0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "last_trades": []
            }
    else:
        state.trade_stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "last_trades": []
        }

def write_trade_notes(message):
    """Write trade-related notes to a file for record-keeping"""
    try:
        with open(TRADE_NOTES_FILE, "a") as file:
            file.write(f"{datetime.now()} - {message}\n")
        logging.info(f"Trade note written to file: {message}")
    except Exception as e:
        logging.error(f"Failed to write trade notes to file: {e}")

def update_trade_stats(trade_result):
    """Update trade statistics based on trade result"""
    try:
        with state.locks['trade_stats']:
            state.trade_stats["total_trades"] += 1
            
            if trade_result["profit"] > 0:
                state.trade_stats["winning_trades"] += 1
                state.trade_stats["total_profit"] += trade_result["profit"]
                state.trade_stats["largest_win"] = max(state.trade_stats["largest_win"], trade_result["profit"])
                state.trade_stats["average_win"] = state.trade_stats["total_profit"] / state.trade_stats["winning_trades"]
            elif trade_result["profit"] < 0:
                state.trade_stats["losing_trades"] += 1
                state.trade_stats["total_loss"] += abs(trade_result["profit"])
                state.trade_stats["largest_loss"] = max(state.trade_stats["largest_loss"], abs(trade_result["profit"]))
                state.trade_stats["average_loss"] = state.trade_stats["total_loss"] / state.trade_stats["losing_trades"]
            else:
                state.trade_stats["breakeven_trades"] += 1
            
            if state.trade_stats["total_trades"] > 0:
                state.trade_stats["win_rate"] = state.trade_stats["winning_trades"] / state.trade_stats["total_trades"] * 100
            
            if state.trade_stats["total_loss"] > 0:
                state.trade_stats["profit_factor"] = state.trade_stats["total_profit"] / state.trade_stats["total_loss"]
            
            # Add to last trades list (keep most recent 50)
            state.trade_stats["last_trades"].append({
                "symbol": trade_result["symbol"],
                "type": trade_result["type"],
                "entry_price": trade_result["entry_price"],
                "exit_price": trade_result["exit_price"],
                "profit": trade_result["profit"],
                "profit_pips": trade_result["profit_pips"],
                "entry_time": trade_result["entry_time"].isoformat() if isinstance(trade_result["entry_time"], datetime) else trade_result["entry_time"],
                "exit_time": trade_result["exit_time"].isoformat() if isinstance(trade_result["exit_time"], datetime) else trade_result["exit_time"],
                "reason": trade_result["reason"]
            })
            
            if len(state.trade_stats["last_trades"]) > 50:
                state.trade_stats["last_trades"] = state.trade_stats["last_trades"][-50:]
            
            # Save updated stats
            save_trade_stats()
            
            # Log trade result
            logging.info(f"Trade completed - Symbol: {trade_result['symbol']}, "
                        f"Type: {trade_result['type']}, "
                        f"Profit: {trade_result['profit']:.2f}, "
                        f"Reason: {trade_result['reason']}")
    except Exception as e:
        logging.error(f"Error updating trade stats: {e}")

def generate_performance_report():
    """Generate a performance report as a string"""
    report = "======= TRADING PERFORMANCE REPORT =======\n\n"
    report += f"Total Trades: {state.trade_stats['total_trades']}\n"
    report += f"Winning Trades: {state.trade_stats['winning_trades']} ({state.trade_stats['win_rate']:.2f}%)\n"
    report += f"Losing Trades: {state.trade_stats['losing_trades']}\n"
    report += f"Breakeven Trades: {state.trade_stats['breakeven_trades']}\n\n"
    
    report += f"Total Profit: {state.trade_stats['total_profit']:.2f}\n"
    report += f"Total Loss: {state.trade_stats['total_loss']:.2f}\n"
    report += f"Net Profit: {state.trade_stats['total_profit'] - state.trade_stats['total_loss']:.2f}\n\n"
    
    report += f"Largest Win: {state.trade_stats['largest_win']:.2f}\n"
    report += f"Largest Loss: {state.trade_stats['largest_loss']:.2f}\n"
    report += f"Average Win: {state.trade_stats['average_win']:.2f}\n"
    report += f"Average Loss: {state.trade_stats['average_loss']:.2f}\n\n"
    
    report += f"Profit Factor: {state.trade_stats['profit_factor']:.2f}\n"
    
    report += "\nRecent Trades:\n"
    for i, trade in enumerate(reversed(state.trade_stats['last_trades'][:10])):
        report += f"{i+1}. {trade['symbol']} {trade['type']} - Profit: {trade['profit']:.2f} - Reason: {trade['reason']}\n"
    
    return report

def create_performance_chart():
    """Create a performance chart as a base64 encoded image"""
    try:
        if len(state.trade_stats['last_trades']) < 2:
            return None
        
        # Extract profit data
        profits = [trade['profit'] for trade in state.trade_stats['last_trades']]
        cumulative_profits = np.cumsum(profits)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot cumulative profit
        plt.subplot(2, 1, 1)
        plt.plot(cumulative_profits, 'b-', linewidth=2)
        plt.title('Cumulative Profit')
        plt.grid(True)
        
        # Plot individual trade profits
        plt.subplot(2, 1, 2)
        colors = ['g' if profit >= 0 else 'r' for profit in profits]
        plt.bar(range(len(profits)), profits, color=colors)
        plt.title('Individual Trade Profits')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save to bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str
    except Exception as e:
        logging.error(f"Error creating performance chart: {e}")
        return None

# ======================================================
# NEW HELPER FUNCTIONS FOR IMPLEMENTATION
# ======================================================

def get_optimal_entry_price(symbol, signal):
    """
    Calculate an optimal entry price based on current market conditions.
    More opportunistic in finding good entries without missing trades.
    """
    try:
        # Get current prices
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
            
        current_bid = tick.bid
        current_ask = tick.ask
        
        # Default to current market price
        if signal == "BUY":
            entry_price = current_ask
        else:  # "SELL"
            entry_price = current_bid
            
        # Check if we should optimize entry based on market conditions
        market_regime, regime_strength = detect_market_regime(symbol)
        
        # In strong trends, enter at market to avoid missing the move
        if market_regime in ["STRONG_UPTREND", "STRONG_DOWNTREND"] and regime_strength > 0.7:
            # Just use market price to avoid missing the move
            return entry_price
            
        # Get recent price data
        df = get_candle_data(symbol, MT5_TIMEFRAMES["PRIMARY"], num_bars=50)
        if df is None:
            return entry_price
            
        # Calculate ATR for reference
        atr = calculate_atr(df)
        
        # For range-bound markets, try to optimize entry
        if market_regime == "RANGE_BOUND":
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower, _ = calculate_bollinger_bands(df)
            
            # For buy signals, see if we can get a better entry near the middle band
            if signal == "BUY" and current_ask > bb_middle:
                better_entry = current_ask - ((current_ask - bb_middle) * 0.3)
                
                # Only use if reasonably close
                if better_entry / current_ask > 0.997:  # Within 0.3%
                    entry_price = better_entry
            
            # For sell signals, see if we can get a better entry near the middle band
            elif signal == "SELL" and current_bid < bb_middle:
                better_entry = current_bid + ((bb_middle - current_bid) * 0.3)
                
                # Only use if reasonably close
                if better_entry / current_bid < 1.003:  # Within 0.3%
                    entry_price = better_entry
        
        # Get symbol info for price rounding
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return entry_price
            
        # Round to symbol precision
        digits = symbol_info.digits
        entry_price = round(entry_price, digits)
        
        return entry_price
        
    except Exception as e:
        logging.error(f"Error calculating optimal entry price: {e}")
        
        # Fall back to current market price
        if signal == "BUY":
            return mt5.symbol_info_tick(symbol).ask
        else:  # "SELL"
            return mt5.symbol_info_tick(symbol).bid

def calculate_advanced_stop_loss_take_profit(symbol, order_type, entry_price, market_state=None):
    """
    Calculate advanced stop loss and take profit levels with key level integration.
    Uses market structure and volatility to place optimal levels.
    Much more aggressive with take profit targets in favorable conditions.
    
    ENHANCED: Better handling for Bitcoin in super-extreme volatility
    """
    try:
        # Get market structure if not provided
        if market_state is None:
            market_state = analyze_market_structure(symbol)
            
        # Get key levels from market structure
        key_levels = market_state.get("key_levels", [])
        
        # Get volatility state
        volatility_level = market_state.get("volatility", "normal")
        
        # Get optimized timeframe based on volatility
        optimized_tf = optimize_timeframe_for_volatility(symbol)
        
        # Get data for ATR calculation
        df = get_candle_data(symbol, optimized_tf)
        if df is None:
            return None, None
            
        # Calculate ATR
        atr = calculate_atr(df)
        
        # Get symbol info
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return None, None
            
        point = symbol_info.point
        min_stop_level_points = symbol_info.trade_stops_level
        
        # Check for chart pattern targets
        pattern_target = None
        if hasattr(state, 'current_signals') and symbol in state.current_signals:
            signal_data = state.current_signals[symbol]
            if "chart_patterns" in signal_data:
                pattern_data = signal_data["chart_patterns"]
                if pattern_data.get("detected", False):
                    pattern_target = pattern_data.get("target")
        
        # Get market regime for target adjustments
        market_regime, regime_strength = detect_market_regime(symbol)
        
        # ENHANCED: Better handling for cryptocurrency in volatile conditions
        is_crypto = symbol.startswith("BTC") or symbol.startswith("ETH") or "BTC" in symbol or "ETH" in symbol
        
        if is_crypto:
            is_bitcoin = symbol.startswith("BTC") or "BTC" in symbol
            
            # Set base percentages - ENHANCED FOR BITCOIN IN SUPER-EXTREME VOLATILITY
            if is_bitcoin:
                if volatility_level == "super-extreme":
                    # ENHANCED: Better SL/TP for super-extreme volatility
                    sl_percent = 0.022  # 2.2% stop loss - reduced from 2.5% for tighter protection
                    tp_percent = 1.2  # 6% take profit target - increased from 5% for better rewards
                elif volatility_level == "extreme":
                    sl_percent = 0.018  # 1.8% stop loss
                    tp_percent = 0.045  # 4.5% take profit - increased from 4%
                elif volatility_level == "high":
                    sl_percent = 0.014  # 1.4% stop loss
                    tp_percent = 0.038  # 3.8% take profit - increased from 3.5%
                else:  # normal or low
                    sl_percent = 0.012  # 1.2% stop loss
                    tp_percent = 0.032  # 3.2% take profit
            else:  # Other crypto
                if volatility_level == "extreme":
                    sl_percent = 0.015  # 1.5% stop loss
                    tp_percent = 0.035  # 3.5% take profit
                elif volatility_level == "high":
                    sl_percent = 0.012  # 1.2% stop loss
                    tp_percent = 0.03  # 3% take profit
                else:  # normal or low
                    sl_percent = 0.01  # 1% stop loss
                    tp_percent = 0.025  # 2.5% take profit
            
            # ENHANCED: Better market regime adjustments for range-bound markets
            if market_regime == "RANGE_BOUND" and regime_strength > 0.65:
                # In strong range-bound markets, adjust for mean reversion
                if order_type == "BUY":
                    # For buys near lower range bound, reduce SL and increase TP
                    current_price = df['close'].iloc[-1]
                    range_low = min(df['low'].iloc[-30:])
                    range_high = max(df['high'].iloc[-30:])
                    range_position = (current_price - range_low) / (range_high - range_low)
                    
                    if range_position < 0.3:  # Near bottom of range
                        sl_percent *= 0.9  # Tighter stop
                        tp_percent *= 1.2  # Higher target
                elif order_type == "SELL":
                    # For sells near upper range bound, reduce SL and increase TP
                    current_price = df['close'].iloc[-1]
                    range_low = min(df['low'].iloc[-30:])
                    range_high = max(df['high'].iloc[-30:])
                    range_position = (current_price - range_low) / (range_high - range_low)
                    
                    if range_position > 0.7:  # Near top of range
                        sl_percent *= 0.9  # Tighter stop
                        tp_percent *= 1.2  # Higher target
            
            # Old code for strong trends can stay as is
            elif market_regime in ["STRONG_UPTREND", "STRONG_DOWNTREND"] and regime_strength > 0.7:
                if (market_regime == "STRONG_UPTREND" and order_type == "BUY") or \
                   (market_regime == "STRONG_DOWNTREND" and order_type == "SELL"):
                    # In aligned trend, be much more ambitious with take profit
                    tp_percent *= 2.0  # DOUBLE the take profit target
                    logging.info(f"Setting extremely ambitious profit target (doubled) due to strong {market_regime}")
            
            # Calculate stop loss and take profit based on percentages
            if order_type == "BUY":
                stop_loss = entry_price * (1 - sl_percent)
                take_profit = entry_price * (1 + tp_percent)
            else:  # "SELL"
                stop_loss = entry_price * (1 + sl_percent)
                take_profit = entry_price * (1 - tp_percent)
                
            # Use pattern target if available and better than calculated target
            if pattern_target is not None:
                if order_type == "BUY" and pattern_target > take_profit:
                    take_profit = pattern_target
                    logging.info(f"Using pattern target for take profit: {pattern_target}")
                elif order_type == "SELL" and pattern_target < take_profit:
                    take_profit = pattern_target
                    logging.info(f"Using pattern target for take profit: {pattern_target}")
        
        else:  # Non-crypto assets
            # Set ATR multipliers based on volatility - MORE AGGRESSIVE
            if volatility_level == "low":
                sl_multiplier = 1.5
                tp_multiplier = 3.0
            elif volatility_level == "normal":
                sl_multiplier = 1.8
                tp_multiplier = 3.5
            elif volatility_level == "high":
                sl_multiplier = 2.0
                tp_multiplier = 4.0
            else:  # "extreme" or "super-extreme"
                sl_multiplier = 2.2
                tp_multiplier = 4.5
                
            # Set more ambitious targets in strong trends
            if market_regime in ["STRONG_UPTREND", "STRONG_DOWNTREND"] and regime_strength > 0.7:
                if (market_regime == "STRONG_UPTREND" and order_type == "BUY") or \
                   (market_regime == "STRONG_DOWNTREND" and order_type == "SELL"):
                    # In aligned strong trends, be extremely ambitious with targets
                    tp_multiplier *= 2.0  # DOUBLE the take profit target
                    logging.info(f"Setting extremely ambitious profit target (doubled) due to strong {market_regime}")
            
            # Calculate stop loss and take profit based on ATR
            if order_type == "BUY":
                stop_loss = entry_price - (atr * sl_multiplier)
                take_profit = entry_price + (atr * tp_multiplier)
            else:  # "SELL"
                stop_loss = entry_price + (atr * sl_multiplier)
                take_profit = entry_price - (atr * tp_multiplier)
                
            # Use pattern target if available and better than calculated target
            if pattern_target is not None:
                if order_type == "BUY" and pattern_target > take_profit:
                    take_profit = pattern_target
                    logging.info(f"Using pattern target for take profit: {pattern_target}")
                elif order_type == "SELL" and pattern_target < take_profit:
                    take_profit = pattern_target
                    logging.info(f"Using pattern target for take profit: {pattern_target}")
        
        # Round to symbol precision
        digits = symbol_info.digits
        stop_loss = round(stop_loss, digits)
        take_profit = round(take_profit, digits)
        
        # Ensure SL and TP are within broker's minimum distance requirements
        min_distance = min_stop_level_points * point
        
        if order_type == "BUY":
            if entry_price - stop_loss < min_distance:
                stop_loss = entry_price - min_distance
            if take_profit - entry_price < min_distance:
                take_profit = entry_price + min_distance
        else:  # "SELL"
            if stop_loss - entry_price < min_distance:
                stop_loss = entry_price + min_distance
            if entry_price - take_profit < min_distance:
                take_profit = entry_price - min_distance
                
        # ENHANCED: Detailed logging of SL/TP calculation
        logging.info(f"Advanced SL/TP for {symbol} {order_type}: " +
                    f"Entry={entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}, " +
                    f"Volatility={volatility_level}, Regime={market_regime}")
                
        return stop_loss, take_profit
        
    except Exception as e:
        logging.error(f"Error calculating advanced SL/TP: {e}")
        return None, None
    
def calculate_currency_conversion_rate(symbol_info, account_info):
    """
    Calculate the conversion rate between the symbol's profit currency and the account currency.
    
    Args:
        symbol_info: MT5 symbol info object
        account_info: MT5 account info object
        
    Returns:
        float: Conversion rate (1.0 if currencies are the same)
    """
    try:
        # Get currencies
        currency_profit = symbol_info.currency_profit
        account_currency = account_info.currency
        
        # If currencies are the same, no conversion needed
        if currency_profit == account_currency:
            return 1.0
            
        # Try direct conversion pair
        conversion_pair = f"{currency_profit}{account_currency}"
        conversion_info = get_symbol_info(conversion_pair)
        
        if conversion_info:
            conversion_rate = mt5.symbol_info_tick(conversion_pair).ask
            return conversion_rate
            
        # Try inverse conversion pair
        inverse_pair = f"{account_currency}{currency_profit}"
        inverse_info = get_symbol_info(inverse_pair)
        
        if inverse_info:
            inverse_rate = mt5.symbol_info_tick(inverse_pair).ask
            if inverse_rate > 0:
                return 1.0 / inverse_rate
                
        # Try USD as intermediate (e.g., for GBPJPY when account is in EUR)
        if currency_profit != "USD" and account_currency != "USD":
            # Try USD/AccountCurrency rate
            usd_account_pair = f"USD{account_currency}"
            usd_info = get_symbol_info(usd_account_pair)
            
            if usd_info:
                usd_account_rate = mt5.symbol_info_tick(usd_account_pair).ask
                
                # Try CurrencyProfit/USD rate
                currency_usd_pair = f"{currency_profit}USD"
                currency_usd_info = get_symbol_info(currency_usd_pair)
                
                if currency_usd_info:
                    currency_usd_rate = mt5.symbol_info_tick(currency_usd_pair).ask
                    return currency_usd_rate * usd_account_rate
                    
                # Try inverse USD/CurrencyProfit rate
                usd_currency_pair = f"USD{currency_profit}"
                usd_currency_info = get_symbol_info(usd_currency_pair)
                
                if usd_currency_info:
                    usd_currency_rate = mt5.symbol_info_tick(usd_currency_pair).ask
                    if usd_currency_rate > 0:
                        return usd_account_rate / usd_currency_rate
                    
        # Default fallback - potentially inaccurate but better than nothing
        logging.warning(f"Could not find conversion rate between {currency_profit} and {account_currency}. Using default 1.0")
        return 1.0
        
    except Exception as e:
        logging.error(f"Error calculating currency conversion rate: {e}")
        return 1.0  # Default fallback

def estimate_profit_potential(symbol, signal, base_signal):
    """
    Estimate the profit potential for a trade based on signal type and market conditions.
    Returns the expected profit factor (reward/risk ratio).
    """
    try:
        # Get market state
        market_regime, regime_strength = detect_market_regime(symbol)
        volatility_level, atr_percent = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
        
        # Get base profit factor from risk-reward ratio
        if "risk_reward" in base_signal and "risk_reward_ratio" in base_signal["risk_reward"]:
            base_profit_factor = base_signal["risk_reward"]["risk_reward_ratio"]
        else:
            # Default risk-reward ratio based on signal type
            if base_signal.get("strategy") == "trend_following":
                base_profit_factor = 1.5
            elif base_signal.get("strategy") == "breakout":
                base_profit_factor = 1.8
            elif base_signal.get("strategy") == "reversal":
                base_profit_factor = 2.0
            else:
                base_profit_factor = 1.5
        
        # Adjust based on market regime - MORE AGGRESSIVE
        regime_adjustment = 1.0
        if signal == "BUY":
            if market_regime == "STRONG_UPTREND":
                regime_adjustment = 1.5  # 50% higher potential in aligned trend
            elif market_regime == "STRONG_DOWNTREND":
                regime_adjustment = 0.7  # 30% lower potential in counter-trend
            # ENHANCED: Better handling for range-bound markets
            elif market_regime == "RANGE_BOUND":
                regime_adjustment = 1.2  # 20% higher potential in ranging markets
        elif signal == "SELL":
            if market_regime == "STRONG_DOWNTREND":
                regime_adjustment = 1.5  # 50% higher potential in aligned trend
            elif market_regime == "STRONG_UPTREND":
                regime_adjustment = 0.7  # 30% lower potential in counter-trend
            # ENHANCED: Better handling for range-bound markets
            elif market_regime == "RANGE_BOUND":
                regime_adjustment = 1.2  # 20% higher potential in ranging markets
                
        # Adjust based on volatility - MORE AGGRESSIVE
        volatility_adjustment = 1.0
        if volatility_level == "high":
            volatility_adjustment = 1.3  # Higher potential in high volatility
        elif volatility_level == "extreme":
            volatility_adjustment = 1.5  # Even higher potential in extreme volatility
        # ENHANCED: Add handling for super-extreme volatility
        elif volatility_level == "super-extreme":
            volatility_adjustment = 1.8  # Much higher potential in super-extreme volatility
        elif volatility_level == "low":
            volatility_adjustment = 0.9  # Lower potential in low volatility
            
        # ENHANCED: Special handling for Bitcoin in super-extreme volatility
        is_bitcoin = symbol.startswith("BTC") or "BTC" in symbol
        if is_bitcoin and volatility_level == "super-extreme":
            # Further boost profit expectation for Bitcoin in super-extreme volatility
            volatility_adjustment *= 1.2  # Additional 20% boost
            
        # Calculate final profit factor
        profit_factor = base_profit_factor * regime_adjustment * volatility_adjustment
        
        # Log detailed estimation for transparency
        logging.info(f"Profit potential estimation for {symbol} {signal}: " +
                    f"Base: {base_profit_factor:.2f}, " +
                    f"Regime adj: {regime_adjustment:.2f}, " +
                    f"Volatility adj: {volatility_adjustment:.2f}, " +
                    f"Final: {profit_factor:.2f}")
        
        return profit_factor
    except Exception as e:
        logging.error(f"Error estimating profit potential: {e}")
        return 1.5  # Default profit factor
    
def get_current_spread(symbol):
    """
    Get the current spread for the symbol in pips.
    """
    try:
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return 0
            
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return 0
            
        spread_points = tick.ask - tick.bid
        
        # Convert to pips
        point = symbol_info.point
        spread_pips = spread_points / point
        
        return spread_pips
    except Exception as e:
        logging.error(f"Error getting current spread: {e}")
        return 0
    
def estimate_transaction_costs(symbol):
    """
    Estimate the transaction costs for trading a symbol.
    Returns cost as a percentage of trade value.
    ENHANCED: More accurate cost estimation for volatile markets
    """
    try:
        # Get spread
        spread_pips = get_current_spread(symbol)
        
        # Get symbol info
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return 0.001  # Default 0.1%
            
        # Convert spread to percentage
        price = mt5.symbol_info_tick(symbol).ask
        point = symbol_info.point
        spread_percent = (spread_pips * point) / price
        
        # Get volatility level for more accurate estimates
        volatility_level, _ = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
        
        # ENHANCED: Adjust slippage estimate based on volatility
        if volatility_level == "super-extreme":
            slippage_percent = 0.00025  # 0.025% for super-extreme volatility
        elif volatility_level == "extreme":
            slippage_percent = 0.0002  # 0.02% for extreme volatility
        elif volatility_level == "high":
            slippage_percent = 0.00015  # 0.015% for high volatility
        else:
            slippage_percent = 0.0001  # 0.01% for normal/low volatility
            
        # ENHANCED: Special handling for Bitcoin
        is_bitcoin = symbol.startswith("BTC") or "BTC" in symbol
        if is_bitcoin:
            # Bitcoin tends to have lower relative slippage due to its higher price
            slippage_percent *= 0.8  # 20% reduction for Bitcoin
        
        # Add estimated commission (typically lower for crypto)
        if is_bitcoin:
            commission_percent = 0.00015  # 0.015% for Bitcoin
        else:
            commission_percent = 0.0002  # 0.02% for other instruments
        
        # Total cost
        total_cost_percent = spread_percent + commission_percent + slippage_percent
        
        # ENHANCED: Add logging to help analyze costs
        logging.info(f"Transaction costs for {symbol}: Total: {total_cost_percent:.6f}%, " +
                    f"Spread: {spread_percent:.6f}%, Commission: {commission_percent:.6f}%, " +
                    f"Slippage: {slippage_percent:.6f}%, Volatility: {volatility_level}")
        
        return total_cost_percent
    except Exception as e:
        logging.error(f"Error estimating transaction costs: {e}")
        return 0.001  # Default 0.1%
    
def detect_market_regime(symbol, timeframes=None):
    """
    Advanced market regime detection using multiple metrics and timeframes.
    Identifies trend strength, volatility, momentum, and market efficiency.
    
    ENHANCED: Better detection of range-bound markets with high volatility
    
    Returns:
        tuple: (regime_type, strength_score)
    """
    try:
        # Add fallback constants if initialization failed
        if not hasattr(state, 'CONSTANTS'):
            # Create default CONSTANTS as fallback
            state.CONSTANTS = {
                'VOLATILITY': {
                    'LOW': 0.0020,
                    'NORMAL': 0.0050,
                    'HIGH': 0.0100,
                    'EXTREME': 0.0200,
                    'SUPER_EXTREME': 0.0300
                },
                'MARKET_REGIME': {
                    'TREND_THRESHOLD': 0.6,
                    'RANGE_THRESHOLD': 0.6,
                    'CHOPPINESS_THRESHOLD': 50
                },
                'RISK_MULTIPLIERS': {
                    'STRONG_TREND': 1.5,
                    'RANGE_BOUND': 0.7,
                    'CHOPPY_VOLATILE': 0.9
                }
            }
            logging.warning("Created fallback CONSTANTS - proper initialization may have failed")
            
        if timeframes is None:
            timeframes = [MT5_TIMEFRAMES["TREND"], MT5_TIMEFRAMES["PRIMARY"]]
        
        # Store results from each timeframe
        regime_scores = {}
        
        for tf in timeframes:
            # Get data
            df = get_candle_data(symbol, tf, num_bars=200)
            if df is None:
                continue
            
            # 1. Calculate trend metrics
            ema_short = calculate_ema(df, EMA_SHORT_PERIOD)
            ema_long = calculate_ema(df, EMA_LONG_PERIOD)
            adx, plus_di, minus_di = calculate_adx(df)
            
            # 2. Calculate volatility
            atr_percent = calculate_atr_percent(df, symbol=symbol)
            bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(df)
            
            # 3. Calculate momentum
            macd_line, signal_line, histogram = calculate_macd(df)
            rsi = calculate_rsi(df)
            
            # 4. Calculate market efficiency ratio (MER)
            # MER = directional movement / total volatility (closer to 1 = more efficient trend)
            close_prices = df['close']
            direction_movement = abs(close_prices.iloc[-1] - close_prices.iloc[-100])
            total_movement = sum(abs(close_prices.diff().fillna(0).iloc[-100:]))
            mer = direction_movement / total_movement if total_movement > 0 else 0
            
            # 5. Calculate swing high/low patterns
            swing_highs, swing_lows = identify_swing_points(df)
            
            # Higher highs & higher lows = uptrend
            # Lower highs & lower lows = downtrend
            higher_highs = is_pattern_higher_highs(swing_highs[-5:] if len(swing_highs) >= 5 else swing_highs)
            higher_lows = is_pattern_higher_lows(swing_lows[-5:] if len(swing_lows) >= 5 else swing_lows)
            lower_highs = is_pattern_lower_highs(swing_highs[-5:] if len(swing_highs) >= 5 else swing_highs)
            lower_lows = is_pattern_lower_lows(swing_lows[-5:] if len(swing_lows) >= 5 else swing_lows)
            
            # 6. Calculate choppiness index
            choppiness = calculate_choppiness_index(df)
            
            # Score each regime type
            trend_score = min(1.0, (adx / 100) * 2)
            range_score = max(0, min(1.0, (60 - adx) / 60))
            volatility_score = min(1.0, atr_percent / 2.0)
            efficiency_score = mer
            momentum_score = abs(rsi - 50) / 50
            
            # ENHANCED: Adjust range score for super-extreme volatility
            # High volatility can mask range-bound behavior
            volatility_level, _ = calculate_market_volatility(symbol, tf)
            if volatility_level == "super-extreme" or volatility_level == "extreme":
                # Check for price containment within a channel despite high volatility
                if bb_width < 0.08:  # Narrow Bollinger Bands relative to price
                    range_score += 0.2  # Boost range detection in volatile conditions
                    logging.info(f"Enhanced range detection for {symbol} in {volatility_level} volatility")
            
            # Determine trend direction
            if plus_di > minus_di and ema_short > ema_long and higher_highs and higher_lows:
                trend_direction = "bullish"
                direction_score = 1.0
            elif minus_di > plus_di and ema_short < ema_long and lower_highs and lower_lows:
                trend_direction = "bearish"
                direction_score = -1.0
            else:
                trend_direction = "neutral"
                direction_score = 0.0
            
            # Store timeframe results
            regime_scores[tf] = {
                "trend_score": trend_score,
                "range_score": range_score,
                "volatility_score": volatility_score,
                "efficiency_score": efficiency_score,
                "direction_score": direction_score,
                "momentum_score": momentum_score,
                "choppiness": choppiness,
                "mer": mer
            }
        
        # Weighted timeframe combination
        weights = {
            MT5_TIMEFRAMES["TREND"]: 0.7,
            MT5_TIMEFRAMES["PRIMARY"]: 0.3
        }
        
        # Calculate weighted scores
        combined_trend_score = 0
        combined_range_score = 0
        combined_volatility_score = 0
        combined_direction_score = 0
        combined_choppiness = 0
        total_weight = 0
        
        for tf, scores in regime_scores.items():
            weight = weights.get(tf, 0.1)
            combined_trend_score += scores["trend_score"] * weight
            combined_range_score += scores["range_score"] * weight
            combined_volatility_score += scores["volatility_score"] * weight
            combined_direction_score += scores["direction_score"] * weight
            combined_choppiness += scores["choppiness"] * weight
            total_weight += weight
        
        if total_weight > 0:
            combined_trend_score /= total_weight
            combined_range_score /= total_weight
            combined_volatility_score /= total_weight
            combined_direction_score /= total_weight
            combined_choppiness /= total_weight
        
        # Use thresholds from centralized constants
        regime_thresholds = state.CONSTANTS['MARKET_REGIME']
        
        # ENHANCED: Special adjustment for volatile range-bound condition
        is_bitcoin = symbol.startswith("BTC") or "BTC" in symbol
        volatility_level, _ = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
        range_threshold_adj = regime_thresholds['RANGE_THRESHOLD']
        
        # For Bitcoin in high volatility, make it easier to detect range-bound conditions
        if is_bitcoin and (volatility_level == "super-extreme" or volatility_level == "extreme"):
            range_threshold_adj *= 0.85  # 15% lower threshold for Bitcoin in extreme volatility
            logging.info(f"Adjusting range detection threshold for {symbol} in {volatility_level} volatility")

        # Classification logic
        if combined_trend_score > regime_thresholds['TREND_THRESHOLD'] and combined_range_score < 0.4 and combined_choppiness < regime_thresholds['CHOPPINESS_THRESHOLD']:
            if combined_direction_score > 0.5:
                regime = "STRONG_UPTREND"
                strength = combined_trend_score * (1 + abs(combined_direction_score))
            elif combined_direction_score < -0.5:
                regime = "STRONG_DOWNTREND"
                strength = combined_trend_score * (1 + abs(combined_direction_score))
            else:
                regime = "EMERGING_TREND"
                strength = combined_trend_score
        
        # Range bound - ENHANCED with adjusted threshold
        elif combined_range_score > range_threshold_adj and combined_trend_score < 0.4:
            regime = "RANGE_BOUND"
            # ENHANCED: Adjust strength calculation to better reflect ranging quality
            if volatility_level == "super-extreme" or volatility_level == "extreme":
                # In high volatility, give more weight to range_score
                strength = combined_range_score * 1.2
            else:
                strength = combined_range_score
        
        # Volatility expansion - potential breakout
        elif combined_volatility_score > 0.7 and combined_trend_score > 0.4:
            regime = "VOLATILITY_BREAKOUT"
            strength = combined_volatility_score
        
        # Choppy volatile market
        elif combined_volatility_score > 0.6 and combined_choppiness > 60:
            regime = "CHOPPY_VOLATILE"
            strength = combined_volatility_score
        
        # Low volatility compression - potential breakout setup
        elif combined_volatility_score < 0.3 and combined_range_score > 0.6:
            regime = "VOLATILITY_COMPRESSION"
            strength = combined_range_score
        
        # Undefined or mixed conditions
        else:
            regime = "UNDEFINED"
            strength = 0.5
        
        logging.info(f"Market regime for {symbol}: {regime} (strength: {strength:.2f})")
        return regime, strength
        
    except Exception as e:
        logging.error(f"Error detecting market regime: {e}")
        return "UNDEFINED", 0.5
    
def identify_swing_points(df, window_size=5):
    """
    Identify swing high and swing low points in price data.
    
    Args:
        df: DataFrame with OHLC data
        window_size: Window size for identifying swing points
        
    Returns:
        tuple: (swing_highs, swing_lows) - lists of (index, price) tuples
    """
    swing_highs = []
    swing_lows = []
    
    # Process data
    for i in range(window_size, len(df) - window_size):
        # Check for swing high
        if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window_size+1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window_size+1)):
            swing_highs.append((i, df['high'].iloc[i]))
        
        # Check for swing low
        if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window_size+1)) and \
           all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window_size+1)):
            swing_lows.append((i, df['low'].iloc[i]))
    
    return swing_highs, swing_lows

def is_pattern_higher_highs(swing_points):
    """Check if swing points form higher highs pattern."""
    if len(swing_points) < 2:
        return False
    return all(swing_points[i][1] > swing_points[i-1][1] for i in range(1, len(swing_points)))

def is_pattern_higher_lows(swing_points):
    """Check if swing points form higher lows pattern."""
    if len(swing_points) < 2:
        return False
    return all(swing_points[i][1] > swing_points[i-1][1] for i in range(1, len(swing_points)))

def is_pattern_lower_highs(swing_points):
    """Check if swing points form lower highs pattern."""
    if len(swing_points) < 2:
        return False
    return all(swing_points[i][1] < swing_points[i-1][1] for i in range(1, len(swing_points)))

def is_pattern_lower_lows(swing_points):
    """Check if swing points form lower lows pattern."""
    if len(swing_points) < 2:
        return False
    return all(swing_points[i][1] < swing_points[i-1][1] for i in range(1, len(swing_points)))

def calculate_choppiness_index(df, period=14):
    """
    Calculate Choppiness Index to determine if market is choppy or trending.
    Values above 61.8 indicate a choppy market, below 38.2 indicate a trending market.
    """
    atr_sum = 0
    
    # Calculate true range
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    
    # Calculate highest high and lowest low
    highest_high = df['high'].rolling(period).max()
    lowest_low = df['low'].rolling(period).min()
    
    # Calculate choppiness index
    choppiness = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness.iloc[-1]

def calculate_dynamic_trailing_stop(symbol, ticket, current_price, position_type, entry_price):
    """
    Highly adaptive trailing stop system that adjusts to market conditions,
    volatility, and trend strength.
    """
    try:
        # Get current position
        positions = get_positions(symbol)
        position = next((p for p in positions if p.ticket == ticket), None)
        
        if position is None:
            logging.warning(f"Position {ticket} not found. Cannot apply trailing stop.")
            return None
        
        # Get current stop loss
        current_sl = position.sl
        
        # Calculate position age in hours
        position_age_hours = (time.time() - position.time) / 3600
        
        # Calculate profit percentage
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return current_sl
            
        point = symbol_info.point
        
        if position_type == mt5.ORDER_TYPE_BUY:
            profit_points = (current_price - entry_price) / point
        else:  # mt5.ORDER_TYPE_SELL
            profit_points = (entry_price - current_price) / point
            
        profit_percent = profit_points / (entry_price / point) * 100
        
        # Get market data
        df = get_candle_data(symbol, optimize_timeframe_for_volatility(symbol))
        if df is None:
            return current_sl
            
        # Get key market metrics
        atr_value = calculate_atr(df)
        market_regime, regime_strength = detect_market_regime(symbol)
        
        # Calculate activation threshold with advanced adjustments
        # MODIFIED: Reduced base threshold to 0.3% (from 0.5%)
        base_activation = 0.3  # 0.3% activation threshold
        
        # Reduce activation threshold based on position age
        # MODIFIED: Increased hourly reduction rate to 0.2% (from 0.1%)
        age_reduction = min(base_activation * 0.5, position_age_hours * 0.2)
        adjusted_activation = base_activation - age_reduction
        
        # Adjust based on volatility
        volatility_level, atr_percent = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
        
        if volatility_level == "extreme":
            volatility_multiplier = 1.2  # Higher threshold in extreme volatility
        elif volatility_level == "high":
            volatility_multiplier = 1.1  # Slightly higher threshold in high volatility
        elif volatility_level == "low":
            volatility_multiplier = 0.8  # Lower threshold in low volatility
        else:
            volatility_multiplier = 1.0
            
        # Adjust based on market regime
        if market_regime == "STRONG_UPTREND" and position_type == mt5.ORDER_TYPE_BUY:
            regime_multiplier = 0.9  # Lower threshold to activate TSL sooner in strong uptrend for buys
        elif market_regime == "STRONG_DOWNTREND" and position_type == mt5.ORDER_TYPE_SELL:
            regime_multiplier = 0.9  # Lower threshold to activate TSL sooner in strong downtrend for sells
        elif market_regime == "CHOPPY_VOLATILE":
            regime_multiplier = 1.2  # Higher threshold in choppy markets
        else:
            regime_multiplier = 1.0
            
        # Calculate final activation threshold
        final_activation = adjusted_activation * volatility_multiplier * regime_multiplier
        
        # Log detailed information
        logging.info(f"Position {ticket} {symbol}: Profit {profit_percent:.2f}%, " + 
                    f"Activation threshold {final_activation:.2f}% (Age: {position_age_hours:.1f}h, " +
                    f"Volatility: {volatility_level}, Regime: {market_regime})")
        
        # Check if we've reached activation threshold
        if profit_percent < final_activation:
            logging.info(f"Position {ticket}: Profit below threshold. No trailing stop update.")
            return current_sl
            
        # NEW: Option to remove take profit after TSL activation
        remove_tp = config.getboolean('TrailingStop', 'remove_take_profit_after_activation', fallback=False)
        if remove_tp and position.tp != 0:
            update_position_stops(symbol, ticket, current_sl, 0)
            logging.info(f"Removed take profit for position {ticket} after TSL activation")
        
        # Calculate adaptive trailing settings based on profit zones
        # Initial activation zone
        if profit_percent < final_activation * 1.5:
            trail_factor = 0.8  # Loose trailing - 80% of initial distance
            
        # Moderate profit zone (1.5x-2.5x activation)
        elif profit_percent < final_activation * 2.5:
            trail_factor = 0.65  # Medium trailing - 65% of initial distance
            
        # Strong profit zone (2.5x-4x activation)
        elif profit_percent < final_activation * 4:
            trail_factor = 0.5  # Tighter trailing - 50% of initial distance
            
        # Exceptional profit zone (>4x activation)
        else:
            trail_factor = 0.35  # Very tight trailing - 35% of initial distance
            
        # Adjust based on market conditions
        # NEW: Check market regime for more aggressive trailing in strong trends
        market_regime, regime_strength = detect_market_regime(symbol)
        
        # In aligned strong trends, use looser trailing to let profits run
        if (market_regime == "STRONG_UPTREND" and position_type == mt5.ORDER_TYPE_BUY) or \
           (market_regime == "STRONG_DOWNTREND" and position_type == mt5.ORDER_TYPE_SELL):
            # Adjust trailing factor based on regime strength
            if regime_strength > 0.8:  # Very strong trend
                trail_factor *= 1.5  # 50% looser trailing - let profits run much longer
                logging.info(f"Using much looser trailing (50% wider) to let profits run in strong {market_regime}")
            elif regime_strength > 0.6:  # Strong trend
                trail_factor *= 1.3  # 30% looser trailing - let profits run longer
                logging.info(f"Using looser trailing (30% wider) to let profits run in strong {market_regime}")
                
        # MODIFIED: Lower profit lock levels in strong trends to avoid early exits
        if profit_percent > final_activation * 2:
            if (market_regime == "STRONG_UPTREND" and position_type == mt5.ORDER_TYPE_BUY) or \
               (market_regime == "STRONG_DOWNTREND" and position_type == mt5.ORDER_TYPE_SELL):
                # In aligned strong trends, reduced profit locking
                lock_percent *= 0.7  # Lock only 70% of what we would normally lock
                logging.info(f"Reduced profit locking to {lock_percent*100:.1f}% in strong trend")
            
        # In choppy or ranging markets, use tighter trailing
        elif market_regime in ["CHOPPY_VOLATILE", "RANGE_BOUND"]:
            trail_factor *= 0.8  # 20% tighter in choppy markets
            
        # In extreme volatility, adjust for risk
        if volatility_level == "extreme":
            trail_factor *= 1.3  # 30% wider trail in extreme volatility
        elif volatility_level == "high":
            trail_factor *= 1.1  # 10% wider trail in high volatility
            
        # Calculate trail distance
        trail_distance = atr_value * trail_factor
        
        # Calculate new stop loss based on position type
        if position_type == mt5.ORDER_TYPE_BUY:
            new_sl = current_price - trail_distance
            
            # Only update if better than current
            if new_sl <= current_sl:
                return current_sl
                
        else:  # mt5.ORDER_TYPE_SELL
            new_sl = current_price + trail_distance
            
            # Only update if better than current
            if new_sl >= current_sl:
                return current_sl
                
        # Ensure stop doesn't go below breakeven after significant profit
        breakeven_buffer = atr_value * 0.2  # Small buffer beyond breakeven
        
        if profit_percent > final_activation * 2:
            if position_type == mt5.ORDER_TYPE_BUY:
                new_sl = max(new_sl, entry_price + breakeven_buffer)
            else:  # mt5.ORDER_TYPE_SELL
                new_sl = min(new_sl, entry_price - breakeven_buffer)
                
        # Apply profit locking based on TSL_PROFIT_LOCK levels
        for i, level in enumerate(TSL_PROFIT_LOCK_LEVELS):
            level_percent = level * 100  # Convert to percentage
            if profit_percent >= level_percent:
                lock_percent = TSL_PROFIT_LOCK_PERCENTS[i]
                
                # Calculate minimum lock level
                if position_type == mt5.ORDER_TYPE_BUY:
                    min_lock_level = entry_price + (profit_points * lock_percent * point)
                    new_sl = max(new_sl, min_lock_level)
                else:  # mt5.ORDER_TYPE_SELL
                    min_lock_level = entry_price - (profit_points * lock_percent * point)
                    new_sl = min(new_sl, min_lock_level)
                    
                logging.info(f"Applied profit lock at {level_percent:.1f}% profit, locking {lock_percent*100:.1f}% of profits")
        
        # Round to symbol precision
        digits = symbol_info.digits
        new_sl = round(new_sl, digits)
        
        # Log detailed information about the update
        if new_sl != current_sl:
            if position_type == mt5.ORDER_TYPE_BUY:
                protected_pips = (new_sl - entry_price) / point
                protected_percent = (protected_pips / profit_points) * 100 if profit_points > 0 else 0
            else:
                protected_pips = (entry_price - new_sl) / point
                protected_percent = (protected_pips / profit_points) * 100 if profit_points > 0 else 0
                
            logging.info(f"Enhanced TSL Update for {symbol} position {ticket}: " +
                        f"Current SL: {current_sl} → New SL: {new_sl} " +
                        f"(Profit: {profit_percent:.2f}%, Protecting: {protected_percent:.1f}%, " +
                        f"Trail factor: {trail_factor:.2f})")
        
        return new_sl
        
    except Exception as e:
        logging.error(f"Error in dynamic trailing stop: {e}")
        return None

def calculate_risk_adjusted_lot_size(symbol, risk_percentage, stop_loss_pips, market_regime="UNDEFINED"):
    """
    Advanced position sizing algorithm that adapts to market conditions,
    account health, and trading performance. Much more aggressive when conditions are favorable.
    """
    try:
        # Check for global safety mode
        if hasattr(state, 'safe_mode_until') and time.time() < state.safe_mode_until:
            logging.warning(f"System in safe mode - using minimum lot size for {symbol}")
            return get_min_lot_size(symbol)
            
        # Get account info
        account_info = get_account_info()
        if account_info is None:
            return get_min_lot_size(symbol)

        # Critical margin level check
        if account_info.margin > 0 and account_info.margin_level is not None:
            if account_info.margin_level < 200:
                logging.warning(f"Critical margin level ({account_info.margin_level:.1f}%) - using minimum lot size")
                return get_min_lot_size(symbol)
                
        # Check for position cooldown after errors
        if hasattr(state, 'margin_error_cooldown') and symbol in state.margin_error_cooldown:
            if time.time() < state.margin_error_cooldown[symbol]:
                logging.info(f"{symbol} in margin error cooldown - using reduced lot size")
                return get_min_lot_size(symbol)
            
        # Get signal quality metrics
        if hasattr(state, 'current_signals') and symbol in state.current_signals:
            signal_data = state.current_signals[symbol]
            signal_strength = signal_data.get("strength", 0)
            
            # Check ML confidence if available
            ml_confidence = 0.5  # Default
            if hasattr(state, 'signal_validation_models') and symbol in state.signal_validation_models:
                _, ml_confidence = validate_trading_signal(symbol, signal_data)

            # Scale risk UP when signal is strong - MUCH MORE AGGRESSIVE
            signal_quality = signal_strength * ml_confidence
            
            # Increase risk for high-quality signals
            if signal_quality > 0.85:  # Extremely strong signal
                risk_multiplier = 2.0  # DOUBLE risk for exceptional signals
                logging.info(f"DOUBLING risk due to exceptional signal quality: {signal_quality:.2f}")
            elif signal_quality > 0.75:  # Very strong signal
                risk_multiplier = 1.7  # Increase risk by 70%
                logging.info(f"Increasing risk by 70% due to very strong signal quality: {signal_quality:.2f}")
            elif signal_quality > 0.65:  # Strong signal
                risk_multiplier = 1.4  # Increase risk by 40%
                logging.info(f"Increasing risk by 40% due to good signal quality: {signal_quality:.2f}")
            else:
                risk_multiplier = 1.0  # Normal risk

            # Apply risk multiplier
            adjusted_risk_percentage = risk_percentage * risk_multiplier
            
            # Check for pattern confirmation
            if "chart_patterns" in signal_data:
                pattern_data = signal_data["chart_patterns"]
                if pattern_data.get("detected", False):
                    pattern_reliability = pattern_data.get("reliability", 0)
                    pattern_type = pattern_data.get("type", "neutral")
                    pattern_name = pattern_data.get("pattern", "unknown")
                    signal_type = signal_data.get("signal", "NONE")
                    
                    # Give extra weight to head and shoulders pattern for SELL signals
                    head_shoulders_bonus = 1.0
                    if pattern_name == "head_and_shoulders" and pattern_type == "bearish" and signal_type == "SELL":
                        head_shoulders_bonus = 1.3  # 30% extra for this specific pattern
                        logging.info(f"Adding 30% extra weight to head and shoulders bearish pattern")
                    
                    # If pattern aligns with signal and is reliable, increase risk further
                    if (pattern_type == "bullish" and signal_type == "BUY") or \
                    (pattern_type == "bearish" and signal_type == "SELL"):
                        if pattern_reliability > 0.7:  # Very reliable pattern
                            pattern_multiplier = 1.5 * head_shoulders_bonus  # Apply bonus for head and shoulders
                            adjusted_risk_percentage *= pattern_multiplier
                            logging.info(f"Increasing risk by {(pattern_multiplier-1)*100:.0f}% due to strong confirming pattern: {pattern_name}")
        else:
            adjusted_risk_percentage = risk_percentage
        
        # Use equity instead of balance for risk calculation
        equity = account_info.equity
        
        # NEW: Adjust for market regime - be much more aggressive in strong trends
        if market_regime == "STRONG_UPTREND" or market_regime == "STRONG_DOWNTREND":
            # Increase size significantly in strong trends
            regime_multiplier = 1.5  # 50% increase - MORE AGGRESSIVE
            adjusted_risk_percentage *= regime_multiplier
            logging.info(f"Increasing position size by 50% due to strong {market_regime}")
        elif market_regime == "CHOPPY_VOLATILE":
            # Reduce size in choppy markets but not as dramatically
            regime_multiplier = 0.8  # 20% reduction
            adjusted_risk_percentage *= regime_multiplier
        elif market_regime == "RANGE_BOUND":
            # Be more opportunistic in range-bound markets
            regime_multiplier = 1.1  # 10% increase - MORE AGGRESSIVE
            adjusted_risk_percentage *= regime_multiplier
            
        # Get symbol info
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return get_min_lot_size(symbol)
        
        # Get symbol properties
        point = symbol_info.point
        contract_size = symbol_info.trade_contract_size
        price = mt5.symbol_info_tick(symbol).ask
        
        # Currency conversion if needed
        conversion_rate = calculate_currency_conversion_rate(symbol_info, account_info)
        
        # Scale account safety factor based on account size - logarithmic scaling
        base_equity = 1000.0  # Reference point
        equity_ratio = equity / base_equity
        
        # Use more aggressive account scaling - MODIFIED
        account_scaling = min(1.5, max(0.5, 0.5 + (0.5 * math.log10(equity_ratio + 0.1))))
        
        # Similarly scale margin safety buffer more aggressively for small accounts
        margin_safety_factor = min(0.95, max(0.4, 0.4 + (0.55 * math.log10(equity_ratio + 0.1))))
        
        # Special handling for crypto with small accounts
        if symbol.startswith("BTC") or symbol.startswith("ETH"):
            if equity < 2000:
                crypto_factor = min(1.0, equity / 2000)
                account_scaling *= crypto_factor
        
        # Check current margin usage and reduce safety factor if already high
        margin_used_percent = account_info.margin / account_info.equity if account_info.equity > 0 else 0
        if margin_used_percent > 0.5:  # Only reduce at higher margin usage
            margin_safety_factor *= (1 - (margin_used_percent - 0.5))
        
        # Apply position count scaling - MODIFIED to be less restrictive
        position_count = len(get_positions())
        if position_count > 3:  # Only scale down significantly above 3 positions
            exposure_scaling = max(0.6, 1.0 - ((position_count - 3) * 0.1))
        else:
            exposure_scaling = 1.0  # No reduction for first 3 positions
        
        # Apply scaling factors to base risk percentage
        final_risk_percentage = (
            adjusted_risk_percentage * 
            account_scaling * 
            exposure_scaling
        )
        
        # Apply global risk modifier
        if hasattr(state, 'risk_modifier'):
            final_risk_percentage *= state.risk_modifier
        
        # Cap maximum risk percentage - INCREASED
        max_risk = MAX_RISK_PER_TRADE * 100 * 1.5  # 50% higher cap - MORE AGGRESSIVE
        final_risk_percentage = min(final_risk_percentage, max_risk)
        
        # Calculate risk amount
        risk_amount = equity * (final_risk_percentage / 100)
        
        # Calculate pip value
        pip_value = (point * contract_size) / price
        
        # Convert pip value to account currency
        pip_value_in_account_currency = pip_value * conversion_rate
        
        # Calculate lot size
        stop_loss_amount = stop_loss_pips * pip_value_in_account_currency
        lot_size = risk_amount / stop_loss_amount if stop_loss_amount > 0 else get_min_lot_size(symbol)
        
        # Scale based on trading performance - MORE AGGRESSIVE
        win_rate = state.trade_stats.get("win_rate", 50)
        profit_factor = state.trade_stats.get("profit_factor", 1.0)
        
        # Scale lot size based on trading performance
        if win_rate > 60 and profit_factor > 1.5:
            performance_multiplier = 1.5  # 50% increase - MORE AGGRESSIVE
        elif win_rate > 55 and profit_factor > 1.2:
            performance_multiplier = 1.3  # 30% increase - MORE AGGRESSIVE
        elif win_rate < 40 or profit_factor < 0.8:
            performance_multiplier = 0.8
        else:
            performance_multiplier = 1.0
        
        lot_size *= performance_multiplier
        
        # Scale back for initial trades but not as much - MORE AGGRESSIVE
        total_trades = state.trade_stats.get("total_trades", 0)
        if total_trades < 10:
            warmup_factor = 0.7 + (total_trades * 0.03)  # Start at 70% and increase faster
            lot_size *= warmup_factor
        
        # Check volatility and adjust lot size - MODIFIED
        volatility_level, _ = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
        if volatility_level == "extreme":
            vol_multiplier = 0.8  # Less reduction in extreme volatility
            lot_size *= vol_multiplier
        elif volatility_level == "high":
            vol_multiplier = 0.9  # Less reduction in high volatility
            lot_size *= vol_multiplier
        
        # Check for high-impact news
        if check_news_events(symbol, hours=12):
            lot_size *= 0.8  # Less reduction around news events
        
        # Ensure lot size is within symbol's limits
        lot_size = max(lot_size, symbol_info.volume_min)
        lot_size = min(lot_size, symbol_info.volume_max)
        
        # Round to nearest step
        lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
        
        # Check if margin is sufficient with safety buffer
        margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, lot_size, price)
        if margin and margin > account_info.margin_free * margin_safety_factor:
            max_affordable_lot = (account_info.margin_free * margin_safety_factor / margin) * lot_size
            max_affordable_lot = max(max_affordable_lot, symbol_info.volume_min)
            max_affordable_lot = min(max_affordable_lot, symbol_info.volume_max)
            max_affordable_lot = round(max_affordable_lot / symbol_info.volume_step) * symbol_info.volume_step
            lot_size = max_affordable_lot
        
        # Maximum exposure limit - INCREASED from 5% to 8% - MORE AGGRESSIVE
        position_value = lot_size * contract_size * price
        max_position_value = equity * 0.08
        
        if position_value > max_position_value:
            reduced_lot_size = max_position_value / (contract_size * price)
            reduced_lot_size = max(reduced_lot_size, symbol_info.volume_min)
            reduced_lot_size = min(reduced_lot_size, symbol_info.volume_max)
            reduced_lot_size = round(reduced_lot_size / symbol_info.volume_step) * symbol_info.volume_step
            lot_size = reduced_lot_size
        
        # Apply additional margin safety if needed but less aggressively
        if hasattr(state, 'margin_errors') and symbol in getattr(state, 'margin_errors', {}):
            logging.warning(f"Applying margin safety factor for {symbol} due to previous margin errors")
            lot_size *= 0.7  # Reduce by 30% instead of 50% - MORE AGGRESSIVE
        
        logging.info(f"Risk-adjusted lot size for {symbol}: {lot_size} lots " +
                    f"(Risk: {final_risk_percentage:.2f}%, Stop loss: {stop_loss_pips} pips, " +
                    f"Regime: {market_regime})")
        
        return lot_size
    
    except Exception as e:
        logging.error(f"Error calculating risk-adjusted lot size: {e}")
        # Track margin exceeded errors to apply additional safety
        if 'margin_exceeded' in str(e):
            if not hasattr(state, 'margin_errors'):
                state.margin_errors = {}
            state.margin_errors[symbol] = True
            
            # Add a cooldown period for this symbol after margin errors
            if not hasattr(state, 'margin_error_cooldown'):
                state.margin_error_cooldown = {}
            
            state.margin_error_cooldown[symbol] = time.time() + 3600  # 1 hour cooldown
            
            logging.warning(f"Margin exceeded error detected for {symbol}. Will apply additional safety in future runs.")
        
        return get_min_lot_size(symbol)
    
def get_min_lot_size(symbol):
    """Helper function to get minimum lot size for a symbol."""
    symbol_info = get_symbol_info(symbol)
    if symbol_info:
        return symbol_info.volume_min
    return 0.01  # Default minimum

def setup_error_handling():
    """
    Configure advanced error handling and recovery mechanisms,
    including signal handling and automatic exception recovery.
    """
    # Register shutdown handler with atexit
    import atexit
    atexit.register(enhanced_shutdown_handler)
    
    # Register signal handlers
    import signal
    
    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else f"Signal {signum}"
        logging.warning(f"Received {signal_name}, initiating graceful shutdown")
        
        # Set exit flag for threads
        state.exit_requested = True
        
        # Run shutdown handler to protect positions
        enhanced_shutdown_handler()
        
        # Wait briefly to let threads see the exit flag
        time.sleep(2)
        
        # Exit with appropriate code
        if signum == signal.SIGINT:
            sys.exit(130)  # Standard exit code for SIGINT
        else:
            sys.exit(128 + signum)
    
    # Register signal handlers for common termination signals
    for sig in [signal.SIGINT, signal.SIGTERM]:
        try:
            signal.signal(sig, signal_handler)
        except (ValueError, OSError):
            # Some signals might not be available on all platforms
            logging.warning(f"Could not register handler for signal {sig}")
    
    # Setup global exception handler
    def global_exception_handler(exctype, value, traceback_obj):
        logging.critical(f"Unhandled exception: {exctype.__name__}: {value}")
        
        # Log traceback
        import traceback
        for line in traceback.format_tb(traceback_obj):
            logging.critical(line.rstrip())
        
        # Run shutdown handler to protect positions
        enhanced_shutdown_handler()
        
        # Call original exception handler
        sys.__excepthook__(exctype, value, traceback_obj)
    
    # Set as global exception handler
    sys.excepthook = global_exception_handler
    
    # Setup thread exception handling
    threading.excepthook = lambda args: (
        logging.critical(f"Unhandled exception in thread {args.thread.name}: {args.exc_type.__name__}: {args.exc_value}"),
        enhanced_shutdown_handler()
    )
    
    logging.info("Advanced error handling and recovery system initialized")

def enhanced_shutdown_handler():
    """
    Advanced shutdown handler that protects positions during unexpected shutdowns,
    network issues, or program termination.
    """
    logging.info("ENHANCED SHUTDOWN SEQUENCE INITIATED")
    
    try:
        # Get all open positions
        positions = get_positions()
        logging.info(f"Managing {len(positions)} open positions during shutdown")
        
        protected_count = 0
        
        for position in positions:
            position_id = position.ticket
            symbol = position.symbol
            position_type = position.type
            entry_price = position.price_open
            
            try:
                # Get current market data
                current_price = mt5.symbol_info_tick(symbol).bid if position_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
                
                # 1. Ensure all positions have stop losses - NEW: More aggressive protection
                if position.sl is None:
                    # Get market data for smarter stop placement
                    df = get_candle_data(symbol, MT5_TIMEFRAMES["PRIMARY"], num_bars=20)
                    if df is not None:
                        atr = calculate_atr(df)
                        
                        # Set stop based on ATR if available
                        if position_type == mt5.ORDER_TYPE_BUY:
                            # Set stop 1.5 ATR below current price or at 98% of entry, whichever is higher
                            atr_stop = current_price - (atr * 1.5)
                            min_stop = entry_price * 0.98
                            stop_price = max(atr_stop, min_stop)
                        else:  # mt5.ORDER_TYPE_SELL
                            # Set stop 1.5 ATR above current price or at 102% of entry, whichever is lower
                            atr_stop = current_price + (atr * 1.5)
                            max_stop = entry_price * 1.02
                            stop_price = min(atr_stop, max_stop)
                    else:
                        # Fallback if no market data available
                        if position_type == mt5.ORDER_TYPE_BUY:
                            stop_price = current_price * 0.99  # 1% below current price
                        else:  # mt5.ORDER_TYPE_SELL
                            stop_price = current_price * 1.01  # 1% above current price
                    
                    # Round to appropriate precision
                    digits = get_symbol_info(symbol).digits
                    stop_price = round(stop_price, digits)
                    
                    # Set the stop loss
                    update_position_stops(symbol, position_id, stop_price)
                    logging.info(f"SHUTDOWN: Added safety stop loss at {stop_price:.2f} for position {position_id}")
                    protected_count += 1
                
                # 2. Protect profitable positions - NEW: Lock in more profits
                if position.sl is not None:
                    # For positions in profit, move stops closer to secure gains
                    if (position_type == mt5.ORDER_TYPE_BUY and current_price > entry_price) or \
                       (position_type == mt5.ORDER_TYPE_SELL and current_price < entry_price):
                        
                        # Calculate profit percentage
                        symbol_info = get_symbol_info(symbol)
                        if symbol_info:
                            point = symbol_info.point
                            
                            if position_type == mt5.ORDER_TYPE_BUY:
                                profit_points = (current_price - entry_price) / point
                            else:  # mt5.ORDER_TYPE_SELL
                                profit_points = (entry_price - current_price) / point
                                
                            profit_percent = profit_points / (entry_price / point) * 100
                            
                            # Set aggressive trailing stop based on profit percentage
                            if profit_percent > 2.0:  # Significant profit
                                if position_type == mt5.ORDER_TYPE_BUY:
                                    # Lock in at least 75% of profit
                                    new_stop = entry_price + (profit_points * 0.75 * point)
                                    if new_stop > position.sl:
                                        update_position_stops(symbol, position_id, new_stop)
                                        logging.info(f"SHUTDOWN: Locked in 75% profit at {new_stop:.2f} for position {position_id}")
                                        protected_count += 1
                                else:  # mt5.ORDER_TYPE_SELL
                                    # Lock in at least 75% of profit
                                    new_stop = entry_price - (profit_points * 0.75 * point)
                                    if new_stop < position.sl:
                                        update_position_stops(symbol, position_id, new_stop)
                                        logging.info(f"SHUTDOWN: Locked in 75% profit at {new_stop:.2f} for position {position_id}")
                                        protected_count += 1
                            elif profit_percent > 1.0:  # Moderate profit
                                if position_type == mt5.ORDER_TYPE_BUY:
                                    # Lock in at least 50% of profit
                                    new_stop = entry_price + (profit_points * 0.5 * point)
                                    if new_stop > position.sl:
                                        update_position_stops(symbol, position_id, new_stop)
                                        logging.info(f"SHUTDOWN: Locked in 50% profit at {new_stop:.2f} for position {position_id}")
                                        protected_count += 1
                                else:  # mt5.ORDER_TYPE_SELL
                                    # Lock in at least 50% of profit
                                    new_stop = entry_price - (profit_points * 0.5 * point)
                                    if new_stop < position.sl:
                                        update_position_stops(symbol, position_id, new_stop)
                                        logging.info(f"SHUTDOWN: Locked in 50% profit at {new_stop:.2f} for position {position_id}")
                                        protected_count += 1
                            elif profit_percent > 0.5:  # Small profit
                                # At least move to breakeven
                                if position_type == mt5.ORDER_TYPE_BUY and position.sl < entry_price:
                                    update_position_stops(symbol, position_id, entry_price)
                                    logging.info(f"SHUTDOWN: Moved stop to breakeven for position {position_id}")
                                    protected_count += 1
                                elif position_type == mt5.ORDER_TYPE_SELL and position.sl > entry_price:
                                    update_position_stops(symbol, position_id, entry_price)
                                    logging.info(f"SHUTDOWN: Moved stop to breakeven for position {position_id}")
                                    protected_count += 1
                
                # 3. NEW: Handle positions in significant loss
                # For positions in significant loss, consider closing or setting tight stops
                if (position_type == mt5.ORDER_TYPE_BUY and current_price < entry_price * 0.97) or \
                   (position_type == mt5.ORDER_TYPE_SELL and current_price > entry_price * 1.03):
                    
                    # Position is in 3%+ loss - consider closing if in volatile market
                    volatility_level, _ = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
                    
                    if volatility_level in ["high", "extreme"]:
                        # Close position in high volatility
                        close_position(position_id)
                        logging.warning(f"SHUTDOWN: Closed losing position {position_id} in {volatility_level} volatility")
                    else:
                        # Set tight stop to limit further losses
                        if position_type == mt5.ORDER_TYPE_BUY:
                            tight_stop = current_price * 0.995  # 0.5% below current
                        else:  # mt5.ORDER_TYPE_SELL
                            tight_stop = current_price * 1.005  # 0.5% above current
                            
                        # Round to appropriate precision
                        digits = get_symbol_info(symbol).digits
                        tight_stop = round(tight_stop, digits)
                        
                        update_position_stops(symbol, position_id, tight_stop)
                        logging.warning(f"SHUTDOWN: Set tight stop at {tight_stop:.2f} for losing position {position_id}")
                        protected_count += 1
                        
            except Exception as e:
                logging.error(f"SHUTDOWN: Error protecting position {position_id}: {e}")
                
        logging.info(f"SHUTDOWN: All positions protected with stop losses ({protected_count} updated)")
        
    except Exception as e:
        logging.error(f"SHUTDOWN: Error during shutdown sequence: {e}")
    
    logging.info("SHUTDOWN: Shutdown sequence completed")

def manage_completed_trade(trade_result):
    """
    Process completed trade for statistical tracking and ML model training.
    """
    try:
        # Update trade statistics
        update_trade_stats(trade_result)
        
        # Check if this trade has a corresponding signal for ML training
        ticket = trade_result.get("ticket")
        if ticket and hasattr(state, 'pending_signals') and ticket in state.pending_signals:
            # Get the original signal data
            signal_data = state.pending_signals[ticket]
            
            # Collect data for ML model training
            collect_signal_results(trade_result["symbol"], signal_data, trade_result)
            
            # Remove from pending signals
            del state.pending_signals[ticket]
            
            logging.info(f"Collected ML training data for trade {ticket}")
            
        # Write trade note
        write_trade_notes(f"Completed {trade_result['type']} trade on {trade_result['symbol']}: " +
                        f"Profit: {trade_result['profit']:.2f}, Reason: {trade_result['reason']}")
        
    except Exception as e:
        logging.error(f"Error managing completed trade: {e}")

def detect_advanced_chart_patterns(symbol, timeframes=None):
    """
    Advanced pattern recognition that identifies complex chart patterns
    and calculates their reliability.
    
    Args:
        symbol: Trading symbol
        timeframes: List of timeframes to analyze, default is PRIMARY and TREND
        
    Returns:
        dict: Comprehensive pattern detection results
    """
    try:
        # Check cache first
        cache_key = f"{symbol}_patterns"
        current_time = time.time()
        
        # Initialize pattern cache if not exists
        if not hasattr(state, 'pattern_cache'):
            state.pattern_cache = {}
            
        # Return cached result if available and recent (within 15 minutes)
        if cache_key in state.pattern_cache:
            cache_entry = state.pattern_cache[cache_key]
            if current_time - cache_entry['timestamp'] < 900:  # 15 minutes
                return cache_entry['data']

        if timeframes is None:
            timeframes = [MT5_TIMEFRAMES["PRIMARY"], MT5_TIMEFRAMES["TREND"]]
            
        pattern_results = {}

        # Limit number of timeframes analyzed to reduce computation
        timeframes = timeframes[:3] if len(timeframes) > 3 else timeframes
        
        for tf in timeframes:
            # Get data from cache if possible
            df_cache_key = f"{symbol}_{tf}_df"
            if hasattr(state, 'data_cache') and df_cache_key in state.data_cache:
                df = state.data_cache[df_cache_key]
            else:
                df = get_candle_data(symbol, tf, num_bars=200)
                if df is None:
                    continue
                    
                # Cache the dataframe
                if not hasattr(state, 'data_cache'):
                    state.data_cache = {}
                state.data_cache[df_cache_key] = df
        
        # for tf in timeframes:
        #     df = get_candle_data(symbol, tf, num_bars=200)
        #     if df is None:
        #         continue
                
            # Get swing points for pattern detection
            swing_highs, swing_lows = identify_swing_points(df)
            
            # Store pattern detection results for this timeframe
            patterns = {}
            
            # 1. Head and Shoulders / Inverse Head and Shoulders
            head_shoulders = detect_head_shoulders_pattern(df, swing_highs, swing_lows)
            if head_shoulders["detected"]:
                patterns["head_shoulders"] = head_shoulders
                
            # 2. Double Top / Double Bottom
            double_patterns = detect_double_formations(df, swing_highs, swing_lows)
            if double_patterns["detected"]:
                patterns["double_formation"] = double_patterns
                
            # 3. Triple Top / Triple Bottom
            triple_patterns = detect_triple_formations(df, swing_highs, swing_lows)
            if triple_patterns["detected"]:
                patterns["triple_formation"] = triple_patterns
                
            # 4. Wedge Patterns (Rising/Falling)
            wedges = detect_wedge_patterns(df, swing_highs, swing_lows)
            if wedges["detected"]:
                patterns["wedge"] = wedges
                
            # 5. Flag and Pennant Patterns
            flags = detect_flag_patterns(df, swing_highs, swing_lows)
            if flags["detected"]:
                patterns["flag"] = flags
                
            # 6. Rounding Patterns (Cup and Handle)
            rounding = detect_rounding_patterns(df)
            if rounding["detected"]:
                patterns["rounding"] = rounding
                
            # 7. Channel Patterns
            channels = detect_channel_patterns(df, swing_highs, swing_lows)
            if channels["detected"]:
                patterns["channel"] = channels
                
            # 8. Rectangle Patterns (Trading Ranges)
            rectangles = detect_rectangle_patterns(df, swing_highs, swing_lows)
            if rectangles["detected"]:
                patterns["rectangle"] = rectangles
                
            # 9. Island Reversals
            islands = detect_island_reversals(df)
            if islands["detected"]:
                patterns["island_reversal"] = islands
                
            # 10. Traditional candlestick patterns
            candlestick = detect_candlestick_patterns(df)
            patterns["candlestick"] = {
                "detected": candlestick.get("bullish_count", 0) > 0 or candlestick.get("bearish_count", 0) > 0,
                "type": candlestick.get("overall_sentiment", "neutral"),
                "reliability": 0.7 if candlestick.get("bullish_count", 0) > 1 or candlestick.get("bearish_count", 0) > 1 else 0.5,
                "details": candlestick
            }
            
            # Extract pattern signals
            bullish_patterns = []
            bearish_patterns = []
            
            for name, pattern in patterns.items():
                if isinstance(pattern, dict) and pattern.get("detected", False):
                    if pattern.get("type") == "bullish":
                        bullish_patterns.append({
                            "name": name,
                            "reliability": pattern.get("reliability", 0.5),
                            "completion": pattern.get("completion", 1.0)
                        })
                    elif pattern.get("type") == "bearish":
                        bearish_patterns.append({
                            "name": name,
                            "reliability": pattern.get("reliability", 0.5),
                            "completion": pattern.get("completion", 1.0)
                        })
            
            # Calculate overall pattern strength and direction
            bullish_strength = sum(p["reliability"] * p["completion"] for p in bullish_patterns)
            bearish_strength = sum(p["reliability"] * p["completion"] for p in bearish_patterns)
            
            # Store timeframe results
            tf_name = {
                mt5.TIMEFRAME_M5: "M5",
                mt5.TIMEFRAME_M15: "M15",
                mt5.TIMEFRAME_M30: "M30",
                mt5.TIMEFRAME_H1: "H1",
                mt5.TIMEFRAME_H4: "H4",
                mt5.TIMEFRAME_D1: "D1",
            }.get(tf, str(tf))
            
            pattern_results[tf_name] = {
                "patterns": patterns,
                "bullish_patterns": bullish_patterns,
                "bearish_patterns": bearish_patterns,
                "bullish_strength": bullish_strength,
                "bearish_strength": bearish_strength,
                "net_direction": bullish_strength - bearish_strength
            }
        
        # Combine results across timeframes with proper weighting
        combined_result = {
            "bullish_strength": 0,
            "bearish_strength": 0,
            "timeframes": pattern_results,
            "patterns_found": {}
        }
        
        # Weight by timeframe importance
        weights = {
            "H4": 0.4, 
            "H1": 0.3, 
            "M30": 0.2,
            "M15": 0.15, 
            "M5": 0.1,
            "D1": 0.5
        }
        
        all_patterns = {}
        
        for tf, result in pattern_results.items():
            weight = weights.get(tf, 0.1)
            combined_result["bullish_strength"] += result["bullish_strength"] * weight
            combined_result["bearish_strength"] += result["bearish_strength"] * weight
            
            # Collect all patterns from this timeframe
            for name, pattern in result.get("patterns", {}).items():
                if isinstance(pattern, dict) and pattern.get("detected", False):
                    if name not in all_patterns:
                        all_patterns[name] = []
                    
                    all_patterns[name].append({
                        "timeframe": tf,
                        "type": pattern.get("type", "neutral"),
                        "reliability": pattern.get("reliability", 0.5),
                        "completion": pattern.get("completion", 1.0)
                    })
        
        # Add all identified patterns to the result
        combined_result["patterns_found"] = all_patterns
        
        # Determine overall direction
        if combined_result["bullish_strength"] > combined_result["bearish_strength"] * 1.5:
            combined_result["direction"] = "bullish"
            combined_result["strength"] = combined_result["bullish_strength"]
        elif combined_result["bearish_strength"] > combined_result["bullish_strength"] * 1.5:
            combined_result["direction"] = "bearish"
            combined_result["strength"] = combined_result["bearish_strength"]
        else:
            combined_result["direction"] = "neutral"
            combined_result["strength"] = max(0.1, min(combined_result["bullish_strength"], combined_result["bearish_strength"]))
        
        # Store result in cache
        state.pattern_cache[cache_key] = {
            'timestamp': current_time,
            'data': combined_result
        }
        
        # Clean up old cache entries to manage memory
        cleanup_cache()

        return combined_result
        
    except Exception as e:
        logging.error(f"Error in advanced pattern recognition: {e}")
        return {"direction": "neutral", "strength": 0, "patterns_found": {}}
    
def cleanup_cache():
    """
    Remove old cache entries to prevent memory bloat.
    """
    try:
        # Clean up pattern cache
        if hasattr(state, 'pattern_cache'):
            current_time = time.time()
            keys_to_remove = []
            
            for key, entry in state.pattern_cache.items():
                # Remove entries older than 1 hour
                if current_time - entry['timestamp'] > 3600:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del state.pattern_cache[key]
                
        # Clean up data cache
        if hasattr(state, 'data_cache'):
            # Limit total entries to 50
            if len(state.data_cache) > 50:
                # Sort by timestamp if available, otherwise remove random entries
                if all('timestamp' in entry for entry in state.data_cache.values()):
                    # Sort by timestamp and keep most recent 50
                    sorted_items = sorted(state.data_cache.items(), 
                                         key=lambda x: x[1].get('timestamp', 0),
                                         reverse=True)
                    state.data_cache = dict(sorted_items[:50])
                else:
                    # Simply keep first 50 entries
                    state.data_cache = dict(list(state.data_cache.items())[:50])
    except Exception as e:
        logging.error(f"Error cleaning up cache: {e}")

def initialize_memory_management():
    """
    Initialize memory management systems for the trading bot.
    """
    try:
        # Set up periodic memory cleanup
        if not hasattr(state, 'last_memory_cleanup'):
            state.last_memory_cleanup = time.time()
            
        # Define data retention limits
        if not hasattr(state, 'data_retention'):
            state.data_retention = {
                'trade_stats': {
                    'last_trades': 100,  # Keep last 100 trades (was 50)
                    'max_age_days': 90   # Keep up to 90 days
                },
                'signals': {
                    'max_count': 500,    # Maximum number of signals to retain
                    'max_age_hours': 24  # Keep up to 24 hours
                },
                'market_state': {
                    'max_count': 50,     # Number of market states to retain
                    'max_age_minutes': 60  # Keep up to 60 minutes
                },
                'missed_opportunities': {
                    'max_count': 200,    # Increased from 100
                    'max_age_hours': 48  # Keep up to 48 hours
                }
            }
            
        logging.info("Memory management initialized")
        
    except Exception as e:
        logging.error(f"Error initializing memory management: {e}")

def perform_memory_cleanup():
    """
    Perform memory cleanup operations to prevent memory leaks.
    Run this function periodically.
    """
    try:
        current_time = time.time()
        
        # Check if cleanup is needed (every 30 minutes)
        if not hasattr(state, 'last_memory_cleanup') or current_time - state.last_memory_cleanup > 1800:
            logging.info("Running memory cleanup...")
            
            # Clean trade statistics
            if hasattr(state, 'trade_stats') and 'last_trades' in state.trade_stats:
                max_trades = state.data_retention['trade_stats']['last_trades']
                if len(state.trade_stats['last_trades']) > max_trades:
                    state.trade_stats['last_trades'] = state.trade_stats['last_trades'][-max_trades:]
                    
                # Remove trades older than retention period
                max_age = state.data_retention['trade_stats']['max_age_days'] * 86400
                current_date = datetime.now()
                
                # Filter out old trades
                filtered_trades = []
                for trade in state.trade_stats['last_trades']:
                    try:
                        # Parse trade date safely
                        trade_date = None
                        if 'exit_time' in trade:
                            if isinstance(trade['exit_time'], str):
                                trade_date = datetime.fromisoformat(trade['exit_time'])
                            elif isinstance(trade['exit_time'], datetime):
                                trade_date = trade['exit_time']
                                
                        if trade_date and (current_date - trade_date).total_seconds() <= max_age:
                            filtered_trades.append(trade)
                    except Exception as e:
                        # If there's any error parsing the date, keep the trade
                        filtered_trades.append(trade)
                        logging.warning(f"Error processing trade date: {e}")
                        
                state.trade_stats['last_trades'] = filtered_trades
            
            # Clean signals cache
            if hasattr(state, 'current_signals'):
                # Limit total count
                max_signals = state.data_retention['signals']['max_count']
                if len(state.current_signals) > max_signals:
                    # Keep only most recent signals
                    state.current_signals = dict(list(state.current_signals.items())[-max_signals:])
            
            # Clean missed opportunities
            if hasattr(state, 'missed_opportunities'):
                max_missed = state.data_retention['missed_opportunities']['max_count']
                if len(state.missed_opportunities) > max_missed:
                    state.missed_opportunities = state.missed_opportunities[-max_missed:]
            
            # Clean market state data
            if hasattr(state, 'market_state'):
                max_states = state.data_retention['market_state']['max_count']
                if len(state.market_state) > max_states:
                    # Remove oldest entries
                    items_to_remove = len(state.market_state) - max_states
                    # Convert to list for easier manipulation
                    market_state_items = list(state.market_state.items())
                    state.market_state = dict(market_state_items[items_to_remove:])
            
            # Clean pending signals
            if hasattr(state, 'pending_signals'):
                # Get all active position tickets
                active_tickets = [p.ticket for p in get_positions()]
                
                # Remove pending signals for positions that no longer exist
                keys_to_remove = [ticket for ticket in state.pending_signals if ticket not in active_tickets]
                for ticket in keys_to_remove:
                    del state.pending_signals[ticket]
            
            # Update last cleanup time
            state.last_memory_cleanup = current_time
            logging.info("Memory cleanup completed")
            
            # Force garbage collection
            import gc
            gc.collect()
            
    except Exception as e:
        logging.error(f"Error in memory cleanup: {e}")

# Individual pattern detection functions
def identify_swing_points(df, window_size=5):
    """
    Identify swing high and swing low points in price data.
    
    Args:
        df: DataFrame with OHLC data
        window_size: Window size for identifying swing points
        
    Returns:
        tuple: (swing_highs, swing_lows) - lists of (index, price, time) tuples
    """
    swing_highs = []
    swing_lows = []
    
    # Process data
    for i in range(window_size, len(df) - window_size):
        # Check for swing high
        if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window_size+1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window_size+1)):
            swing_highs.append((i, df['high'].iloc[i], df['time'].iloc[i]))
        
        # Check for swing low
        if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window_size+1)) and \
           all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window_size+1)):
            swing_lows.append((i, df['low'].iloc[i], df['time'].iloc[i]))
    
    return swing_highs, swing_lows

def detect_head_shoulders_pattern(df, swing_highs, swing_lows):
    """
    Detect Head and Shoulders pattern (regular and inverse).
    
    Returns:
        dict: Pattern details including type, reliability, and price targets
    """
    result = {"detected": False}
    
    try:
        if len(swing_highs) < 3:
            return result
            
        # Look at last 5 significant swing highs for regular H&S
        last_swing_highs = swing_highs[-7:]
        
        # Check for head and shoulders pattern
        for i in range(len(last_swing_highs)-2):
            left = last_swing_highs[i]
            head = last_swing_highs[i+1]
            right = last_swing_highs[i+2]
            
            # Head should be higher than shoulders
            if head[1] > left[1] and head[1] > right[1]:
                # Shoulders should be roughly equal height (within 20%)
                shoulder_diff = abs(left[1] - right[1]) / min(left[1], right[1])
                
                if shoulder_diff < 0.2:  # Shoulders within 20% of each other
                    # Find neckline between shoulders
                    left_idx, right_idx = left[0], right[0]
                    troughs_between = [sl for sl in swing_lows if left_idx < sl[0] < right_idx]
                    
                    if len(troughs_between) >= 2:
                        # Calculate neckline (connecting the troughs)
                        neckline_left = troughs_between[0]
                        neckline_right = troughs_between[-1]
                        
                        # Calculate neckline slope
                        neckline_slope = (neckline_right[1] - neckline_left[1]) / (neckline_right[0] - neckline_left[0])
                        
                        # Calculate neckline at current bar
                        current_idx = len(df) - 1
                        neckline_current = neckline_left[1] + neckline_slope * (current_idx - neckline_left[0])
                        
                        # Calculate pattern height
                        pattern_height = head[1] - ((neckline_left[1] + neckline_right[1]) / 2)
                        
                        # Check if price has broken the neckline
                        current_price = df['close'].iloc[-1]
                        completion = 1.0 if current_price < neckline_current else 0.5
                        
                        # Calculate target
                        target = neckline_current - pattern_height
                        
                        # Calculate reliability based on pattern quality
                        shoulder_symmetry = 1 - shoulder_diff  # 0.8 to 1.0
                        height_factor = pattern_height / df['close'].mean() * 10  # Normalize by price
                        reliability = min(0.9, (shoulder_symmetry * 0.5 + min(height_factor, 0.5)) * completion)
                        
                        result = {
                            "detected": True,
                            "type": "bearish",
                            "pattern": "head_and_shoulders",
                            "reliability": reliability,
                            "completion": completion,
                            "neckline": neckline_current,
                            "target": target,
                            "pattern_height": pattern_height,
                            "head": head[1],
                            "left_shoulder": left[1],
                            "right_shoulder": right[1]
                        }
                        
                        return result
        
        # Check for inverse head and shoulders
        if len(swing_lows) >= 3:
            last_swing_lows = swing_lows[-7:]
            
            for i in range(len(last_swing_lows)-2):
                left = last_swing_lows[i]
                head = last_swing_lows[i+1]
                right = last_swing_lows[i+2]
                
                # Head should be lower than shoulders
                if head[1] < left[1] and head[1] < right[1]:
                    # Shoulders should be roughly equal height (within 20%)
                    shoulder_diff = abs(left[1] - right[1]) / min(left[1], right[1])
                    
                    if shoulder_diff < 0.2:  # Shoulders within 20% of each other
                        # Find neckline between shoulders
                        left_idx, right_idx = left[0], right[0]
                        peaks_between = [sh for sh in swing_highs if left_idx < sh[0] < right_idx]
                        
                        if len(peaks_between) >= 2:
                            # Calculate neckline (connecting the peaks)
                            neckline_left = peaks_between[0]
                            neckline_right = peaks_between[-1]
                            
                            # Calculate neckline slope
                            neckline_slope = (neckline_right[1] - neckline_left[1]) / (neckline_right[0] - neckline_left[0])
                            
                            # Calculate neckline at current bar
                            current_idx = len(df) - 1
                            neckline_current = neckline_left[1] + neckline_slope * (current_idx - neckline_left[0])
                            
                            # Calculate pattern height
                            pattern_height = ((neckline_left[1] + neckline_right[1]) / 2) - head[1]
                            
                            # Check if price has broken the neckline
                            current_price = df['close'].iloc[-1]
                            completion = 1.0 if current_price > neckline_current else 0.5
                            
                            # Calculate target
                            target = neckline_current + pattern_height
                            
                            # Calculate reliability based on pattern quality
                            shoulder_symmetry = 1 - shoulder_diff  # 0.8 to 1.0
                            height_factor = pattern_height / df['close'].mean() * 10  # Normalize by price
                            reliability = min(0.9, (shoulder_symmetry * 0.5 + min(height_factor, 0.5)) * completion)
                            
                            result = {
                                "detected": True,
                                "type": "bullish",
                                "pattern": "inverse_head_and_shoulders",
                                "reliability": reliability,
                                "completion": completion,
                                "neckline": neckline_current,
                                "target": target,
                                "pattern_height": pattern_height,
                                "head": head[1],
                                "left_shoulder": left[1],
                                "right_shoulder": right[1]
                            }
                            
                            return result
        
        return result
        
    except Exception as e:
        logging.error(f"Error detecting head and shoulders pattern: {e}")
        return result

def detect_double_formations(df, swing_highs, swing_lows):
    """
    Detect Double Top and Double Bottom patterns.
    
    Returns:
        dict: Pattern details including type, reliability, and price targets
    """
    result = {"detected": False}
    
    try:
        # Check for double top
        if len(swing_highs) >= 2:
            # Look at last 2 significant swing highs
            peak1 = swing_highs[-2]
            peak2 = swing_highs[-1]
            
            # Peaks should be similar height (within 3%)
            peak_diff = abs(peak1[1] - peak2[1]) / peak1[1]
            
            if peak_diff < 0.03:
                # Find the trough between peaks
                trough = None
                for sl in swing_lows:
                    if peak1[0] < sl[0] < peak2[0]:
                        trough = sl
                        break
                        
                if trough:
                    # Calculate pattern height
                    pattern_height = ((peak1[1] + peak2[1]) / 2) - trough[1]
                    
                    # Check distance between peaks (should be sufficient)
                    time_diff = abs(peak2[0] - peak1[0])
                    if time_diff >= 10:  # At least 10 bars apart
                        # Check if price has broken the neckline (trough level)
                        current_price = df['close'].iloc[-1]
                        completion = 1.0 if current_price < trough[1] else 0.4
                        
                        # Calculate target
                        target = trough[1] - pattern_height
                        
                        # Calculate reliability
                        reliability = min(0.85, (1 - peak_diff * 10) * completion)
                        
                        # Only return if pattern is somewhat complete
                        if completion > 0.3:
                            result = {
                                "detected": True,
                                "type": "bearish",
                                "pattern": "double_top",
                                "reliability": reliability,
                                "completion": completion,
                                "neckline": trough[1],
                                "target": target,
                                "pattern_height": pattern_height,
                                "peak1": peak1[1],
                                "peak2": peak2[1],
                                "trough": trough[1]
                            }
                            
                            return result
        
        # Check for double bottom
        if len(swing_lows) >= 2:
            # Look at last 2 significant swing lows
            trough1 = swing_lows[-2]
            trough2 = swing_lows[-1]
            
            # Troughs should be similar height (within 3%)
            trough_diff = abs(trough1[1] - trough2[1]) / trough1[1]
            
            if trough_diff < 0.03:
                # Find the peak between troughs
                peak = None
                for sh in swing_highs:
                    if trough1[0] < sh[0] < trough2[0]:
                        peak = sh
                        break
                        
                if peak:
                    # Calculate pattern height
                    pattern_height = peak[1] - ((trough1[1] + trough2[1]) / 2)
                    
                    # Check distance between troughs (should be sufficient)
                    time_diff = abs(trough2[0] - trough1[0])
                    if time_diff >= 10:  # At least 10 bars apart
                        # Check if price has broken the neckline (peak level)
                        current_price = df['close'].iloc[-1]
                        completion = 1.0 if current_price > peak[1] else 0.4
                        
                        # Calculate target
                        target = peak[1] + pattern_height
                        
                        # Calculate reliability
                        reliability = min(0.85, (1 - trough_diff * 10) * completion)
                        
                        # Only return if pattern is somewhat complete
                        if completion > 0.3:
                            result = {
                                "detected": True,
                                "type": "bullish",
                                "pattern": "double_bottom",
                                "reliability": reliability,
                                "completion": completion,
                                "neckline": peak[1],
                                "target": target,
                                "pattern_height": pattern_height,
                                "trough1": trough1[1],
                                "trough2": trough2[1],
                                "peak": peak[1]
                            }
                            
                            return result
        
        return result
        
    except Exception as e:
        logging.error(f"Error detecting double formations: {e}")
        return result

def detect_triple_formations(df, swing_highs, swing_lows):
    """
    Detect Triple Top and Triple Bottom patterns.
    
    Returns:
        dict: Pattern details including type, reliability, and price targets
    """
    result = {"detected": False}
    
    try:
        # Check for triple top
        if len(swing_highs) >= 3:
            # Look at last 3 significant swing highs
            peak1 = swing_highs[-3]
            peak2 = swing_highs[-2]
            peak3 = swing_highs[-1]
            
            # Peaks should be similar height (within 5%)
            max_peak = max(peak1[1], peak2[1], peak3[1])
            min_peak = min(peak1[1], peak2[1], peak3[1])
            peak_diff = (max_peak - min_peak) / min_peak
            
            if peak_diff < 0.05:
                # Find the troughs between peaks
                trough1 = None
                trough2 = None
                
                for sl in swing_lows:
                    if peak1[0] < sl[0] < peak2[0]:
                        trough1 = sl
                    elif peak2[0] < sl[0] < peak3[0]:
                        trough2 = sl
                
                if trough1 and trough2:
                    # Troughs should be at similar levels
                    trough_diff = abs(trough1[1] - trough2[1]) / trough1[1]
                    
                    if trough_diff < 0.1:  # Within 10%
                        # Pattern height
                        avg_peak = (peak1[1] + peak2[1] + peak3[1]) / 3
                        avg_trough = (trough1[1] + trough2[1]) / 2
                        pattern_height = avg_peak - avg_trough
                        
                        # Check if pattern is well-formed (sufficient width)
                        pattern_width = peak3[0] - peak1[0]
                        if pattern_width >= 20:  # At least 20 bars wide
                            # Check if price has broken the neckline
                            neckline = avg_trough
                            current_price = df['close'].iloc[-1]
                            completion = 1.0 if current_price < neckline else 0.4
                            
                            # Calculate target
                            target = neckline - pattern_height
                            
                            # Calculate reliability (triple tops are more reliable than double)
                            reliability = min(0.9, (1 - peak_diff * 5) * completion)
                            
                            # Only return if pattern is somewhat complete
                            if completion > 0.3:
                                result = {
                                    "detected": True,
                                    "type": "bearish",
                                    "pattern": "triple_top",
                                    "reliability": reliability,
                                    "completion": completion,
                                    "neckline": neckline,
                                    "target": target,
                                    "pattern_height": pattern_height,
                                    "peak1": peak1[1],
                                    "peak2": peak2[1],
                                    "peak3": peak3[1]
                                }
                                
                                return result
        
        # Check for triple bottom
        if len(swing_lows) >= 3:
            # Look at last 3 significant swing lows
            trough1 = swing_lows[-3]
            trough2 = swing_lows[-2]
            trough3 = swing_lows[-1]
            
            # Troughs should be similar height (within 5%)
            max_trough = max(trough1[1], trough2[1], trough3[1])
            min_trough = min(trough1[1], trough2[1], trough3[1])
            trough_diff = (max_trough - min_trough) / min_trough
            
            if trough_diff < 0.05:
                # Find the peaks between troughs
                peak1 = None
                peak2 = None
                
                for sh in swing_highs:
                    if trough1[0] < sh[0] < trough2[0]:
                        peak1 = sh
                    elif trough2[0] < sh[0] < trough3[0]:
                        peak2 = sh
                
                if peak1 and peak2:
                    # Peaks should be at similar levels
                    peak_diff = abs(peak1[1] - peak2[1]) / peak1[1]
                    
                    if peak_diff < 0.1:  # Within 10%
                        # Pattern height
                        avg_trough = (trough1[1] + trough2[1] + trough3[1]) / 3
                        avg_peak = (peak1[1] + peak2[1]) / 2
                        pattern_height = avg_peak - avg_trough
                        
                        # Check if pattern is well-formed (sufficient width)
                        pattern_width = trough3[0] - trough1[0]
                        if pattern_width >= 20:  # At least 20 bars wide
                            # Check if price has broken the neckline
                            neckline = avg_peak
                            current_price = df['close'].iloc[-1]
                            completion = 1.0 if current_price > neckline else 0.4
                            
                            # Calculate target
                            target = neckline + pattern_height
                            
                            # Calculate reliability (triple bottoms are more reliable than double)
                            reliability = min(0.9, (1 - trough_diff * 5) * completion)
                            
                            # Only return if pattern is somewhat complete
                            if completion > 0.3:
                                result = {
                                    "detected": True,
                                    "type": "bullish",
                                    "pattern": "triple_bottom",
                                    "reliability": reliability,
                                    "completion": completion,
                                    "neckline": neckline,
                                    "target": target,
                                    "pattern_height": pattern_height,
                                    "trough1": trough1[1],
                                    "trough2": trough2[1],
                                    "trough3": trough3[1]
                                }
                                
                                return result
        
        return result
        
    except Exception as e:
        logging.error(f"Error detecting triple formations: {e}")
        return result

def detect_wedge_patterns(df, swing_highs, swing_lows):
    """
    Detect Rising and Falling Wedge patterns.
    
    Returns:
        dict: Pattern details including type, reliability, and price targets
    """
    result = {"detected": False}
    
    try:
        # Need at least 3 swing highs and 3 swing lows for a wedge
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return result
            
        # Take the most recent 4 swing points (or less if not available)
        recent_highs = swing_highs[-4:] if len(swing_highs) >= 4 else swing_highs[-(len(swing_highs)):]
        recent_lows = swing_lows[-4:] if len(swing_lows) >= 4 else swing_lows[-(len(swing_lows)):]
        
        # Need at least 3 points to define a trendline
        if len(recent_highs) < 3 or len(recent_lows) < 3:
            return result
            
        # Fit upper and lower trendlines
        x_highs = [p[0] for p in recent_highs]
        y_highs = [p[1] for p in recent_highs]
        
        x_lows = [p[0] for p in recent_lows]
        y_lows = [p[1] for p in recent_lows]
        
        # Calculate trendline slopes
        upper_slope = calculate_linear_regression_slope(x_highs, y_highs)
        lower_slope = calculate_linear_regression_slope(x_lows, y_lows)
        
        # For wedge patterns, slopes should be in the same direction but converging
        if (upper_slope > 0 and lower_slope > 0 and upper_slope < lower_slope) or \
           (upper_slope < 0 and lower_slope < 0 and upper_slope < lower_slope):
            
            # Check if pattern is converging sufficiently
            convergence_ratio = abs(upper_slope / lower_slope) if lower_slope != 0 else 0
            
            if 0.5 < convergence_ratio < 0.95:  # Significant convergence
                # Calculate where the wedge apex would be
                upper_intercept = y_highs[0] - upper_slope * x_highs[0]
                lower_intercept = y_lows[0] - lower_slope * x_lows[0]
                
                apex_x = (lower_intercept - upper_intercept) / (upper_slope - lower_slope)
                apex_y = upper_slope * apex_x + upper_intercept
                
                # Check if the wedge is well-formed (not too wide or narrow)
                pattern_width = apex_x - min(min(x_highs), min(x_lows))
                
                if pattern_width > 10 and pattern_width < 100:  # Reasonable width
                    # Determine pattern type
                    if upper_slope < 0 and lower_slope < 0:
                        pattern_type = "falling_wedge"
                        expected_direction = "bullish"
                    else:  # upper_slope > 0 and lower_slope > 0
                        pattern_type = "rising_wedge"
                        expected_direction = "bearish"
                    
                    # Calculate current position in the wedge
                    current_idx = len(df) - 1
                    upper_current = upper_slope * current_idx + upper_intercept
                    lower_current = lower_slope * current_idx + lower_intercept
                    current_price = df['close'].iloc[-1]
                    
                    # Calculate how far the price is in the wedge (0-1, where 1 is at apex)
                    wedge_height = upper_current - lower_current
                    position_in_wedge = (apex_x - current_idx) / pattern_width
                    
                    # Check for breakout
                    breakout = False
                    completion = 0.5
                    
                    if expected_direction == "bullish" and current_price > upper_current:
                        breakout = True
                        completion = 1.0
                    elif expected_direction == "bearish" and current_price < lower_current:
                        breakout = True
                        completion = 1.0
                    
                    # Calculate target (conservative: height of the pattern at breakout)
                    pattern_height = wedge_height
                    if expected_direction == "bullish":
                        target = upper_current + pattern_height
                    else:
                        target = lower_current - pattern_height
                    
                    # Calculate reliability
                    if position_in_wedge < 0.7:  # Not too close to apex
                        reliability = min(0.9, 0.6 + (0.3 * completion))
                    else:
                        # Less reliable when price is close to the apex
                        reliability = min(0.7, 0.4 + (0.3 * completion))
                    
                    result = {
                        "detected": True,
                        "type": expected_direction,
                        "pattern": pattern_type,
                        "reliability": reliability,
                        "completion": completion,
                        "breakout": breakout,
                        "target": target,
                        "upper_slope": upper_slope,
                        "lower_slope": lower_slope,
                        "apex_x": apex_x,
                        "apex_y": apex_y,
                        "pattern_height": pattern_height,
                        "upper_current": upper_current,
                        "lower_current": lower_current
                    }
                    
                    return result
        
        return result
        
    except Exception as e:
        logging.error(f"Error detecting wedge patterns: {e}")
        return result

def calculate_linear_regression_slope(x, y):
    """
    Calculate the slope of a linear regression line.
    
    Returns:
        float: Slope of the regression line
    """
    try:
        n = len(x)
        if n < 2:
            return 0
            
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_xx = sum(x[i] * x[i] for i in range(n))
        
        # Calculate denominator and check for division by zero
        denominator = (n * sum_xx - sum_x * sum_x)
        if denominator == 0:
            logging.warning("Division by zero prevented in linear regression slope calculation")
            return 0

        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
        
    except Exception as e:
        logging.error(f"Error calculating regression slope: {e}")
        return 0

def detect_flag_patterns(df, swing_highs, swing_lows):
    """
    Detect Flag and Pennant patterns.
    
    Returns:
        dict: Pattern details including type, reliability, and price targets
    """
    result = {"detected": False}
    
    try:
        # Need recent price history for the flag pole
        if len(df) < 30:
            return result
        
        # Look for a strong preceding move (flag pole)
        lookback = min(30, len(df) - 1)
        
        # Calculate recent price movement
        recent_high = df['high'].iloc[-lookback:].max()
        recent_low = df['low'].iloc[-lookback:].min()
        recent_move = recent_high - recent_low
        
        # Calculate the average candle size
        avg_candle_size = ((df['high'] - df['low']).sum() / len(df))
        
        # Flag pole should be a significant move
        if recent_move < 5 * avg_candle_size:
            return result
        
        # Determine if the pole is bullish or bearish
        pole_start_idx = df['low'].iloc[-lookback:].idxmin() if df['close'].iloc[-1] > df['open'].iloc[-lookback] else df['high'].iloc[-lookback:].idxmax()
        pole_end_idx = df['high'].iloc[-lookback:].idxmax() if df['close'].iloc[-1] > df['open'].iloc[-lookback] else df['low'].iloc[-lookback:].idxmin()
        
        # Ensure the pole moves in the right direction
        if (pole_end_idx <= pole_start_idx) or abs(pole_end_idx - pole_start_idx) < 3:
            return result
        
        # Check for consolidation after the pole (the flag part)
        consolidation_start = pole_end_idx
        consolidation_length = len(df) - 1 - consolidation_start
        
        # Need at least a few bars of consolidation
        if consolidation_length < 5:
            return result
        
        # Calculate the range of the consolidation
        consolidation_high = df['high'].iloc[consolidation_start:].max()
        consolidation_low = df['low'].iloc[consolidation_start:].min()
        consolidation_range = consolidation_high - consolidation_low
        
        # Consolidation should be smaller than the pole
        if consolidation_range > recent_move * 0.5:
            return result
        
        # Determine if it's a bull flag or bear flag
        is_bull_flag = df['close'].iloc[pole_end_idx] > df['close'].iloc[pole_start_idx]
        
        # Calculate upper and lower boundaries of the flag
        consolidation_highs = df['high'].iloc[consolidation_start:]
        consolidation_lows = df['low'].iloc[consolidation_start:]
        
        # Fit trendlines to the consolidation
        x_values = range(len(consolidation_highs))
        upper_slope = calculate_linear_regression_slope(x_values, consolidation_highs)
        lower_slope = calculate_linear_regression_slope(x_values, consolidation_lows)
        
        # For flags, the slopes should be similar (parallel or nearly parallel)
        slope_diff = abs(upper_slope - lower_slope)
        avg_slope = (abs(upper_slope) + abs(lower_slope)) / 2
        
        is_flag = slope_diff < avg_slope * 0.3  # Slopes within 30% of each other
        is_pennant = not is_flag and slope_diff < avg_slope * 0.8  # Converging but not parallel
        
        if not (is_flag or is_pennant):
            return result
        
        # Check for breakout
        current_price = df['close'].iloc[-1]
        
        # Calculate the breakout level
        if is_bull_flag:
            upper_boundary = consolidation_highs.iloc[0] + upper_slope * (len(consolidation_highs) - 1)
            breakout_level = upper_boundary
            breakout = current_price > breakout_level
        else:  # Bear flag
            lower_boundary = consolidation_lows.iloc[0] + lower_slope * (len(consolidation_lows) - 1)
            breakout_level = lower_boundary
            breakout = current_price < breakout_level
        
        # Calculate completion based on breakout
        completion = 1.0 if breakout else 0.6
        
        # Calculate target (conservative: the length of the pole)
        pole_height = abs(df['close'].iloc[pole_end_idx] - df['close'].iloc[pole_start_idx])
        
        if is_bull_flag:
            target = breakout_level + pole_height
            pattern_type = "bullish"
        else:
            target = breakout_level - pole_height
            pattern_type = "bearish"
        
        # Calculate reliability
        pattern_name = "flag" if is_flag else "pennant"
        
        # Flags and pennants are among the most reliable continuation patterns
        reliability = min(0.85, 0.7 + (0.15 * completion))
        
        result = {
            "detected": True,
            "type": pattern_type,
            "pattern": pattern_name,
            "reliability": reliability,
            "completion": completion,
            "breakout": breakout,
            "target": target,
            "pole_height": pole_height,
            "consolidation_range": consolidation_range,
            "breakout_level": breakout_level
        }
        
        return result
        
    except Exception as e:
        logging.error(f"Error detecting flag patterns: {e}")
        return result

def detect_rounding_patterns(df):
    """
    Detect Rounding Bottom (Cup) and Rounding Top patterns.
    
    Returns:
        dict: Pattern details including type, reliability, and price targets
    """
    result = {"detected": False}
    
    try:
        # Need sufficient data for rounding patterns
        if len(df) < 50:
            return result
        
        # Cup and Handle pattern (Rounding Bottom)
        # Step 1: Analyze price curve for curvature
        
        # Use a subset of the data (last 50 bars)
        subset_length = min(50, len(df) - 1)
        price_subset = df['close'].iloc[-subset_length:].values
        
        # Normalize the data to 0-1 range for easier analysis
        normalized_prices = (price_subset - min(price_subset)) / (max(price_subset) - min(price_subset))
        
        # Create x values (time indices)
        x_values = np.array(range(len(normalized_prices)))
        normalized_x = x_values / max(x_values)
        
        # Fit a quadratic curve to check for rounding pattern
        try:
            coeffs = np.polyfit(normalized_x, normalized_prices, 2)
            
            # Check the sign of the quadratic term (a in ax^2 + bx + c)
            # Positive a means upward curve (cup), negative means downward (inverse cup)
            curvature = coeffs[0]
            
            # Linear component strength
            linear_strength = abs(coeffs[1]) / (abs(coeffs[0]) + abs(coeffs[1]) + abs(coeffs[2]))
            
            # For a good cup, we want strong curvature and moderate linear component
            if abs(curvature) > 0.5 and linear_strength < 0.7:
                # Determine if it's a cup or an inverted cup
                if curvature > 0:  # Cup (rounding bottom)
                    pattern_type = "bullish"
                    pattern_name = "rounding_bottom"
                else:  # Inverted cup (rounding top)
                    pattern_type = "bearish"
                    pattern_name = "rounding_top"
                
                # Calculate the fit quality (how well the curve matches)
                p = np.poly1d(coeffs)
                fitted_values = p(normalized_x)
                residuals = normalized_prices - fitted_values
                fit_quality = 1 - (np.std(residuals) / np.std(normalized_prices))
                
                # Need good fit for a reliable pattern
                if fit_quality > 0.7:
                    # Calculate where we are in the pattern
                    # For a cup, we want to be near the right lip
                    # For an inverted cup, we want to be breaking below the right lip
                    
                    # Find the price level of the "lips" (start and end of the pattern)
                    left_lip = price_subset[0]
                    right_lip = price_subset[-1]
                    
                    # Compare current price to lip level
                    current_price = df['close'].iloc[-1]
                    
                    # Calculate completion and check for breakout
                    lip_diff = abs(right_lip - left_lip) / left_lip
                    
                    if pattern_type == "bullish":
                        # For a cup, breakout is above the lip
                        breakout = current_price > max(left_lip, right_lip)
                        # Completion is how far we've come up the right side
                        bottom_idx = np.argmin(price_subset)
                        completion = min(1.0, (normalized_prices[-1] - normalized_prices[bottom_idx]) / 
                                             (1 - normalized_prices[bottom_idx]))
                        
                        # Pattern is more reliable if lips are at similar levels
                        reliability = min(0.9, fit_quality * (1 - min(0.5, lip_diff)))
                        
                        # Calculate target (conservative: height of the cup)
                        cup_depth = max(left_lip, right_lip) - min(price_subset)
                        target = max(left_lip, right_lip) + cup_depth
                        
                    else:  # "bearish"
                        # For an inverted cup, breakout is below the lip
                        breakout = current_price < min(left_lip, right_lip)
                        # Completion is how far we've come down the right side
                        top_idx = np.argmax(price_subset)
                        completion = min(1.0, (normalized_prices[top_idx] - normalized_prices[-1]) / 
                                             normalized_prices[top_idx])
                        
                        # Pattern is more reliable if lips are at similar levels
                        reliability = min(0.9, fit_quality * (1 - min(0.5, lip_diff)))
                        
                        # Calculate target (conservative: height of the inverted cup)
                        cup_height = max(price_subset) - min(left_lip, right_lip)
                        target = min(left_lip, right_lip) - cup_height
                    
                    # Adjust completion based on breakout
                    if breakout:
                        completion = 1.0
                    
                    # Adjust reliability based on completion
                    reliability *= (0.7 + 0.3 * completion)
                    
                    result = {
                        "detected": True,
                        "type": pattern_type,
                        "pattern": pattern_name,
                        "reliability": reliability,
                        "completion": completion,
                        "breakout": breakout,
                        "target": target,
                        "curvature": curvature,
                        "fit_quality": fit_quality,
                        "left_lip": left_lip,
                        "right_lip": right_lip
                    }
                    
                    return result
            
        except Exception as e:
            logging.error(f"Error fitting curve in rounding pattern detection: {e}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error detecting rounding patterns: {e}")
        return result

def detect_channel_patterns(df, swing_highs, swing_lows):
    """
    Detect Channel patterns (ascending, descending, horizontal).
    
    Returns:
        dict: Pattern details including type, reliability, and price targets
    """
    result = {"detected": False}
    
    try:
        # Need at least 3 swing highs and 3 swing lows for a well-defined channel
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return result
            
        # Use recent swing points
        recent_highs = swing_highs[-5:] if len(swing_highs) >= 5 else swing_highs
        recent_lows = swing_lows[-5:] if len(swing_lows) >= 5 else swing_lows
        
        # Need at least 3 points to define each boundary
        if len(recent_highs) < 3 or len(recent_lows) < 3:
            return result
            
        # Fit upper and lower trendlines
        x_highs = [p[0] for p in recent_highs]
        y_highs = [p[1] for p in recent_highs]
        
        x_lows = [p[0] for p in recent_lows]
        y_lows = [p[1] for p in recent_lows]
        
        # Calculate trendline slopes
        upper_slope = calculate_linear_regression_slope(x_highs, y_highs)
        lower_slope = calculate_linear_regression_slope(x_lows, y_lows)
        
        # For a channel, slopes should be similar (parallel)
        slope_diff = abs(upper_slope - lower_slope)
        avg_slope = (abs(upper_slope) + abs(lower_slope)) / 2
        
        # Allow for some tolerance in parallelism
        if slope_diff > avg_slope * 0.2:  # More than 20% difference in slopes
            return result
        
        # Calculate channel width
        upper_intercept = y_highs[0] - upper_slope * x_highs[0]
        lower_intercept = y_lows[0] - lower_slope * x_lows[0]
        
        # Calculate current upper and lower bounds
        current_idx = len(df) - 1
        upper_bound = upper_slope * current_idx + upper_intercept
        lower_bound = lower_slope * current_idx + lower_intercept
        
        channel_width = upper_bound - lower_bound
        
        # Check if the width is consistent
        if channel_width <= 0:
            return result
        
        # Determine channel type based on slope
        if abs(upper_slope) < 0.0001:  # Very close to horizontal
            channel_type = "horizontal_channel"
            expected_breakout_direction = "neutral"  # Could break either way
        elif upper_slope > 0:
            channel_type = "ascending_channel"
            expected_breakout_direction = "bullish"  # Upside breakout is more likely
        else:
            channel_type = "descending_channel"
            expected_breakout_direction = "bearish"  # Downside breakout is more likely
        
        # Check where the price is within the channel
        current_price = df['close'].iloc[-1]
        channel_position = (current_price - lower_bound) / channel_width
        
        # Check for breakout
        breakout = False
        breakout_direction = "none"
        
        if current_price > upper_bound:
            breakout = True
            breakout_direction = "bullish"
        elif current_price < lower_bound:
            breakout = True
            breakout_direction = "bearish"
        
        # If there's a breakout, the signal direction matches the breakout
        if breakout:
            pattern_type = breakout_direction
            completion = 1.0
        else:
            # No breakout yet - signal based on position within channel and channel type
            if channel_position > 0.8:  # Near upper bound
                if channel_type == "ascending_channel":
                    pattern_type = "bullish"
                else:
                    pattern_type = "bearish"
            elif channel_position < 0.2:  # Near lower bound
                if channel_type == "descending_channel":
                    pattern_type = "bearish"
                else:
                    pattern_type = "bullish"
            else:
                # In the middle of the channel - neutral
                pattern_type = "neutral"
            
            completion = 0.5
        
        # Calculate target based on channel width
        if pattern_type == "bullish":
            target = upper_bound + channel_width
        elif pattern_type == "bearish":
            target = lower_bound - channel_width
        else:
            # For neutral signals, use the opposite bound as target
            target = upper_bound if channel_position < 0.5 else lower_bound
        
        # Channels are reliable patterns, especially when broken
        reliability = min(0.85, 0.7 + (0.15 * completion))
        
        # Adjust reliability based on how well-defined the channel is
        # Check the R-squared of the regression lines
        r2_upper = calculate_r_squared(x_highs, y_highs, upper_slope, upper_intercept)
        r2_lower = calculate_r_squared(x_lows, y_lows, lower_slope, lower_intercept)
        
        # Channel quality is the average of the two R-squared values
        channel_quality = (r2_upper + r2_lower) / 2
        
        # Adjust reliability based on channel quality
        reliability *= channel_quality
        
        result = {
            "detected": True,
            "type": pattern_type,
            "pattern": channel_type,
            "reliability": reliability,
            "completion": completion,
            "breakout": breakout,
            "breakout_direction": breakout_direction,
            "target": target,
            "channel_width": channel_width,
            "channel_position": channel_position,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "channel_quality": channel_quality
        }
        
        return result
        
    except Exception as e:
        logging.error(f"Error detecting channel patterns: {e}")
        return result

def calculate_r_squared(x, y, slope, intercept):
    """
    Calculate the R-squared (coefficient of determination) for a linear regression.
    
    Returns:
        float: R-squared value (0-1)
    """
    try:
        if len(x) < 2:
            return 0
            
        # Calculate predictions
        y_pred = [slope * x_i + intercept for x_i in x]
        
        # Calculate sum of squares
        ss_total = sum((y_i - sum(y) / len(y)) ** 2 for y_i in y)
        ss_residual = sum((y_i - y_pred_i) ** 2 for y_i, y_pred_i in zip(y, y_pred))
        
        # Calculate R-squared
        if ss_total == 0:
            return 0
            
        r_squared = 1 - (ss_residual / ss_total)
        return max(0, min(1, r_squared))  # Ensure value is between 0 and 1
        
    except Exception as e:
        logging.error(f"Error calculating R-squared: {e}")
        return 0

def detect_rectangle_patterns(df, swing_highs, swing_lows):
    """
    Detect Rectangle patterns (trading ranges).
    
    Returns:
        dict: Pattern details including type, reliability, and price targets
    """
    result = {"detected": False}
    
    try:
        # Need sufficient data
        if len(df) < 20:
            return result
            
        # Need at least 2 swing highs and 2 swing lows for a rectangle
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return result
            
        # Use recent swing points
        recent_highs = swing_highs[-4:] if len(swing_highs) >= 4 else swing_highs
        recent_lows = swing_lows[-4:] if len(swing_lows) >= 4 else swing_lows
        
        # Need at least 2 points for each boundary
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return result
            
        # Check if swing highs are at similar levels (resistance)
        high_prices = [p[1] for p in recent_highs]
        max_high = max(high_prices)
        min_high = min(high_prices)
        high_diff = (max_high - min_high) / min_high
        
        # Check if swing lows are at similar levels (support)
        low_prices = [p[1] for p in recent_lows]
        max_low = max(low_prices)
        min_low = min(low_prices)
        low_diff = (max_low - min_low) / min_low
        
        # For a good rectangle, both tops and bottoms should be well-aligned
        if high_diff > 0.02 or low_diff > 0.02:  # More than 2% variation
            return result
        
        # Calculate rectangle boundaries
        resistance = sum(high_prices) / len(high_prices)
        support = sum(low_prices) / len(low_prices)
        
        # Rectangle height
        rectangle_height = resistance - support
        
        # Rectangle should have meaningful height
        avg_price = df['close'].mean()
        min_height = avg_price * 0.01  # At least 1% of price
        
        if rectangle_height < min_height:
            return result
        
        # Check rectangle width (duration)
        earliest_point = min(min(p[0] for p in recent_highs), min(p[0] for p in recent_lows))
        latest_point = max(max(p[0] for p in recent_highs), max(p[0] for p in recent_lows))
        rectangle_width = latest_point - earliest_point
        
        # Rectangle should have sufficient width (at least 15 bars)
        if rectangle_width < 15:
            return result
        
        # Check where the price is within the rectangle
        current_price = df['close'].iloc[-1]
        position_in_rectangle = (current_price - support) / rectangle_height
        
        # Check for breakout
        breakout = False
        breakout_direction = "none"
        
        if current_price > resistance * 1.01:  # 1% above resistance
            breakout = True
            breakout_direction = "bullish"
        elif current_price < support * 0.99:  # 1% below support
            breakout = True
            breakout_direction = "bearish"
        
        # Determine pattern type based on breakout
        if breakout:
            pattern_type = breakout_direction
            completion = 1.0
        else:
            # No breakout yet - signal based on position within rectangle
            if position_in_rectangle > 0.8:  # Near resistance
                pattern_type = "bearish"  # Potential resistance rejection
            elif position_in_rectangle < 0.2:  # Near support
                pattern_type = "bullish"  # Potential support bounce
            else:
                # In the middle of the rectangle - neutral
                pattern_type = "neutral"
            
            completion = 0.5
        
        # Calculate target based on rectangle height
        if breakout_direction == "bullish":
            target = resistance + rectangle_height
        elif breakout_direction == "bearish":
            target = support - rectangle_height
        else:
            # No breakout yet - target is the opposite boundary
            target = resistance if position_in_rectangle < 0.5 else support
        
        # Calculate reliability
        reliability = min(0.85, 0.6 + (0.25 * completion))
        
        # Adjust based on the quality of the rectangle
        quality_factor = 1 - (high_diff + low_diff) / 2
        reliability *= quality_factor
        
        result = {
            "detected": True,
            "type": pattern_type,
            "pattern": "rectangle",
            "reliability": reliability,
            "completion": completion,
            "breakout": breakout,
            "breakout_direction": breakout_direction,
            "target": target,
            "rectangle_height": rectangle_height,
            "rectangle_width": rectangle_width,
            "resistance": resistance,
            "support": support,
            "position_in_rectangle": position_in_rectangle
        }
        
        return result
        
    except Exception as e:
        logging.error(f"Error detecting rectangle patterns: {e}")
        return result

def detect_island_reversals(df):
    """
    Detect Island Reversal patterns.
    
    Returns:
        dict: Pattern details including type, reliability, and price targets
    """
    result = {"detected": False}
    
    try:
        # Need sufficient data
        if len(df) < 15:
            return result
        
        # Look for gaps in price
        gaps = []
        
        for i in range(1, len(df)):
            prev_close = df['close'].iloc[i-1]
            current_open = df['open'].iloc[i]
            
            # Calculate gap as percentage of price
            gap_pct = abs(current_open - prev_close) / prev_close
            
            # Consider gaps of at least 0.3%
            if gap_pct >= 0.003:
                gaps.append({
                    "index": i,
                    "prev_close": prev_close,
                    "current_open": current_open,
                    "direction": "up" if current_open > prev_close else "down",
                    "size": gap_pct
                })
        
        # Need at least 2 gaps to form an island
        if len(gaps) < 2:
            return result
        
        # Check for island reversals in the last 15 bars
        recent_gaps = [g for g in gaps if g["index"] > len(df) - 15]
        
        # Need at least 2 recent gaps
        if len(recent_gaps) < 2:
            return result
        
        # Check pairs of gaps for island formations
        for i in range(len(recent_gaps) - 1):
            first_gap = recent_gaps[i]
            
            # Look for an opposite gap within 7 bars
            for j in range(i + 1, min(i + 8, len(recent_gaps))):
                second_gap = recent_gaps[j]
                
                # Gaps should be in opposite directions
                if first_gap["direction"] == second_gap["direction"]:
                    continue
                
                # The island should be 2-7 bars wide
                island_width = second_gap["index"] - first_gap["index"]
                if island_width < 1 or island_width > 7:
                    continue
                
                # Extract price behavior in the island
                island_start = first_gap["index"]
                island_end = second_gap["index"]
                
                island_prices = df.iloc[island_start:island_end]
                
                # Determine pattern type
                if first_gap["direction"] == "up" and second_gap["direction"] == "down":
                    # Bullish island bottom (rare and typically weaker)
                    pattern_type = "bullish"
                    
                    # Check if prices dropped during the island formation
                    price_drop = island_prices['low'].min() < df['low'].iloc[island_start-1]
                    if not price_drop:
                        continue
                        
                elif first_gap["direction"] == "down" and second_gap["direction"] == "up":
                    # Bearish island top (more common)
                    pattern_type = "bearish"
                    
                    # Check if prices rose during the island formation
                    price_rise = island_prices['high'].max() > df['high'].iloc[island_start-1]
                    if not price_rise:
                        continue
                else:
                    continue
                
                # Check if we've had a confirmation candle after the second gap
                if island_end >= len(df) - 1:
                    # No confirmation yet
                    completion = 0.7
                else:
                    # Confirmation candle exists
                    completion = 1.0
                
                # Calculate target (conservative: size of the island formation)
                island_high = island_prices['high'].max()
                island_low = island_prices['low'].min()
                island_height = island_high - island_low
                
                current_price = df['close'].iloc[-1]
                
                if pattern_type == "bullish":
                    target = current_price + island_height
                else:
                    target = current_price - island_height
                
                # Calculate reliability (island reversals are moderately reliable)
                # Factors: gap size, island width, confirmation
                gap_factor = min(1.0, (first_gap["size"] + second_gap["size"]) * 50)  # Normalize gap size
                width_factor = 1 - abs(island_width - 3) / 5  # Optimal width around 3 bars
                
                reliability = min(0.8, 0.6 * gap_factor * width_factor * completion)
                
                result = {
                    "detected": True,
                    "type": pattern_type,
                    "pattern": "island_reversal",
                    "reliability": reliability,
                    "completion": completion,
                    "target": target,
                    "first_gap": f"{first_gap['size']:.2%}",
                    "second_gap": f"{second_gap['size']:.2%}",
                    "island_width": island_width,
                    "island_height": island_height
                }
                
                return result
        
        return result
        
    except Exception as e:
        logging.error(f"Error detecting island reversals: {e}")
        return result
    
import pickle
import os

def setup_ml_validation_system():
    """
    Initialize the machine learning signal validation system with model persistence.
    """
    try:
        # Initialize model storage if not exists
        if not hasattr(state, 'signal_validation_models'):
            state.signal_validation_models = {}
            
        # Initialize training data storage if not exists
        if not hasattr(state, 'signal_training_data'):
            state.signal_training_data = {}
            
        # Create models directory if it doesn't exist
        models_dir = 'ml_models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        # Load saved models if available
        load_ml_models()
            
        # Check if models need training
        check_and_train_validation_models()
        
        logging.info("ML signal validation system initialized")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize ML validation system: {e}")
        return False

def save_ml_models():
    """
    Save ML models to disk for persistence.
    """
    try:
        with state.locks['ml_models']:
            if not hasattr(state, 'signal_validation_models'):
                return
                
            models_dir = 'ml_models'
            for symbol, model in state.signal_validation_models.items():
                # Create a safe filename
                safe_symbol = symbol.replace('/', '_').replace('\\', '_')
                model_path = os.path.join(models_dir, f"{safe_symbol}_model.pkl")
                
                # Save the model
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                    
                # Save metadata separately (in case pickle fails on complete model)
                if hasattr(model, 'meta'):
                    meta_path = os.path.join(models_dir, f"{safe_symbol}_meta.pkl")
                    with open(meta_path, 'wb') as f:
                        pickle.dump(model.meta, f)
                        
            logging.info(f"Saved {len(state.signal_validation_models)} ML models to disk")
            
            # Save training data for critical symbols
            if hasattr(state, 'signal_training_data') and len(state.signal_training_data) > 0:
                training_data_path = os.path.join(models_dir, "training_data.pkl")
                with open(training_data_path, 'wb') as f:
                    # Limit the size to avoid huge files
                    limited_data = {}
                    for symbol, data in state.signal_training_data.items():
                        limited_data[symbol] = {
                            "features": data["features"][-200:],  # Keep last 200 samples
                            "outcomes": data["outcomes"][-200:]
                        }
                    pickle.dump(limited_data, f)
    except Exception as e:
        logging.error(f"Error saving ML models: {e}")

def load_ml_models():
    """
    Load ML models from disk.
    """
    try:
        with state.locks['ml_models']:
            models_dir = 'ml_models'
            if not os.path.exists(models_dir):
                return
                
            model_count = 0
            # Load each model file
            for filename in os.listdir(models_dir):
                if filename.endswith('_model.pkl'):
                    try:
                        symbol = filename.replace('_model.pkl', '').replace('_', '/')
                        model_path = os.path.join(models_dir, filename)
                        
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                            
                        # Try to load metadata if it exists
                        meta_path = os.path.join(models_dir, filename.replace('_model.pkl', '_meta.pkl'))
                        if os.path.exists(meta_path):
                            with open(meta_path, 'rb') as f:
                                model.meta = pickle.load(f)
                                
                        # Store the model
                        state.signal_validation_models[symbol] = model
                        model_count += 1
                    except Exception as e:
                        logging.error(f"Error loading model {filename}: {e}")
                        
            logging.info(f"Loaded {model_count} ML models from disk")
            
            # Load training data if available
            training_data_path = os.path.join(models_dir, "training_data.pkl")
            if os.path.exists(training_data_path):
                try:
                    with open(training_data_path, 'rb') as f:
                        state.signal_training_data = pickle.load(f)
                    logging.info(f"Loaded training data for {len(state.signal_training_data)} symbols")
                except Exception as e:
                    logging.error(f"Error loading training data: {e}")
    except Exception as e:
        logging.error(f"Error in load_ml_models: {e}")

def extract_signal_features(symbol, signal_data):
    """
    Extract features from trading signal data for ML validation.
    
    Args:
        symbol (str): The trading symbol
        signal_data (dict): Signal data from strategy
        
    Returns:
        list: Feature vector for ML model
    """
    features = []
    
    try:
        # Basic signal features
        signal_type = 1 if signal_data.get("signal") == "BUY" else 0
        signal_strength = float(signal_data.get("strength", 0))
        
        # Get market data
        df = get_candle_data(symbol, MT5_TIMEFRAMES["PRIMARY"], num_bars=100)
        if df is None:
            return []
            
        # Technical indicators
        rsi = calculate_rsi(df)
        adx, plus_di, minus_di = calculate_adx(df)
        bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(df)
        macd_line, signal_line, histogram = calculate_macd(df)
        
        # Volatility
        volatility_level, atr_percent = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
        volatility_encoding = {
            "low": 0,
            "normal": 1,
            "high": 2,
            "extreme": 3,
            "super-extreme": 4
        }.get(volatility_level, 1)
        
        # Market regime
        market_regime, regime_strength = detect_market_regime(symbol)
        regime_encoding = {
            "STRONG_UPTREND": 4,
            "STRONG_DOWNTREND": 0,
            "CHOPPY_VOLATILE": 1,
            "RANGE_BOUND": 2,
            "EMERGING_TREND": 3,
            "UNDEFINED": 2
        }.get(market_regime, 2)
        
        # Price relative to EMAs
        ema_short = calculate_ema(df, EMA_SHORT_PERIOD)
        ema_long = calculate_ema(df, EMA_LONG_PERIOD)
        
        current_price = df['close'].iloc[-1]
        price_vs_short_ema = (current_price / ema_short) - 1
        price_vs_long_ema = (current_price / ema_long) - 1
        
        # Momentum
        momentum_data = calculate_multi_timeframe_momentum(symbol)
        momentum_consensus = momentum_data.get("consensus", {}).get("momentum", "neutral")
        momentum_strength = momentum_data.get("consensus", {}).get("strength", 0)
        
        momentum_encoding = {
            "bullish": 2,
            "neutral": 1,
            "bearish": 0
        }.get(momentum_consensus, 1)
        
        # Pattern information
        pattern_data = detect_advanced_chart_patterns(symbol)
        pattern_direction = pattern_data.get("direction", "neutral")
        pattern_strength = pattern_data.get("strength", 0)
        
        pattern_encoding = {
            "bullish": 2,
            "neutral": 1,
            "bearish": 0
        }.get(pattern_direction, 1)
        
        # Assemble feature vector - keep order consistent!
        features = [
            signal_type,
            signal_strength,
            rsi / 100,  # Normalize to 0-1
            adx / 100,  # Normalize to 0-1
            plus_di / 100,  # Normalize to 0-1
            minus_di / 100,  # Normalize to 0-1
            bb_width,
            histogram,
            atr_percent / 10,  # Normalize large values
            volatility_encoding / 4,  # Normalize to 0-1
            regime_encoding / 4,  # Normalize to 0-1
            regime_strength,
            price_vs_short_ema,
            price_vs_long_ema,
            momentum_encoding / 2,  # Normalize to 0-1
            momentum_strength,
            pattern_encoding / 2,  # Normalize to 0-1
            pattern_strength
        ]
        
        return features
        
    except Exception as e:
        logging.error(f"Error extracting signal features: {e}")
        # Return a default feature vector with neutral values
        return [0, 0, 0.5, 0.25, 0.5, 0.5, 0.03, 0, 0.2, 0.25, 0.5, 0.5, 0, 0, 0.5, 0, 0.5, 0]

def collect_signal_results(symbol, signal_data, trade_result):
    """
    Collect training data for ML model by storing signal features and trade outcomes.
    
    Args:
        symbol (str): Trading symbol
        signal_data (dict): Signal data that generated the trade
        trade_result (dict): Result of the trade execution
    """
    try:
        # Extract features
        features = extract_signal_features(symbol, signal_data)
        if not features:
            return
            
        # Extract outcome (1 = profitable, 0 = unprofitable)
        profitable = 1 if trade_result.get("profit", 0) > 0 else 0
        
        # Store features and outcome
        if symbol not in state.signal_training_data:
            state.signal_training_data[symbol] = {"features": [], "outcomes": []}
            
        state.signal_training_data[symbol]["features"].append(features)
        state.signal_training_data[symbol]["outcomes"].append(profitable)
        
        # Limit data size to prevent memory issues
        max_samples = 1000
        if len(state.signal_training_data[symbol]["outcomes"]) > max_samples:
            state.signal_training_data[symbol]["features"] = state.signal_training_data[symbol]["features"][-max_samples:]
            state.signal_training_data[symbol]["outcomes"] = state.signal_training_data[symbol]["outcomes"][-max_samples:]
            
        logging.info(f"Collected ML training data for {symbol}: {profitable} outcome")
        
        # Check if we have enough data to train/update the model
        min_samples = 50
        if len(state.signal_training_data[symbol]["outcomes"]) >= min_samples:
            # Trigger model training if we have enough data
            train_validation_model(symbol)
            
    except Exception as e:
        logging.error(f"Error collecting signal results: {e}")

def train_validation_model(symbol):
    """
    Train a machine learning model to validate trading signals.
    Uses random forest classifier for robust performance with model persistence.
    
    Args:
        symbol (str): Trading symbol to train model for
    """
    try:
        # Check if we have enough data
        if symbol not in state.signal_training_data:
            logging.warning(f"No training data available for {symbol}")
            return False
            
        training_data = state.signal_training_data[symbol]
        if len(training_data["outcomes"]) < 30:
            logging.warning(f"Insufficient training data for {symbol} (need at least 30 samples)")
            return False
            
        # Check if scikit-learn is installed
        sklearn_available = True
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        except ImportError:
            sklearn_available = False
            logging.error("scikit-learn library not available. Please install with: pip install scikit-learn")
            return False
            
        if not sklearn_available:
            return False
            
        # Prepare data
        X = np.array(training_data["features"])
        y = np.array(training_data["outcomes"])
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        # Calculate average profit factor
        win_indices = np.where(y == 1)[0]
        loss_indices = np.where(y == 0)[0]
        
        win_rate = len(win_indices) / len(y) if len(y) > 0 else 0
        
        # Only save model if it's better than random guessing
        if accuracy > 0.55 and precision > 0.5:
            # Save metadata with the model
            model.meta = {
                "trained_on": len(training_data["outcomes"]),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "win_rate": win_rate,
                "avg_profit_factor": 1.5,  # Reasonable default
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Store the model
            if hasattr(state, 'locks') and 'ml_models' in state.locks:
                with state.locks['ml_models']:
                    state.signal_validation_models[symbol] = model
            else:
                state.signal_validation_models[symbol] = model
                
            # Save model to disk if save function exists
            if 'save_ml_models' in globals() and callable(globals()['save_ml_models']):
                save_ml_models()
            else:
                # Create models directory if it doesn't exist
                models_dir = 'ml_models'
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)
                    
                # Save the model using pickle
                import pickle
                model_path = os.path.join(models_dir, f"{symbol.replace('/', '_')}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logging.info(f"Saved model for {symbol} to {model_path}")
            
            logging.info(f"Trained ML validation model for {symbol}: " +
                       f"Accuracy={accuracy:.2f}, Precision={precision:.2f}, " +
                       f"Recall={recall:.2f}, F1={f1:.2f}")
            return True
        else:
            logging.warning(f"ML model for {symbol} not accurate enough: " +
                         f"Accuracy={accuracy:.2f}, Precision={precision:.2f}")
            return False
            
    except Exception as e:
        logging.error(f"Error training validation model: {e}")
        return False
    
def check_and_train_validation_models():
    """
    Check if validation models need training/updating and trigger training if needed.
    """
    try:
        # Check if we have any training data
        if not hasattr(state, 'signal_training_data') or not state.signal_training_data:
            logging.info("No training data available for ML models")
            return
            
        # Process each symbol with training data
        for symbol in state.signal_training_data:
            # Check if we have a model already
            model_exists = hasattr(state, 'signal_validation_models') and symbol in state.signal_validation_models
            
            # Check if we have enough new data to retrain
            training_data = state.signal_training_data[symbol]
            sample_count = len(training_data["outcomes"])
            
            if not model_exists and sample_count >= 50:
                # No model exists and we have enough data - train a new one
                train_validation_model(symbol)
            elif model_exists:
                # Model exists - check if it's time to update
                model = state.signal_validation_models[symbol]
                trained_on = model.meta.get("trained_on", 0) if hasattr(model, 'meta') else 0
                
                # Retrain if we have at least 20% more data
                if sample_count >= trained_on * 1.2 and sample_count - trained_on >= 20:
                    logging.info(f"Retraining ML model for {symbol}: {trained_on} → {sample_count} samples")
                    train_validation_model(symbol)
                    
    except Exception as e:
        logging.error(f"Error checking validation models: {e}")

def validate_trading_signal(symbol, signal_data):
    """
    Use ML model to validate a trading signal before execution.
    
    Args:
        symbol (str): Trading symbol
        signal_data (dict): Signal data to validate
        
    Returns:
        tuple: (valid, confidence) - Whether signal is valid and confidence level
    """
    try:
        # Check if we have a model for this symbol
        if not hasattr(state, 'signal_validation_models') or symbol not in state.signal_validation_models:
            logging.info(f"No ML model available for {symbol}, assuming signal is valid")
            return True, 1.0
            
        # Extract features from signal data
        features = extract_signal_features(symbol, signal_data)
        if not features:
            logging.warning(f"Couldn't extract features for {symbol} signal, assuming valid")
            return True, 1.0
            
        # Get prediction from model
        model = state.signal_validation_models[symbol]
        
        # Ensure features have the right shape
        features_array = np.array([features])
        
        # Get probability prediction
        try:
            predictions = model.predict_proba(features_array)
            # Class 1 probability (probability of a profitable trade)
            confidence = predictions[0][1] if len(predictions[0]) > 1 else 0.5
        except:
            # Fall back to binary prediction if probability not available
            prediction = model.predict(features_array)[0]
            confidence = 0.8 if prediction == 1 else 0.2
        
        # Evaluate confidence threshold
        threshold = 0.65  # Minimum confidence level
        valid = confidence >= threshold
        
        # Log the validation result
        if valid:
            logging.info(f"ML validation PASSED for {symbol} signal: {confidence:.2f} confidence")
        else:
            logging.info(f"ML validation REJECTED {symbol} signal: {confidence:.2f} confidence (below {threshold})")
        
        return valid, confidence
        
    except Exception as e:
        logging.error(f"Error validating signal with ML: {e}")
        # Default to accepting the signal if validation fails
        return True, 1.0

def shutdown_handler():
    """
    Proper shutdown handler to manage positions when the script is terminated
    """
    logging.info("Shutdown sequence initiated")
    
    try:
        # Get all open positions
        positions = get_positions()
        logging.info(f"Managing {len(positions)} open positions during shutdown")
        
        protected_count = 0
        
        for position in positions:
            position_id = position.ticket
            symbol = position.symbol
            position_type = position.type
            entry_price = position.price_open
            
            try:
                # Get current market data
                current_price = mt5.symbol_info_tick(symbol).bid if position_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
                
                # Ensure all positions have stop losses
                if position.sl is None:
                    # Calculate a reasonable stop based on current price and position type
                    if position_type == mt5.ORDER_TYPE_BUY:
                        # For long positions, set stop 2% below current price
                        stop_price = current_price * 0.98
                        # But not below 5% of entry price (catastrophic stop)
                        min_stop = entry_price * 0.95
                        stop_price = max(stop_price, min_stop)
                    else:  # mt5.ORDER_TYPE_SELL
                        # For short positions, set stop 2% above current price
                        stop_price = current_price * 1.02
                        # But not above 5% of entry price (catastrophic stop)
                        max_stop = entry_price * 1.05
                        stop_price = min(stop_price, max_stop)
                    
                    # Round to appropriate precision
                    digits = get_symbol_info(symbol).digits
                    stop_price = round(stop_price, digits)
                    
                    # Set the stop loss
                    update_position_stops(symbol, position_id, stop_price)
                    logging.info(f"Position {position_id}: Added safety stop loss at {stop_price:.2f} during shutdown")
                    protected_count += 1
                else:
                    logging.info(f"Position {position_id}: Already has stop loss at {position.sl}")
                
                # Optional: Tighten existing stops for safer shutdown
                if position.sl is not None:
                    if position_type == mt5.ORDER_TYPE_BUY and current_price > position.sl * 1.02:
                        # Current price is at least 2% above stop - consider tightening
                        new_stop = current_price * 0.99  # Set stop 1% below current
                        update_position_stops(symbol, position_id, new_stop)
                        logging.info(f"Position {position_id}: Tightened stop loss to {new_stop:.2f} during shutdown")
                        protected_count += 1
                    elif position_type == mt5.ORDER_TYPE_SELL and current_price < position.sl * 0.98:
                        # Current price is at least 2% below stop - consider tightening
                        new_stop = current_price * 1.01  # Set stop 1% above current
                        update_position_stops(symbol, position_id, new_stop)
                        logging.info(f"Position {position_id}: Tightened stop loss to {new_stop:.2f} during shutdown")
                        protected_count += 1
                        
            except Exception as e:
                logging.error(f"Error protecting position {position_id} during shutdown: {str(e)}")
                
        logging.info(f"All positions protected with stop losses for safe shutdown ({protected_count} updated)")
        
    except Exception as e:
        logging.error(f"Error during shutdown sequence: {str(e)}")
    
    logging.info("Shutdown sequence completed")

def detect_emergency_conditions(symbol=None):
    """
    Detect emergency market conditions that might warrant special position protection.
    Returns a tuple: (emergency_detected, emergency_type, details)
    """
    try:
        # 1. Check for extreme volatility
        if symbol:
            # Check individual symbol
            volatility_level, atr_percent = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
            
            if volatility_level == "extreme":
                # Check for sudden price movements
                sudden_movement, price_range_pct = detect_sudden_price_movement(symbol, time_window=3)  # 3-minute window
                
                if sudden_movement and price_range_pct > 2.5:  # Very large move
                    return True, "extreme_volatility", {
                        "symbol": symbol,
                        "volatility": volatility_level,
                        "price_range_pct": price_range_pct
                    }
        else:
            # Check all symbols with open positions
            positions = get_positions()
            for position in positions:
                emergency, emergency_type, details = detect_emergency_conditions(position.symbol)
                if emergency:
                    return True, emergency_type, details
            
        # 2. Check for critical margin level
        account_info = get_account_info()
        if account_info and account_info.margin_level is not None and account_info.margin > 0:
            if account_info.margin_level < 150:  # Critically low margin level
                return True, "critical_margin", {
                    "margin_level": account_info.margin_level,
                    "margin": account_info.margin,
                    "equity": account_info.equity
                }
        
        # 3. Check for extreme drawdown
        if state.initial_balance > 0 and account_info:
            current_drawdown = (state.initial_balance - account_info.equity) / state.initial_balance
            if current_drawdown > MAX_DRAWDOWN_PER_DAY * 0.9:  # Near maximum allowed drawdown
                return True, "extreme_drawdown", {
                    "drawdown_percent": current_drawdown * 100,
                    "initial_balance": state.initial_balance,
                    "current_equity": account_info.equity
                }
        
        # 4. Check for connection issues
        if not check_mt5_connection():
            return True, "connection_issues", {
                "details": "MT5 connection lost or unstable"
            }
            
        # No emergency detected
        return False, None, {}
        
    except Exception as e:
        logging.error(f"Error detecting emergency conditions: {e}")
        # Return emergency on error to be safe
        return True, "detection_error", {"error": str(e)}
    
def test_order_placement(symbol):
    """
    Test function to verify if basic order placement works for a particular symbol.
    This can be used to troubleshoot trading issues.
    """
    logging.info(f"Starting basic order placement test for {symbol}")
    
    # Get symbol info
    symbol_info = get_symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Symbol {symbol} not found or not available")
        return False
        
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.error(f"Failed to get price tick for {symbol}")
        return False
        
    # Prepare the most basic order possible
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": symbol_info.volume_min,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask
    }
    
    logging.info(f"Sending basic test order: {request}")
    result = mt5.order_send(request)
    
    if result is None:
        logging.error("Order send returned None, possible timeout or connection issue")
        return False
        
    logging.info(f"Order result: code={result.retcode}, comment={result.comment}")
    
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info(f"Test order successfully placed! Position #{result.order}")
        
        # Try to close the test position immediately
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": symbol_info.volume_min,
            "type": mt5.ORDER_TYPE_SELL,
            "position": result.order,
            "price": tick.bid
        }
        
        close_result = mt5.order_send(close_request)
        
        if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info("Test position successfully closed")
            return True
        else:
            logging.warning(f"Failed to close test position: {close_result.comment if close_result else 'Unknown error'}")
            logging.warning("You will need to manually close the test position")
            return True
    
    # Order failed, add detailed explanation
    error_descriptions = {
        10004: "Requote",
        10006: "Order rejected",
        10007: "Order canceled by client",
        10008: "Order already executed",
        10009: "Order already exists",
        10010: "Order conditions not met",
        10011: "Too many orders",
        10012: "Trade disabled",
        10013: "Market closed",
        10014: "Not enough money",
        10015: "Price changed",
        10016: "Price off",
        10017: "Invalid expiration",
        10018: "Order locked",
        10019: "Buy only allowed",
        10020: "Sell only allowed",
        10021: "Too many requests",
        10022: "Trade timeout"
    }
    
    if result.retcode in error_descriptions:
        logging.error(f"Test order failed: {error_descriptions[result.retcode]}")
    else:
        logging.error(f"Test order failed with code {result.retcode}: {result.comment}")
        
    return False

def retry_with_logging(func, *args, max_attempts=3, delay=2, **kwargs):
    """
    Retry a function with exponential backoff and detailed logging.
    
    Args:
        func: The function to retry
        *args: Positional arguments to pass to the function
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (will be doubled each retry)
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function call, or None if all attempts failed
    """
    for attempt in range(1, max_attempts + 1):
        try:
            logging.info(f"Attempt {attempt}/{max_attempts} for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Check if result indicates success
            if result is not None:
                if hasattr(result, 'retcode') and result.retcode != mt5.TRADE_RETCODE_DONE:
                    logging.warning(f"Attempt {attempt} returned error code: {result.retcode}, comment: {result.comment}")
                else:
                    # Success
                    if attempt > 1:
                        logging.info(f"Function {func.__name__} succeeded on attempt {attempt}")
                    return result
            else:
                logging.warning(f"Attempt {attempt} returned None result")
                
        except Exception as e:
            logging.error(f"Attempt {attempt} failed with error: {e}")
            logging.error(traceback.format_exc())
        
        # Don't sleep after the last attempt
        if attempt < max_attempts:
            # Exponential backoff
            wait_time = delay * (2 ** (attempt - 1))
            logging.info(f"Waiting {wait_time} seconds before next attempt")
            time.sleep(wait_time)
    
    logging.error(f"All {max_attempts} attempts failed for {func.__name__}")
    return None

def diagnose_trading_issues(symbol):
    """
    Run detailed diagnostics to identify potential trading issues.
    This function performs extensive checks and logs detailed information
    to help troubleshoot order placement failures.
    """
    try:
        logging.info(f"======= DIAGNOSTIC REPORT FOR {symbol} =======")
        
        # Check if MT5 is connected
        if mt5.terminal_info() is None:
            logging.error("DIAGNOSTIC: MT5 is not connected")
            return
            
        logging.info(f"DIAGNOSTIC: MT5 terminal connected: {mt5.terminal_info().connected}")
        logging.info(f"DIAGNOSTIC: MT5 trading allowed: {mt5.terminal_info().trade_allowed}")
        
        # Check symbol properties
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logging.error(f"DIAGNOSTIC: Symbol {symbol} not found")
            
            # Attempt to enable symbol
            if mt5.symbol_select(symbol, True):
                logging.info(f"DIAGNOSTIC: Symbol {symbol} has been enabled")
                symbol_info = mt5.symbol_info(symbol)
            else:
                logging.error(f"DIAGNOSTIC: Failed to enable symbol {symbol}")
                return
        
        # Log detailed symbol information
        logging.info(f"DIAGNOSTIC: Symbol {symbol} properties:")
        logging.info(f"  Trade mode: {symbol_info.trade_mode}")
        if hasattr(symbol_info, 'trade_execution_mode'):
            logging.info(f"  Execution mode: {symbol_info.trade_execution_mode}")
        if hasattr(symbol_info, 'filling_mode'):
            logging.info(f"  Filling mode: {symbol_info.filling_mode}")
        logging.info(f"  Digits: {symbol_info.digits}")
        logging.info(f"  Spread: {symbol_info.spread}")
        logging.info(f"  Min volume: {symbol_info.volume_min}")
        logging.info(f"  Max volume: {symbol_info.volume_max}")
        logging.info(f"  Volume step: {symbol_info.volume_step}")
        logging.info(f"  StopsLevel: {symbol_info.trade_stops_level}")
        
        # Check account trading status - Fix account_type attribute error
        account_info = mt5.account_info()
        if account_info:
            logging.info(f"DIAGNOSTIC: Account info:")
            logging.info(f"  Trade allowed: {account_info.trade_allowed}")
            # Removed account_type line that was causing errors
            logging.info(f"  Account number: {account_info.login}")
            logging.info(f"  Server: {account_info.server}")
            logging.info(f"  Balance: {account_info.balance}")
            logging.info(f"  Equity: {account_info.equity}")
            logging.info(f"  Margin: {account_info.margin}")
            logging.info(f"  Free margin: {account_info.margin_free}")
            if account_info.margin > 0:
                logging.info(f"  Margin level: {account_info.margin_level}%")
            
        # Check if MT5 expert advisors are enabled
        terminal_info = mt5.terminal_info()
        if terminal_info:
            logging.info(f"DIAGNOSTIC: EA trading allowed: {terminal_info.trade_allowed}")
            logging.info(f"DIAGNOSTIC: EA enabled: {terminal_info.trade_expert}")
            logging.info(f"DIAGNOSTIC: Automated trading allowed: {terminal_info.trade_automation}")
            
        # Check current price
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            logging.info(f"DIAGNOSTIC: Current prices - Bid: {tick.bid}, Ask: {tick.ask}")
            
        # Try a minimal test order if conditions allow
        if account_info and account_info.trade_allowed and terminal_info and terminal_info.trade_allowed:
            logging.info("DIAGNOSTIC: Attempting minimal test order without SL/TP...")
            
            # Use minimal volume
            min_volume = symbol_info.volume_min
            
            # Prepare a simple market order request
            test_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": min_volume,
                "type": mt5.ORDER_TYPE_BUY,  # Always use BUY for test
                "price": tick.ask,
                "deviation": 50,
                "magic": 999999,  # Special magic for test orders
                "comment": "DIAGNOSTIC TEST",
                "type_time": mt5.ORDER_TIME_GTC
            }
            
            # Remove type_filling to simplify the request
            if "type_filling" in test_request:
                del test_request["type_filling"]
                
            logging.info(f"DIAGNOSTIC: Test order details: {test_request}")
            
            # Send test order
            result = mt5.order_send(test_request)
            
            if result:
                logging.info(f"DIAGNOSTIC: Test order result code: {result.retcode}")
                logging.info(f"DIAGNOSTIC: Test order comment: {result.comment}")
                
                # Check if test order was successful
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"DIAGNOSTIC: Test order placed successfully with ticket: {result.order}")
                    
                    # Close the test position immediately
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": min_volume,
                        "type": mt5.ORDER_TYPE_SELL,
                        "position": result.order,
                        "price": tick.bid,
                        "deviation": 50,
                        "magic": 999999,
                        "comment": "DIAGNOSTIC TEST CLOSE",
                        "type_time": mt5.ORDER_TIME_GTC
                    }
                    
                    # Try to close the test position
                    close_result = mt5.order_send(close_request)
                    if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.info("DIAGNOSTIC: Test position closed successfully")
                    else:
                        logging.warning(f"DIAGNOSTIC: Failed to close test position: {close_result.comment if close_result else 'Unknown error'}")
                else:
                    logging.warning(f"DIAGNOSTIC: Test order failed with code: {result.retcode}, message: {result.comment}")
                    
                    # Decode common error codes
                    error_descriptions = {
                        10004: "Requote",
                        10006: "Order rejected",
                        10007: "Order canceled by client",
                        10008: "Order already executed",
                        10009: "Order already exists",
                        10010: "Order conditions not met",
                        10011: "Too many orders",
                        10012: "Trade disabled",
                        10013: "Market closed",
                        10014: "Not enough money",
                        10015: "Price changed",
                        10016: "Price off",
                        10017: "Invalid expiration",
                        10018: "Order locked",
                        10019: "Buy only allowed",
                        10020: "Sell only allowed",
                        10021: "Too many requests",
                        10022: "Trade timeout",
                        10023: "Invalid price",
                        10024: "Invalid stops",
                        10025: "Invalid trade volume",
                        10026: "Market closed",
                        10027: "Trade disabled"
                    }
                    
                    if result.retcode in error_descriptions:
                        logging.info(f"DIAGNOSTIC: Error description: {error_descriptions[result.retcode]}")
            else:
                logging.error("DIAGNOSTIC: Test order returned None")
                
        # Check current positions
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            logging.info(f"DIAGNOSTIC: Current open positions for {symbol}: {len(positions)}")
        else:
            logging.info(f"DIAGNOSTIC: No open positions for {symbol}")
            
        # Check pending orders
        orders = mt5.orders_get(symbol=symbol)
        if orders:
            logging.info(f"DIAGNOSTIC: Current pending orders for {symbol}: {len(orders)}")
        else:
            logging.info(f"DIAGNOSTIC: No pending orders for {symbol}")
            
        logging.info(f"======= END DIAGNOSTIC REPORT FOR {symbol} =======")
        
    except Exception as e:
        logging.error(f"DIAGNOSTIC: Error running diagnostics: {e}")
        logging.error(traceback.format_exc())

def optimize_crypto_parameters():
    """
    Weekly optimization for cryptocurrency parameters.
    Adjusts risk, volatility thresholds, and other settings specifically for crypto.
    """
    try:
        # Only run this optimization once per week
        current_time = time.time()
        if hasattr(state, 'last_crypto_optimization') and current_time - state.last_crypto_optimization < 7 * 24 * 3600:
            return
            
        logging.info("Performing weekly cryptocurrency parameter optimization")
        
        # For each crypto symbol
        for symbol in [s for s in SYMBOLS if "BTC" in s or "ETH" in s]:
            # Get historical data for deeper analysis
            df = get_candle_data(symbol, MT5_TIMEFRAMES["TREND"], num_bars=1000)
            if df is None:
                continue
                
            # Calculate historical volatility to adjust parameters
            close_prices = df['close']
            returns = np.log(close_prices / close_prices.shift(1)).dropna()
            hist_volatility = returns.std() * np.sqrt(365)
            
            # Store optimized parameters for this crypto
            if not hasattr(state, 'crypto_params'):
                state.crypto_params = {}
                
            state.crypto_params[symbol] = {
                "hist_volatility": hist_volatility,
                "suggested_risk": max(0.3, min(0.8, 0.5 / hist_volatility)),  # Inverse relation to volatility
                "sl_multiplier": max(1.2, min(3.0, 1.5 + hist_volatility)),   # Increase with volatility
                "tp_multiplier": max(1.8, min(4.0, 2.0 + hist_volatility)),   # Increase with volatility
                "last_updated": datetime.now().isoformat()
            }
            
        state.last_crypto_optimization = current_time
        logging.info(f"Crypto parameter optimization complete: {state.crypto_params}")
        
    except Exception as e:
        logging.error(f"Error in crypto parameter optimization: {e}")

def log_missed_opportunity(symbol, filter_reason, signal_data=None):
    """
    Log potentially missed trading opportunities that were filtered out.
    This helps monitor if good signals are being blocked by overly restrictive filters.
    """
    try:
        # If no signal data provided, try to calculate one
        if signal_data is None:
            # Quick calculation of potential signal without full validation
            signal_data = calculate_consolidated_signal(symbol)
        
        # Only log if there was an actual signal
        if signal_data and signal_data.get("signal", "NONE") != "NONE":
            # Get current market details
            price = mt5.symbol_info_tick(symbol).ask
            
            # Use a distinctive log format for easy filtering in log analysis
            logging.warning(f"MISSED OPPORTUNITY: {symbol} {signal_data['signal']} at {price} - "
                           f"Strength: {signal_data.get('strength', 0):.2f} - "
                           f"Rejected by: {filter_reason}")
            
            # Store in a global tracker for later analysis
            if not hasattr(state, 'missed_opportunities'):
                state.missed_opportunities = []
            
            state.missed_opportunities.append({
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "signal": signal_data.get("signal"),
                "strength": signal_data.get("strength", 0),
                "price": price,
                "filter_reason": filter_reason
            })
            
            # Prevent list from growing too large
            if len(state.missed_opportunities) > 100:
                state.missed_opportunities = state.missed_opportunities[-100:]
    except Exception as e:
        logging.error(f"Error logging missed opportunity: {e}")

def calculate_lot_size_enhanced(symbol, risk_percentage, stop_loss_pips):
    """
    Enhanced lot size calculation with improved risk management features.
    Uses equity instead of balance, adapts to market conditions, and applies additional safety checks.
    
    Args:
        symbol (str): Trading symbol
        risk_percentage (float): Base risk percentage
        stop_loss_pips (float): Stop loss distance in pips
        
    Returns:
        float: Calculated lot size
    """
    try:
        # Check for global safety mode
        if hasattr(state, 'safe_mode_until') and time.time() < state.safe_mode_until:
            logging.warning(f"System in safe mode - using minimum lot size for {symbol}")
            return 0.01  # Minimum lot size during safe mode
            
        # Get account info
        account_info = get_account_info()
        if account_info is None:
            return 0.01  # Default minimum

        # Only check margin level if margin is actually being used
        if account_info.margin > 0:
            if account_info.margin_level is not None and account_info.margin_level < 200:
                logging.warning(f"Critical margin level ({account_info.margin_level:.1f}%) - using minimum lot size")
                return 0.01  # Minimum lot size when margin is critical
                
        # # Check margin level - if critically low, return minimum lot size
        # if account_info.margin_level is not None and account_info.margin_level < 200:
        #     logging.warning(f"Critical margin level ({account_info.margin_level:.1f}%) - using minimum lot size")
        #     return 0.01  # Minimum lot size when margin is critical
            
        # Check if symbol is in cooldown period after margin errors
        current_time = time.time()
        if hasattr(state, 'margin_error_cooldown') and symbol in state.margin_error_cooldown:
            if current_time < state.margin_error_cooldown[symbol]:
                cooldown_minutes = (state.margin_error_cooldown[symbol] - current_time) / 60
                logging.info(f"{symbol} in margin error cooldown ({cooldown_minutes:.1f} minutes remaining) - using reduced lot size")
                return 0.01  # Minimum lot size during cooldown
        
        # Use equity instead of balance for risk calculation
        # This accounts for unrealized P/L in open positions
        equity = account_info.equity
        
        # Calculate total risk exposure from existing positions
        existing_exposure = 0
        current_positions = get_positions()
        for position in current_positions:
            pos_symbol_info = get_symbol_info(position.symbol)
            if pos_symbol_info:
                # Calculate risk exposure of existing position
                exposure_multiplier = 1.0
                # Apply correlation discount for related symbols
                if position.symbol != symbol and position.symbol[:3] in symbol or symbol[:3] in position.symbol:
                    exposure_multiplier = 0.5  # Reduce counted exposure for correlated pairs
                existing_exposure += position.volume * exposure_multiplier
        
        # Get symbol info
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return 0.01  # Default minimum
        
        # Get symbol-specific values
        point = symbol_info.point
        contract_size = symbol_info.trade_contract_size
        price = mt5.symbol_info_tick(symbol).ask
        
        # Currency conversion if needed (for cross pairs)
        currency_profit = symbol_info.currency_profit
        account_currency = account_info.currency
        
        conversion_rate = 1.0
        if currency_profit != account_currency:
            conversion_pair = f"{currency_profit}{account_currency}"
            conversion_info = get_symbol_info(conversion_pair)
            if conversion_info:
                conversion_rate = mt5.symbol_info_tick(conversion_pair).ask
        
        # Apply dynamic risk adjustment based on market conditions
        adjusted_risk_percentage = calculate_dynamic_risk_percentage(symbol)
        
        # ADAPTIVE SCALING BASED ON ACCOUNT SIZE
        # Use a smooth logarithmic scaling function for account size factor
        # This ensures small accounts get protection but larger accounts aren't restricted
        base_equity = 1000.0  # Reference point
        equity_ratio = equity / base_equity
        
        # Scale account safety factor based on account size - logarithmic scaling
        # This gives more protection to smaller accounts while allowing larger accounts to use normal risk
        account_scaling = min(1.1, max(0.4, 0.4 + (0.3 * math.log10(equity_ratio + 0.1))))
        
        # Similarly scale margin safety buffer more aggressively for small accounts
        margin_safety_factor = min(0.8, max(0.3, 0.3 + (0.5 * math.log10(equity_ratio + 0.1))))
        
        # For crypto specifically, be more conservative with very small accounts
        if symbol.startswith("BTC") or symbol.startswith("ETH"):
            if equity < 2000:
                # Extra safety for crypto with small accounts
                crypto_factor = min(1.0, equity / 2000)
                account_scaling *= crypto_factor
        
        # Check current margin usage first
        margin_used_percent = account_info.margin / account_info.equity if account_info.equity > 0 else 0
        
        # Reduce safety factor if margin usage is already high
        if margin_used_percent > 0.2:  # More than 20% margin already used
            margin_safety_factor *= (1 - margin_used_percent)
        
        # Apply position count scaling more aggressively
        position_count = len(current_positions)
        if position_count > 1:
            # Reduce lot size more significantly with more positions
            exposure_scaling = max(0.3, 1.0 - (position_count * 0.15))
        else:
            exposure_scaling = max(0.5, 1.0 - (existing_exposure * 0.1))
        
        # Apply scaling factors to base risk percentage
        final_risk_percentage = (
            adjusted_risk_percentage * 
            account_scaling * 
            exposure_scaling
        )
        
        # Apply global risk modifier
        if hasattr(state, 'risk_modifier'):
            final_risk_percentage *= state.risk_modifier
        
        # Cap maximum risk percentage
        final_risk_percentage = min(final_risk_percentage, MAX_RISK_PER_TRADE * 100)
        
        # Calculate risk amount
        risk_amount = equity * (final_risk_percentage / 100)
        
        # Calculate pip value
        pip_value = (point * contract_size) / price
        
        # Convert pip value to account currency
        pip_value_in_account_currency = pip_value * conversion_rate
        
        # Calculate lot size
        stop_loss_amount = stop_loss_pips * pip_value_in_account_currency
        lot_size = risk_amount / stop_loss_amount if stop_loss_amount > 0 else 0.01
        
        # Apply progressive lot sizing based on recent performance
        win_rate = state.trade_stats.get("win_rate", 50)
        profit_factor = state.trade_stats.get("profit_factor", 1.0)
        
        # Scale lot size based on trading performance
        if win_rate > 60 and profit_factor > 1.5:
            # Trading well - can increase position size slightly
            performance_multiplier = 1.1
        elif win_rate < 40 or profit_factor < 0.8:
            # Trading poorly - reduce position size to preserve capital
            performance_multiplier = 0.8
        else:
            # Normal performance - no adjustment
            performance_multiplier = 1.0
        
        lot_size *= performance_multiplier
        
        # Scale back for initial trades to build confidence
        total_trades = state.trade_stats.get("total_trades", 0)
        if total_trades < 10:
            # Start with smaller positions for first few trades
            warmup_factor = 0.5 + (total_trades * 0.05)  # Scales from 0.5 to 1.0 over first 10 trades
            lot_size *= warmup_factor
        
        # Check volatility and adjust lot size
        volatility_level, _ = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
        if volatility_level == "extreme":
            lot_size *= 0.6  # 60% lot size in extreme volatility
        elif volatility_level == "high":
            lot_size *= 0.8  # 80% lot size in high volatility
        
        # Check for high-impact news and reduce size if found
        if check_news_events(symbol, hours=12):  # Check for news in next 12 hours
            lot_size *= 0.75  # Reduce lot size around news events
        
        # Ensure lot size is within symbol's limits
        lot_size = max(lot_size, symbol_info.volume_min)
        lot_size = min(lot_size, symbol_info.volume_max)
        
        # Round to nearest step
        lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
        
        # Check if margin is sufficient with safety buffer
        margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, lot_size, price)
        if margin and margin > account_info.margin_free * margin_safety_factor:
            # Reduce lot size to fit available margin with safety buffer
            max_affordable_lot = (account_info.margin_free * margin_safety_factor / margin) * lot_size
            max_affordable_lot = max(max_affordable_lot, symbol_info.volume_min)
            max_affordable_lot = min(max_affordable_lot, symbol_info.volume_max)
            max_affordable_lot = round(max_affordable_lot / symbol_info.volume_step) * symbol_info.volume_step
            lot_size = max_affordable_lot
        
        # Apply maximum exposure limit (max 5% of account equity per position)
        position_value = lot_size * contract_size * price
        max_position_value = equity * 0.05  # Max 5% of equity in one position
        
        if position_value > max_position_value:
            reduced_lot_size = max_position_value / (contract_size * price)
            reduced_lot_size = max(reduced_lot_size, symbol_info.volume_min)
            reduced_lot_size = min(reduced_lot_size, symbol_info.volume_max)
            reduced_lot_size = round(reduced_lot_size / symbol_info.volume_step) * symbol_info.volume_step
            lot_size = reduced_lot_size
        
        # Safety check for margin - reduce further if we've seen margin exceeded errors
        # Check if we have had margin errors for this symbol before
        if hasattr(state, 'margin_errors') and symbol in getattr(state, 'margin_errors', {}):
            logging.warning(f"Applying extra margin safety factor for {symbol} due to previous margin errors")
            lot_size *= 0.5  # Reduce lot size by 50% when margin errors occurred previously
        
        # Additional global safety check for overall margin errors
        if hasattr(state, 'margin_error_global') and state.margin_error_global:
            lot_size *= 0.7  # Reduce all lot sizes by 30% if we've seen global margin errors
        
        logging.info(f"Enhanced lot size calculation for {symbol}: {lot_size} lots " +
                    f"(Risk: {final_risk_percentage:.2f}%, " +
                    f"Stop loss: {stop_loss_pips} pips)")
        
        return lot_size
    
    except Exception as e:
        logging.error(f"Error calculating enhanced lot size: {e}")
        # Track margin exceeded errors to apply additional safety in future calculations
        if 'margin_exceeded' in str(e):
            if not hasattr(state, 'margin_errors'):
                state.margin_errors = {}
            state.margin_errors[symbol] = True
            
            # Add a cooldown period for this symbol after margin errors
            if not hasattr(state, 'margin_error_cooldown'):
                state.margin_error_cooldown = {}
            
            state.margin_error_cooldown[symbol] = time.time() + 3600  # 1 hour cooldown
            
            logging.warning(f"Margin exceeded error detected for {symbol}. Will apply additional safety in future runs.")
        
        return 0.01  # Default minimum
    
    except Exception as e:
        logging.error(f"Error calculating enhanced lot size: {e}")
        # Track margin exceeded errors to apply additional safety in future calculations
        if 'margin_exceeded' in str(e):
            if not hasattr(state, 'margin_errors'):
                state.margin_errors = {}
            state.margin_errors[symbol] = True
            
            # Add a cooldown period for this symbol after margin errors
            if not hasattr(state, 'margin_error_cooldown'):
                state.margin_error_cooldown = {}
            
            state.margin_error_cooldown[symbol] = time.time() + 3600  # 1 hour cooldown
            
            logging.warning(f"Margin exceeded error detected for {symbol}. Will apply additional safety in future runs.")
        
        return 0.01  # Default minimum

def enhance_risk_management():
    """
    Apply enhanced risk management to overall trading system.
    This function should be called periodically to update risk parameters.
    """
    try:
        with state.locks['risk_modifier']:
            # Get account info
            account_info = get_account_info()
            if account_info is None:
                return
            
            equity = account_info.equity
            balance = account_info.balance

            # Get positions to see if we have any open trades
            positions = get_positions()
            
            # CHECK IF NO POSITIONS - if no positions, margin level doesn't matter
            if len(positions) == 0 or account_info.margin == 0:
                # No positions open or zero margin, so no risk to manage - proceed with normal trading
                state.risk_modifier = 1.0
                logging.info("No positions open - setting normal risk parameters")
                return 1.0
            
            # Check margin level first - if critically low, block all trading
            if account_info.margin_level is not None and account_info.margin_level < 300:  # 300% is a safe margin level
                state.risk_modifier = 0.0  # Block all trading
                logging.warning(f"Critical margin level detected ({account_info.margin_level:.1f}%). Blocking all trading.")
                return 0.0
                
            # Check if margin usage is already very high
            margin_usage_percent = (account_info.margin / account_info.equity) * 100 if account_info.equity > 0 else 0
            if margin_usage_percent > 50:  # Using more than 50% of equity as margin
                state.risk_modifier = 0.3  # Severely reduce risk
                logging.warning(f"High margin usage detected ({margin_usage_percent:.1f}%). Severely reducing risk.")
                return 0.3
            
            # Calculate drawdown
            initial_balance = state.initial_balance or balance
            current_drawdown = (initial_balance - equity) / initial_balance if initial_balance > 0 else 0
            
            # Get exposure information
            exposure_info = calculate_total_exposure(account_info)
            
            # Scale global risk based on drawdown
            if current_drawdown > 0.15:  # > 15% drawdown
                # Significantly reduce risk
                risk_modifier = 0.5
                logging.warning(f"Significant drawdown detected ({current_drawdown:.2f}%). Reducing risk to 50%.")
            elif current_drawdown > 0.1:  # > 10% drawdown
                # Moderately reduce risk
                risk_modifier = 0.7
                logging.info(f"Moderate drawdown detected ({current_drawdown:.2f}%). Reducing risk to 70%.")
            elif current_drawdown > 0.05:  # > 5% drawdown
                # Slightly reduce risk
                risk_modifier = 0.85
                logging.info(f"Minor drawdown detected ({current_drawdown:.2f}%). Reducing risk to 85%.")
            else:
                # Normal risk
                risk_modifier = 1.0
            
            # NEW: Special handling for Bitcoin volatility
            for position in positions:
                if position.symbol.startswith("BTC") or "BTC" in position.symbol:
                    volatility_level, _ = calculate_market_volatility(position.symbol, MT5_TIMEFRAMES["PRIMARY"])
                    if volatility_level in ["extreme", "super-extreme"]:
                        # Adjust risk for Bitcoin in extreme conditions
                        logging.info(f"Bitcoin in {volatility_level} volatility - adjusting risk tolerance")
                        if risk_modifier > 0.8:  # Only increase if we're not already in drawdown
                            risk_modifier = 0.8  # Cap risk to 80% during extreme BTC volatility

            # Further adjust based on performance
            win_rate = state.trade_stats.get("win_rate", 50)
            profit_factor = state.trade_stats.get("profit_factor", 1.0)
            
            # Scale risk based on trading performance
            if win_rate > 65 and profit_factor > 2.0:
                # Excellent performance
                performance_modifier = 1.2
            elif win_rate > 55 and profit_factor > 1.5:
                # Good performance
                performance_modifier = 1.1
            elif win_rate < 40 or profit_factor < 0.8:
                # Poor performance
                performance_modifier = 0.8
                logging.info(f"Suboptimal trading performance (WR: {win_rate:.1f}%, PF: {profit_factor:.2f}). Reducing risk.")
            else:
                # Normal performance
                performance_modifier = 1.0
            
            # Adjust for excessive exposure
            if exposure_info["margin_exceeded"]:
                exposure_modifier = 0.6
                logging.warning(f"High margin usage detected ({exposure_info['exposure_percent']:.1f}%). Reducing risk.")
            elif exposure_info["concentration_exceeded"]:
                exposure_modifier = 0.7
                logging.warning(f"Currency concentration detected in {exposure_info['max_currency']}. Reducing risk.")
            elif exposure_info["position_count"] > MAX_OPEN_POSITIONS * 0.7:
                exposure_modifier = 0.8
                logging.info(f"Multiple positions open ({exposure_info['position_count']}). Moderating risk.")
            else:
                exposure_modifier = 1.0
            
            # Calculate combined risk modifier
            combined_modifier = risk_modifier * performance_modifier * exposure_modifier
            
            # Store the modifier in state for use by other functions
            state.risk_modifier = combined_modifier
            
            logging.info(f"Risk management updated - Modifier: {combined_modifier:.2f} (Drawdown: {current_drawdown:.2f}%, " +
                    f"Performance: {performance_modifier:.1f}, Exposure: {exposure_modifier:.1f})")
            
            # Update the risk modifier safely
            state.risk_modifier = combined_modifier

            return combined_modifier
        
    except Exception as e:
        logging.error(f"Error enhancing risk management: {e}")
        # Track margin exceeded errors to apply additional safety in future calculations
        if 'margin_exceeded' in str(e):
            if not hasattr(state, 'margin_error_global'):
                state.margin_error_global = True
            logging.warning("Margin exceeded error detected globally. Will apply additional safety in future runs.")
        return 1.0  # Default - no change to risk

def calculate_total_exposure(account_info=None):
    """
    Calculate the total risk exposure across all open positions.
    This helps prevent overexposure when multiple positions are open.
    
    Returns:
        dict: Information about current exposure
    """
    try:
        if account_info is None:
            account_info = get_account_info()
            if account_info is None:
                return {"total_exposure": 0, "exposure_percent": 0, "max_exceeded": False}
        
        equity = account_info.equity
        positions = get_positions()
        
        if not positions:
            return {"total_exposure": 0, "exposure_percent": 0, "max_exceeded": False}
        
        # Calculate exposure metrics
        margin_used = account_info.margin
        free_margin = account_info.margin_free
        total_volume = sum(position.volume for position in positions)
        position_count = len(positions)
        
        # Group positions by currency to check for concentration
        currency_exposure = {}
        pair_exposure = {}
        
        for position in positions:
            symbol = position.symbol
            pair_exposure[symbol] = pair_exposure.get(symbol, 0) + position.volume
            
            # Extract currencies for forex pairs
            if len(symbol) >= 6 and symbol[:6].isalpha():
                base_currency = symbol[:3]
                quote_currency = symbol[3:6]
                
                currency_exposure[base_currency] = currency_exposure.get(base_currency, 0) + position.volume
                currency_exposure[quote_currency] = currency_exposure.get(quote_currency, 0) + position.volume
        
        # Calculate exposure percentages
        margin_percent = (margin_used / equity) * 100 if equity > 0 else 0
        
        # Check if any currency has excessive concentration
        max_currency_exposure = max(currency_exposure.values()) if currency_exposure else 0
        max_currency = max(currency_exposure.items(), key=lambda x: x[1])[0] if currency_exposure else "None"
        
        # Flag excessive exposure situations
        margin_exceeded = margin_percent > 30  # Warning if using more than 30% of equity as margin
        concentration_exceeded = max_currency_exposure > total_volume * 0.7  # Warning if >70% concentrated in one currency
        
        return {
            "total_exposure": total_volume,
            "exposure_percent": margin_percent,
            "position_count": position_count,
            "margin_used": margin_used,
            "free_margin": free_margin,
            "currency_exposure": currency_exposure,
            "pair_exposure": pair_exposure,
            "max_currency": max_currency,
            "max_currency_exposure": max_currency_exposure,
            "margin_exceeded": margin_exceeded,
            "concentration_exceeded": concentration_exceeded,
            "max_exceeded": margin_exceeded or concentration_exceeded
        }
    
    except Exception as e:
        logging.error(f"Error calculating total exposure: {e}")
        return {"total_exposure": 0, "exposure_percent": 0, "max_exceeded": False}

def calculate_minimum_viable_profit(symbol, lot_size=0.01):
    """
    Calculate the minimum profit needed to cover spread, commissions, and slippage.
    This function determines the minimum profit required for a trade to be worthwhile.
    """
    try:
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return 0.0
        
        # Calculate spread cost
        spread_in_points = symbol_info.spread
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value  # Value of a tick in account currency
        
        # Avoid division by zero
        if tick_size == 0:
            tick_size = symbol_info.point
        
        point_value = tick_value / tick_size
        spread_cost = spread_in_points * point_value * lot_size
        
        # Estimate commission (varies by broker)
        # This is typically per standard lot, so scale by lot size
        base_commission = 7.0  # Default estimate in account currency per standard lot
        estimated_commission = base_commission * (lot_size / 1.0)  # Scale by lot size
        
        # Add slippage estimate (usually 1-3 points)
        slippage_points = 2
        slippage_cost = slippage_points * point_value * lot_size
        
        # Total transaction cost (round trip - entry and exit)
        total_cost = (spread_cost + estimated_commission + slippage_cost) * 2
        
        # Add a safety margin (20%)
        return total_cost * 1.2
    except Exception as e:
        logging.error(f"Error calculating minimum viable profit: {e}")
        return 10.0  # Return a default value if calculation fails

def is_high_liquidity_session(symbol):
    """
    Check if current time is in a high-liquidity trading session for the symbol.
    This helps avoid trading during low liquidity periods when spreads might be wider.
    """
    try:
        now = datetime.now(timezone.utc)
        
        # Special handling for cryptocurrencies - they trade 24/7
        if symbol.startswith("BTC") or symbol.startswith("ETH") or symbol.startswith("LTC") or symbol.startswith("XRP"):
            return True
        
        # Check if it's weekend - special handling
        is_weekend = now.weekday() >= 5  # Saturday=5, Sunday=6
        
        # For crypto-fiat pairs (even if not starting with crypto symbol)
        if "BTC" in symbol or "ETH" in symbol or "USDT" in symbol:
            return True
            
        # Extract currency pairs
        if len(symbol) >= 6 and symbol[:6].isalpha():
            base_currency = symbol[:3]
            quote_currency = symbol[3:6]
        else:
            # For non-forex symbols, use a default approach
            hour = now.hour
            # During weekends, only allow crypto
            if is_weekend:
                return False
            # Standard market hours (8 AM to 4 PM local time zone)
            return 8 <= hour <= 16
        
        # During weekends, forex markets are closed
        if is_weekend:
            return False
            
        # Define major sessions (times in UTC)
        asian_session = 0 <= now.hour < 8
        european_session = 7 <= now.hour < 16
        us_session = 12 <= now.hour < 21
        
        # Match currencies to their primary sessions
        asian_currencies = ['JPY', 'AUD', 'NZD', 'SGD', 'HKD', 'CNH']
        european_currencies = ['EUR', 'GBP', 'CHF', 'SEK', 'NOK', 'DKK']
        us_currencies = ['USD', 'CAD', 'MXN']
        
        # Check if currency is traded primarily in a session that's currently active
        if base_currency in asian_currencies or quote_currency in asian_currencies:
            if asian_session:
                return True
        
        if base_currency in european_currencies or quote_currency in european_currencies:
            if european_session:
                return True
        
        if base_currency in us_currencies or quote_currency in us_currencies:
            if us_session:
                return True
        
        # Overlapping sessions have higher liquidity
        if (european_session and us_session) or (asian_session and european_session):
            return True
        
        return False
    except Exception as e:
        logging.error(f"Error checking liquidity session: {e}")
        return True  # Default to assuming high liquidity if check fails

def calculate_dynamic_risk_percentage(symbol, market_regime=None):
    """
    Adjust risk percentage based on market conditions.
    Reduces risk in volatile markets and increases it in favorable conditions.
    """
    try:
        # Get market structure
        market_state = analyze_market_structure(symbol)
        volatility = market_state.get("volatility", "normal")
        trend_strength = market_state.get("trend_strength", 0)
        structure = market_state.get("structure", "unknown")
        
        # Get base risk from config
        base_risk = RISK_PERCENTAGE
        
        # Adjust for volatility
        if volatility == "extreme":
            risk_multiplier = 0.5  # Half normal risk in extreme volatility
        elif volatility == "high":
            risk_multiplier = 0.7  # Reduce risk in high volatility
        elif volatility == "low" and trend_strength > 0.7:
            # Increase risk in strong trends with low volatility
            risk_multiplier = 1.2
        else:
            risk_multiplier = 1.0
        
        # Get market regime if not provided
        if market_regime is None:
            market_regime, _ = detect_market_regime(symbol)
            
        # Adjust for market structure or regime
        if market_regime:
            # Use provided market regime
            if market_regime == "STRONG_UPTREND" or market_regime == "STRONG_DOWNTREND":
                structure_multiplier = 1.3  # Increase risk in strong trends
            elif market_regime == "RANGE_BOUND":
                structure_multiplier = 0.9  # Reduce risk in ranging markets
            elif market_regime == "CHOPPY_VOLATILE":
                structure_multiplier = 0.7  # Reduce risk more in choppy markets
            else:
                structure_multiplier = 1.0
        else:
            # Use detected structure
            if structure == "trending" and trend_strength > 0.7:
                structure_multiplier = 1.1  # Increase risk in strong trends
            elif structure == "ranging":
                structure_multiplier = 0.9  # Reduce risk in ranging markets
            else:
                structure_multiplier = 1.0
        
        # Adjust for liquidity
        if is_high_liquidity_session(symbol):
            liquidity_multiplier = 1.0
        else:
            liquidity_multiplier = 0.8  # Reduce risk in low liquidity sessions
        
        # Calculate final risk percentage
        adjusted_risk = base_risk * risk_multiplier * structure_multiplier * liquidity_multiplier
        
        # Ensure risk percentage stays within reasonable bounds
        min_risk = base_risk * 0.3  # Don't go below 30% of base risk
        max_risk = base_risk * 1.5  # Don't go above 150% of base risk
        
        return max(min_risk, min(adjusted_risk, max_risk))
    except Exception as e:
        logging.error(f"Error calculating dynamic risk: {e}")
        return RISK_PERCENTAGE  # Return default risk percentage if calculation fails
    
def analyze_market_structure_enhanced(symbol, timeframe=None):
    """
    Enhanced market structure analysis that identifies swing highs/lows
    and key support/resistance levels using volume profile.
    """
    try:
        if timeframe is None:
            timeframe = MT5_TIMEFRAMES["TREND"]
        
        # Get candle data with enough bars for analysis
        df = get_candle_data(symbol, timeframe, num_bars=1000)
        if df is None:
            return None
        
        # Calculate original market structure
        base_market_structure = analyze_market_structure(symbol)
        
        # Identify swing highs and lows
        window_size = 5  # Look 5 bars to each side
        
        # Initialize with empty arrays
        swing_highs = []
        swing_lows = []
        
        # Identify swing points
        for i in range(window_size, len(df) - window_size):
            # Check for swing high
            if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window_size+1)) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window_size+1)):
                swing_highs.append((df.index[i], df['high'].iloc[i], df['time'].iloc[i]))
            
            # Check for swing low
            if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window_size+1)) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window_size+1)):
                swing_lows.append((df.index[i], df['low'].iloc[i], df['time'].iloc[i]))
        
        # Calculate higher highs, lower lows, etc.
        recent_swing_highs = swing_highs[-5:] if len(swing_highs) >= 5 else swing_highs
        recent_swing_lows = swing_lows[-5:] if len(swing_lows) >= 5 else swing_lows
        
        higher_highs = all(recent_swing_highs[i][1] > recent_swing_highs[i-1][1] for i in range(1, len(recent_swing_highs))) if len(recent_swing_highs) > 1 else False
        higher_lows = all(recent_swing_lows[i][1] > recent_swing_lows[i-1][1] for i in range(1, len(recent_swing_lows))) if len(recent_swing_lows) > 1 else False
        lower_highs = all(recent_swing_highs[i][1] < recent_swing_highs[i-1][1] for i in range(1, len(recent_swing_highs))) if len(recent_swing_highs) > 1 else False
        lower_lows = all(recent_swing_lows[i][1] < recent_swing_lows[i-1][1] for i in range(1, len(recent_swing_lows))) if len(recent_swing_lows) > 1 else False
        
        # Create volume profile
        volume_profile = {}
        
        # Define price ranges for volume profile
        min_price = df['low'].min()
        max_price = df['high'].max()
        price_range = max_price - min_price
        
        # Create 20 price zones
        zone_size = price_range / 20
        
        # Initialize volume zones
        for i in range(20):
            zone_min = min_price + i * zone_size
            zone_max = min_price + (i + 1) * zone_size
            volume_profile[f"zone_{i}"] = {
                "min_price": zone_min,
                "max_price": zone_max,
                "volume": 0
            }
        
        # Aggregate volume by price zone
        for i in range(len(df)):
            candle_high = df['high'].iloc[i]
            candle_low = df['low'].iloc[i]
            candle_volume = df['volume'].iloc[i]
            
            # Determine which zones this candle touches
            for j in range(20):
                zone_min = min_price + j * zone_size
                zone_max = min_price + (j + 1) * zone_size
                
                # Check if candle overlaps with this zone
                if candle_low <= zone_max and candle_high >= zone_min:
                    # Calculate how much of the candle is in this zone
                    overlap = min(candle_high, zone_max) - max(candle_low, zone_min)
                    candle_range = candle_high - candle_low
                    
                    # Proportion of volume to attribute to this zone
                    volume_proportion = overlap / candle_range if candle_range > 0 else 0
                    volume_profile[f"zone_{j}"]["volume"] += candle_volume * volume_proportion
        
        # Find high volume zones (potential support/resistance)
        sorted_zones = sorted(volume_profile.items(), key=lambda x: x[1]["volume"], reverse=True)
        high_volume_zones = sorted_zones[:5]  # Top 5 high volume zones
        
        # Extract key levels from high volume zones
        key_volume_levels = []
        for zone_name, zone_data in high_volume_zones:
            # Use the middle of the high volume zone as a key level
            level_price = (zone_data["min_price"] + zone_data["max_price"]) / 2
            key_volume_levels.append({
                "price": level_price,
                "type": "volume_level",
                "strength": zone_data["volume"] / max(vz["volume"] for _, vz in high_volume_zones)  # Normalized strength
            })
        
        # Enhance the original market structure with new data
        enhanced_structure = base_market_structure.copy() if base_market_structure else {}
        
        # Add swing points and trend patterns
        enhanced_structure.update({
            "swing_highs": [(str(time), price) for _, price, time in swing_highs[-10:]] if swing_highs else [],
            "swing_lows": [(str(time), price) for _, price, time in swing_lows[-10:]] if swing_lows else [],
            "higher_highs": higher_highs,
            "higher_lows": higher_lows,
            "lower_highs": lower_highs,
            "lower_lows": lower_lows,
            "volume_profile": {k: {"min": v["min_price"], "max": v["max_price"], "volume": v["volume"]} 
                               for k, v in sorted_zones[:10]},  # Include top 10 volume zones
        })
        
        # Determine trend direction based on swing patterns
        if higher_highs and higher_lows:
            enhanced_structure["swing_trend"] = "bullish"
        elif lower_highs and lower_lows:
            enhanced_structure["swing_trend"] = "bearish"
        else:
            enhanced_structure["swing_trend"] = "neutral"
        
        # Add volume-based key levels
        if "key_levels" in enhanced_structure:
            enhanced_structure["key_levels"].extend(key_volume_levels)
            # Sort key levels by price
            enhanced_structure["key_levels"].sort(key=lambda x: x["price"])
        else:
            enhanced_structure["key_levels"] = key_volume_levels
        
        return enhanced_structure
    except Exception as e:
        logging.error(f"Error in enhanced market structure analysis: {e}")
        # Fall back to basic market structure analysis
        return analyze_market_structure(symbol)

def calculate_multi_timeframe_momentum(symbol):
    """
    Calculate momentum indicators across multiple timeframes.
    This provides a more holistic view of market momentum.
    """
    try:
        momentum_data = {}
        
        # Define timeframes to analyze
        timeframes = [
            mt5.TIMEFRAME_M5,
            mt5.TIMEFRAME_M15,
            mt5.TIMEFRAME_M30,
            mt5.TIMEFRAME_H1,
            mt5.TIMEFRAME_H4
        ]
        
        for tf in timeframes:
            df = get_candle_data(symbol, tf)
            if df is None:
                continue
            
            # Calculate RSI
            rsi = calculate_rsi(df)
            
            # Calculate MACD
            macd_line, signal_line, histogram = calculate_macd(df)
            
            # Calculate Rate of Change (ROC)
            close_prices = df['close']
            roc_10 = ((close_prices / close_prices.shift(10)) - 1) * 100
            
            # Determine if momentum is bullish, bearish, or neutral
            rsi_signal = "bullish" if rsi > 50 else "bearish"
            macd_signal = "bullish" if histogram > 0 else "bearish"
            roc_signal = "bullish" if roc_10.iloc[-1] > 0 else "bearish"
            
            # Combined momentum signal
            if rsi_signal == macd_signal == roc_signal:
                momentum = rsi_signal
            elif (rsi_signal == macd_signal) or (rsi_signal == roc_signal) or (macd_signal == roc_signal):
                # At least 2 agree
                if rsi_signal == macd_signal:
                    momentum = rsi_signal
                elif rsi_signal == roc_signal:
                    momentum = rsi_signal
                else:
                    momentum = macd_signal
            else:
                momentum = "neutral"
            
            # Calculate momentum strength (0-1)
            rsi_strength = abs(rsi - 50) / 50  # 0 at RSI 50, 1 at RSI 0 or 100
            macd_norm = abs(histogram) / (abs(macd_line) + 1e-10)  # Normalized MACD histogram
            roc_strength = min(abs(roc_10.iloc[-1]) / 5, 1.0)  # Cap at 5% change for max strength
            
            # Overall momentum strength
            momentum_strength = (rsi_strength + macd_norm + roc_strength) / 3
            
            # Store results for this timeframe
            tf_name = {
                mt5.TIMEFRAME_M5: "M5",
                mt5.TIMEFRAME_M15: "M15",
                mt5.TIMEFRAME_M30: "M30",
                mt5.TIMEFRAME_H1: "H1",
                mt5.TIMEFRAME_H4: "H4"
            }.get(tf, str(tf))
            
            momentum_data[tf_name] = {
                "momentum": momentum,
                "strength": momentum_strength,
                "rsi": rsi,
                "macd_histogram": histogram,
                "roc": roc_10.iloc[-1]
            }
        
        # Calculate weighted multi-timeframe momentum
        weights = {
            "M5": 0.1,
            "M15": 0.15,
            "M30": 0.2,
            "H1": 0.25,
            "H4": 0.3
        }
        
        weighted_score = 0
        total_weight = 0
        
        for tf, data in momentum_data.items():
            if tf in weights:
                weight = weights[tf]
                direction_score = 1 if data["momentum"] == "bullish" else -1 if data["momentum"] == "bearish" else 0
                weighted_score += direction_score * data["strength"] * weight
                total_weight += weight
        
        # Overall momentum consensus
        if total_weight > 0:
            consensus_score = weighted_score / total_weight
            
            if consensus_score > 0.3:
                consensus = "bullish"
            elif consensus_score < -0.3:
                consensus = "bearish"
            else:
                consensus = "neutral"
            
            momentum_data["consensus"] = {
                "momentum": consensus,
                "strength": abs(consensus_score),
                "score": consensus_score
            }
        
        return momentum_data
    except Exception as e:
        logging.error(f"Error calculating multi-timeframe momentum: {e}")
        return {"consensus": {"momentum": "neutral", "strength": 0, "score": 0}}

def analyze_currency_correlations(symbol):
    """
    Analyze correlations between the current symbol and other currency pairs.
    This helps identify potential conflicting signals across correlated pairs.
    """
    try:
        # Only do correlation analysis for currency pairs
        if len(symbol) < 6 or not symbol[:6].isalpha():
            return {}
        
        # Extract base and quote currencies
        base_currency = symbol[:3]
        quote_currency = symbol[3:6]
        
        # Find related pairs containing either the base or quote currency
        related_pairs = []
        
        for pair in SYMBOLS:
            if len(pair) >= 6 and pair[:6].isalpha():
                pair_base = pair[:3]
                pair_quote = pair[3:6]
                
                if pair != symbol and (pair_base == base_currency or pair_base == quote_currency or 
                                       pair_quote == base_currency or pair_quote == quote_currency):
                    related_pairs.append(pair)
        
        if not related_pairs:
            return {}
        
        # Get price data for main symbol
        main_df = get_candle_data(symbol, MT5_TIMEFRAMES["PRIMARY"], num_bars=100)
        if main_df is None:
            return {}
        
        main_returns = main_df['close'].pct_change().dropna()
        
        correlations = {}
        
        # Calculate correlations with related pairs
        for pair in related_pairs:
            pair_df = get_candle_data(pair, MT5_TIMEFRAMES["PRIMARY"], num_bars=100)
            if pair_df is None:
                continue
            
            pair_returns = pair_df['close'].pct_change().dropna()
            
            # Ensure both series have the same length
            min_len = min(len(main_returns), len(pair_returns))
            if min_len < 20:  # Need at least 20 points for meaningful correlation
                continue
            
            # Calculate correlation
            correlation = main_returns.iloc[-min_len:].corr(pair_returns.iloc[-min_len:])
            
            # Only include strong correlations
            if abs(correlation) > 0.5:
                correlations[pair] = {
                    "correlation": correlation,
                    "strength": abs(correlation)
                }
        
        # Calculate average correlation
        if correlations:
            avg_correlation = sum(data["correlation"] for data in correlations.values()) / len(correlations)
        else:
            avg_correlation = 0
        
        return {
            "correlations": correlations,
            "average_correlation": avg_correlation
        }
    except Exception as e:
        logging.error(f"Error analyzing currency correlations: {e}")
        return {}

def optimize_strategy_parameters(symbol, days=30):
    """
    Optimize strategy parameters based on historical data.
    This function would be used periodically to fine-tune the strategy.
    """
    try:
        logging.info(f"Starting parameter optimization for {symbol}")
        
        # Get historical data
        now = datetime.now()
        start_date = now - timedelta(days=days)
        
        # Convert to timestamp
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(now.timestamp())
        
        # Get historical candles
        rates = mt5.copy_rates_range(symbol, MT5_TIMEFRAMES["PRIMARY"], start_timestamp, end_timestamp)
        if rates is None or len(rates) == 0:
            logging.error(f"Failed to get historical data for {symbol}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Define parameter ranges to test
        parameters = {
            'atr_period': [10, 14, 21],
            'rsi_period': [9, 14, 21],
            'ema_short_period': [20, 50, 100],
            'ema_long_period': [100, 200, 300],
            'adx_threshold': [15, 20, 25]
        }
        
        best_params = {}
        best_profit = 0
        best_trades = 0
        best_win_rate = 0
        
        # Simple grid search - in reality, you'd want a more sophisticated approach
        # This is a simplified version that would need to be expanded
        for atr_period in parameters['atr_period']:
            for rsi_period in parameters['rsi_period']:
                for ema_short in parameters['ema_short_period']:
                    for ema_long in parameters['ema_long_period']:
                        if ema_short >= ema_long:
                            continue  # Skip invalid combinations
                        
                        for adx_threshold in parameters['adx_threshold']:
                            # Set up test parameters
                            test_params = {
                                'atr_period': atr_period,
                                'rsi_period': rsi_period,
                                'ema_short_period': ema_short,
                                'ema_long_period': ema_long,
                                'adx_threshold': adx_threshold
                            }
                            
                            # Simulate trades with these parameters
                            total_profit, win_rate, trade_count = backtest_parameters(df, test_params)
                            
                            # Compare results
                            if trade_count > 5:  # Only consider parameter sets with enough trades
                                # Calculate a score combining profit and win rate
                                score = total_profit * (win_rate / 100)
                                
                                if score > best_profit:
                                    best_profit = score
                                    best_params = test_params
                                    best_trades = trade_count
                                    best_win_rate = win_rate
        
        if best_params:
            logging.info(f"Found optimal parameters for {symbol}: {best_params}")
            logging.info(f"Optimization results: Profit Score: {best_profit:.2f}, Win Rate: {best_win_rate:.2f}%, Trades: {best_trades}")
            return best_params
        else:
            logging.warning(f"Could not find optimal parameters for {symbol}. Using defaults.")
            return None
        
    except Exception as e:
        logging.error(f"Error optimizing strategy parameters: {e}")
        return None

def backtest_parameters(df, params):
    """
    Simple backtesting function to evaluate strategy parameters.
    This is called by the optimize_strategy_parameters function.
    """
    try:
        # Apply parameters
        atr_period = params['atr_period']
        rsi_period = params['rsi_period']
        ema_short_period = params['ema_short_period']
        ema_long_period = params['ema_long_period']
        adx_threshold = params['adx_threshold']
        
        # Calculate indicators with test parameters
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
        df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period)
        df['ema_short'] = talib.EMA(df['close'], timeperiod=ema_short_period)
        df['ema_long'] = talib.EMA(df['close'], timeperiod=ema_long_period)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Drop NaN values
        df = df.dropna()
        
        # Simulate trades
        positions = []
        open_position = None
        trades = []
        
        for i in range(1, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            
            # Check for signals
            buy_signal = (
                prev_bar['ema_short'] < prev_bar['ema_long'] and
                current_bar['ema_short'] > current_bar['ema_long'] and
                current_bar['adx'] > adx_threshold and
                current_bar['plus_di'] > current_bar['minus_di'] and
                current_bar['rsi'] > 40 and current_bar['rsi'] < 70
            )
            
            sell_signal = (
                prev_bar['ema_short'] > prev_bar['ema_long'] and
                current_bar['ema_short'] < current_bar['ema_long'] and
                current_bar['adx'] > adx_threshold and
                current_bar['minus_di'] > current_bar['plus_di'] and
                current_bar['rsi'] < 60 and current_bar['rsi'] > 30
            )
            
            # Execute trades
            if open_position is None:
                # Open new position
                if buy_signal:
                    entry_price = current_bar['close']
                    stop_loss = entry_price - 2 * current_bar['atr']
                    take_profit = entry_price + 3 * current_bar['atr']
                    open_position = {
                        'type': 'buy',
                        'entry_price': entry_price,
                        'entry_time': current_bar['time'],
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                elif sell_signal:
                    entry_price = current_bar['close']
                    stop_loss = entry_price + 2 * current_bar['atr']
                    take_profit = entry_price - 3 * current_bar['atr']
                    open_position = {
                        'type': 'sell',
                        'entry_price': entry_price,
                        'entry_time': current_bar['time'],
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
            else:
                # Check if position should be closed
                if open_position['type'] == 'buy':
                    # Check stop loss and take profit
                    if current_bar['low'] <= open_position['stop_loss']:
                        # Stop loss hit
                        exit_price = open_position['stop_loss']
                        profit = exit_price - open_position['entry_price']
                        trades.append({
                            'type': 'buy',
                            'entry_price': open_position['entry_price'],
                            'exit_price': exit_price,
                            'profit': profit,
                            'exit_reason': 'stop_loss'
                        })
                        open_position = None
                    elif current_bar['high'] >= open_position['take_profit']:
                        # Take profit hit
                        exit_price = open_position['take_profit']
                        profit = exit_price - open_position['entry_price']
                        trades.append({
                            'type': 'buy',
                            'entry_price': open_position['entry_price'],
                            'exit_price': exit_price,
                            'profit': profit,
                            'exit_reason': 'take_profit'
                        })
                        open_position = None
                elif open_position['type'] == 'sell':
                    # Check stop loss and take profit
                    if current_bar['high'] >= open_position['stop_loss']:
                        # Stop loss hit
                        exit_price = open_position['stop_loss']
                        profit = open_position['entry_price'] - exit_price
                        trades.append({
                            'type': 'sell',
                            'entry_price': open_position['entry_price'],
                            'exit_price': exit_price,
                            'profit': profit,
                            'exit_reason': 'stop_loss'
                        })
                        open_position = None
                    elif current_bar['low'] <= open_position['take_profit']:
                        # Take profit hit
                        exit_price = open_position['take_profit']
                        profit = open_position['entry_price'] - exit_price
                        trades.append({
                            'type': 'sell',
                            'entry_price': open_position['entry_price'],
                            'exit_price': exit_price,
                            'profit': profit,
                            'exit_reason': 'take_profit'
                        })
                        open_position = None
        
        # Calculate results
        if trades:
            total_profit = sum(trade['profit'] for trade in trades)
            winning_trades = sum(1 for trade in trades if trade['profit'] > 0)
            win_rate = (winning_trades / len(trades)) * 100
            return total_profit, win_rate, len(trades)
        else:
            return 0, 0, 0
    except Exception as e:
        logging.error(f"Error in backtest_parameters: {e}")
        return 0, 0, 0
    
def enhanced_trailing_stop(symbol, ticket, current_price, position_type, entry_price, atr_value=None):
    """
    Enhanced trailing stop with dynamic activation thresholds based on volatility.
    Much more aggressive in letting profits run in strong trends.
    """
    try:
        # Get current position
        positions = get_positions(symbol)
        position = next((p for p in positions if p.ticket == ticket), None)
        
        if position is None:
            logging.warning(f"Position {ticket} not found. Cannot apply trailing stop.")
            return None
        
        # Get current stop loss
        current_sl = position.sl
        
        # Get market data and calculate ATR if not provided
        if atr_value is None:
            optimized_tf = optimize_timeframe_for_volatility(symbol)
            df = get_candle_data(symbol, optimized_tf)
            if df is None:
                return current_sl
            atr_value = calculate_atr(df)
            
        # Get market regime for adaptive trailing
        market_regime, regime_strength = detect_market_regime(symbol)
            
        # Get symbol info
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return current_sl
            
        point = symbol_info.point
        
        # Calculate profit in points and percentage
        if position_type == mt5.ORDER_TYPE_BUY:
            profit_points = (current_price - entry_price) / point
        else:  # mt5.ORDER_TYPE_SELL
            profit_points = (entry_price - current_price) / point
            
        profit_percent = profit_points / (entry_price / point) * 100
        
        # Calculate position age in hours
        position_age_hours = (time.time() - position.time) / 3600  # Hours
        
        # Get volatility level (with caching if available)
        if 'get_cached_volatility' in globals() and callable(globals()['get_cached_volatility']):
            volatility_level = get_cached_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
        else:
            volatility_level, _ = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
        
        # IMPROVED: Dynamic activation threshold based on volatility
        base_activation = TSL_ACTIVATION_THRESHOLD * 100  # Convert to percentage
        
        # Adaptive threshold based on volatility and position age
        if volatility_level == "super-extreme":
            # Much lower threshold for super-extreme volatility
            adjusted_activation = max(0.15, base_activation * 0.4)  # As low as 0.15%
        elif volatility_level == "extreme":
            adjusted_activation = max(0.20, base_activation * 0.5)  # As low as 0.20%
        elif volatility_level == "high":
            adjusted_activation = max(0.25, base_activation * 0.6)  # As low as 0.25%
        else:
            adjusted_activation = max(0.30, base_activation * 0.8)  # As low as 0.30%
        
        # Further reduce threshold based on position age - more aggressive with time
        age_reduction = min(adjusted_activation * 0.5, position_age_hours * 0.1)
        final_activation = max(0.10, adjusted_activation - age_reduction)  # Never below 0.10%
        
        # Only log once per minute to reduce log spam
        should_log = int(time.time()) % 60 == 0
        
        if should_log or profit_percent >= final_activation * 0.8:  # Log if close to activation
            logging.info(f"Position {ticket} {symbol}: Profit {profit_percent:.2f}%, " + 
                      f"Activation threshold {final_activation:.2f}% (Age: {position_age_hours:.1f}h, Volatility: {volatility_level})")
        
        if profit_percent < final_activation:
            if should_log:
                logging.info(f"Position {ticket}: Profit below threshold. No trailing stop update.")
            return current_sl
        
        # Calculate trail factor based on market regime - MUCH MORE AGGRESSIVE
        regime_adjustment = 1.0
        
        # Check for aligned strong trend
        aligned_strong_trend = False
        if (market_regime == "STRONG_UPTREND" and position_type == mt5.ORDER_TYPE_BUY) or \
           (market_regime == "STRONG_DOWNTREND" and position_type == mt5.ORDER_TYPE_SELL):
            aligned_strong_trend = True
            
        # Set aggressive trail factors based on regime
        if aligned_strong_trend and regime_strength > 0.8:
            # In very strong aligned trend, use extremely loose trailing to maximize profits
            regime_adjustment = 2.0  # 100% wider trail - EXTREMELY AGGRESSIVE
            logging.info(f"Using extremely loose trailing (100% wider) to maximize profits in strong {market_regime}")
        elif aligned_strong_trend and regime_strength > 0.6:
            # In strong aligned trend, use very loose trailing
            regime_adjustment = 1.7  # 70% wider trail - VERY AGGRESSIVE
            logging.info(f"Using very loose trailing (70% wider) to let profits run in strong {market_regime}")
        elif aligned_strong_trend:
            # In aligned trend, use looser trailing
            regime_adjustment = 1.4  # 40% wider trail - AGGRESSIVE
            logging.info(f"Using looser trailing (40% wider) to let profits run in {market_regime}")
        
        # Set trail factor based on profit zones
        # 1. Initial profit zone (just activated trailing)
        if profit_percent < base_activation * 1.5:
            trail_factor = 1.0 * regime_adjustment  # Very loose trailing initially - 100% of ATR - MORE AGGRESSIVE
            
        # 2. Solid profit zone (1.5x-2.5x activation threshold)
        elif profit_percent < base_activation * 2.5:
            trail_factor = 0.8 * regime_adjustment  # Looser trailing - 80% of ATR - MORE AGGRESSIVE
            
        # 3. Strong profit zone (2.5x-4x activation threshold)
        elif profit_percent < base_activation * 4:
            trail_factor = 0.6 * regime_adjustment  # Looser trailing - 60% of ATR - MORE AGGRESSIVE
            
        # 4. Exceptional profit zone (>4x activation threshold)
        else:
            # Still apply regime adjustment but tighter overall
            trail_factor = 0.5 * regime_adjustment  # Tighter trailing - 50% of ATR
        
        # Never let trail get too tight in a strong trend
        if aligned_strong_trend and trail_factor < 0.6:
            trail_factor = 0.6  # Minimum trail factor in strong trends
            
        # Adjust for market volatility
        if volatility_level == "super-extreme":
            trail_factor *= 1.4  # Even wider trail in super-extreme volatility
        elif volatility_level == "extreme":
            trail_factor *= 1.3  # Wider trail in extreme volatility
        elif volatility_level == "high":
            trail_factor *= 1.2  # Wider trail in high volatility
        
        # Calculate trail distance
        trail_distance = atr_value * trail_factor
        
        # Calculate new stop loss
        if position_type == mt5.ORDER_TYPE_BUY:
            new_sl = current_price - trail_distance
            
            # Only move stop loss up
            if new_sl <= current_sl:
                return current_sl
        else:  # mt5.ORDER_TYPE_SELL
            new_sl = current_price + trail_distance
            
            # Only move stop loss down
            if new_sl >= current_sl:
                return current_sl
        
        # IMPORTANT: Profit protection only on exceptional gains
        # Ensure minimum profit is secured on huge gains (4x activation)
        if profit_percent > base_activation * 4 and position_type == mt5.ORDER_TYPE_BUY:
            min_sl = entry_price + (profit_points * point * 0.3)  # Secure 30% of gain
            if new_sl < min_sl:
                new_sl = min_sl
                logging.info(f"Secured minimum 30% of exceptional gain for position {ticket}")
        elif profit_percent > base_activation * 4 and position_type == mt5.ORDER_TYPE_SELL:
            min_sl = entry_price - (profit_points * point * 0.3)  # Secure 30% of gain
            if new_sl > min_sl:
                new_sl = min_sl
                logging.info(f"Secured minimum 30% of exceptional gain for position {ticket}")
        
        # Round to symbol precision
        digits = symbol_info.digits
        new_sl = round(new_sl, digits)
        
        # Log the update
        if new_sl != current_sl:
            if position_type == mt5.ORDER_TYPE_BUY:
                protected_pips = (new_sl - entry_price) / point
                protected_percent = (protected_pips / profit_points) * 100 if profit_points > 0 else 0
                logging.info(f"Enhanced TSL Update for {symbol} position {ticket}: " +
                            f"Current SL: {current_sl} → New SL: {new_sl} " +
                            f"(Profit: {profit_percent:.2f}%, Protecting: {protected_percent:.1f}%)")
            else:
                protected_pips = (entry_price - new_sl) / point
                protected_percent = (protected_pips / profit_points) * 100 if profit_points > 0 else 0
                logging.info(f"Enhanced TSL Update for {symbol} position {ticket}: " +
                            f"Current SL: {current_sl} → New SL: {new_sl} " +
                            f"(Profit: {profit_percent:.2f}%, Protecting: {protected_percent:.1f}%)")
        
        return new_sl
        
    except Exception as e:
        logging.error(f"Error in enhanced trailing stop: {e}")
        return None
    
def advanced_trailing_stop(symbol, ticket, current_price, position_type, entry_price, atr_value=None):
    """
    Advanced trailing stop that adapts to market volatility and trend strength.
    More intelligent than the standard trailing stop.
    """
    try:
        # Get current position
        positions = get_positions(symbol)
        position = next((p for p in positions if p.ticket == ticket), None)
        
        if position is None:
            logging.warning(f"Position {ticket} not found. Cannot apply trailing stop.")
            return None
        
        # Get current stop loss
        current_sl = position.sl
        
        # Get market data
        optimized_tf = optimize_timeframe_for_volatility(symbol)
        df = get_candle_data(symbol, optimized_tf)
        if df is None:
            return current_sl
        
        # Calculate ATR if not provided
        if atr_value is None:
            atr_value = calculate_atr(df)
        
        # Get market structure
        market_state = analyze_market_structure(symbol)
        trend_strength = market_state.get("trend_strength", 0)
        volatility = market_state.get("volatility", "normal")
        
        # Calculate profit
        symbol_info = get_symbol_info(symbol)
        point = symbol_info.point
        
        if position_type == mt5.ORDER_TYPE_BUY:
            profit_points = (current_price - entry_price) / point
        else:  # mt5.ORDER_TYPE_SELL
            profit_points = (entry_price - current_price) / point
        
        # Log profit information for debugging
        logging.info(f"Position {ticket} profit points: {profit_points}, ATR in points: {atr_value/point}")
        
        # Only activate after certain profit threshold - REDUCED from 1.5 to 1.0 ATR
        activation_threshold = atr_value / point * 1.0  # 1.0 ATR (was 1.5)
        if profit_points < activation_threshold:
            logging.info(f"Position {ticket}: Profit points {profit_points} below activation threshold {activation_threshold}. No trailing stop update.")
            return current_sl
        
        logging.info(f"Position {ticket}: Profit points {profit_points} above activation threshold {activation_threshold}. Calculating new trailing stop.")
        
        # Calculate different stop distance based on market conditions and profit level
        profit_level = profit_points / (entry_price / point) * 100
        
        # Base trail factor
        if profit_level < 1.0:
            base_trail_factor = 0.9  # Keep 90% of risk on during initial profit
        elif profit_level < 2.0:
            base_trail_factor = 0.75
        elif profit_level < 3.0:
            base_trail_factor = 0.6
        elif profit_level < 5.0:
            base_trail_factor = 0.4
        else:
            base_trail_factor = 0.25  # Tighter trailing stop at high profit levels
        
        # Adjust based on trend strength and volatility
        if trend_strength > 0.7:
            # Strong trend - looser stop
            trail_factor = base_trail_factor * 1.2
        elif volatility == "high" or volatility == "extreme":
            # High volatility - wider stop
            trail_factor = base_trail_factor * 1.3
        else:
            trail_factor = base_trail_factor
        
        # Calculate trail distance
        trail_distance = atr_value * trail_factor
        
        # Calculate new stop loss
        if position_type == mt5.ORDER_TYPE_BUY:
            new_sl = current_price - trail_distance
            # Only move stop loss up
            if new_sl <= current_sl:
                logging.info(f"Position {ticket}: New SL {new_sl} not better than current SL {current_sl}. No update needed.")
                return current_sl
        else:  # mt5.ORDER_TYPE_SELL
            new_sl = current_price + trail_distance
            # Only move stop loss down
            if new_sl >= current_sl:
                logging.info(f"Position {ticket}: New SL {new_sl} not better than current SL {current_sl}. No update needed.")
                return current_sl
        
        # Round to symbol precision
        digits = symbol_info.digits
        new_sl = round(new_sl, digits)
        
        logging.info(f"Position {ticket}: New trailing stop calculated: {new_sl} (current: {current_sl}, trail factor: {trail_factor})")
        return new_sl
    except Exception as e:
        logging.error(f"Error in advanced trailing stop: {e}")
        return None
    
def manage_position_with_partial_close(symbol, ticket):
    """
    Manage position with partial profit-taking at key levels.
    This provides more sophisticated money management.
    """
    try:
        # Get position info
        positions = get_positions(symbol)
        position = next((p for p in positions if p.ticket == ticket), None)
        
        if position is None:
            return
        
        # Get market data
        optimized_tf = optimize_timeframe_for_volatility(symbol)
        df = get_candle_data(symbol, optimized_tf)
        if df is None:
            return
        
        # Calculate profit
        symbol_info = get_symbol_info(symbol)
        point = symbol_info.point
        current_price = mt5.symbol_info_tick(symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
        entry_price = position.price_open
        
        if position.type == mt5.ORDER_TYPE_BUY:
            profit_points = (current_price - entry_price) / point
        else:  # mt5.ORDER_TYPE_SELL
            profit_points = (entry_price - current_price) / point
        
        profit_percent = profit_points / (entry_price / point) * 100

        
        # Define profit levels for partial closing
        if profit_percent >= 3.0 and position.volume > symbol_info.volume_min * 2:
            # Close 50% at 3% profit
            partial_volume = position.volume * 0.5
            partial_volume = round(partial_volume / symbol_info.volume_step) * symbol_info.volume_step
            
            if partial_volume >= symbol_info.volume_min:
                close_position(ticket, partial_volume)
                logging.info(f"Partial close (50%) on position {ticket} at {profit_percent:.2f}% profit")
                
                # Adjust stop loss to entry for remaining position
                if position.sl != entry_price:
                    update_position_stops(symbol, ticket, entry_price)
                    logging.info(f"Updated stop loss to break-even for position {ticket}")
        
        elif profit_percent >= 1.5 and position.sl < entry_price and position.type == mt5.ORDER_TYPE_BUY:
            # Move stop loss to entry at 1.5% profit for buy positions
            update_position_stops(symbol, ticket, entry_price)
            logging.info(f"Updated stop loss to break-even for position {ticket}")
        
        elif profit_percent >= 1.5 and position.sl > entry_price and position.type == mt5.ORDER_TYPE_SELL:
            # Move stop loss to entry at 1.5% profit for sell positions
            update_position_stops(symbol, ticket, entry_price)
            logging.info(f"Updated stop loss to break-even for position {ticket}")
        
    except Exception as e:
        logging.error(f"Error in manage_position_with_partial_close: {e}")

def evaluate_risk_reward(symbol, order_type, entry_price, stop_loss, take_profit):
    """
    Evaluate risk-reward ratio and potential for a trade.
    Returns a dictionary with risk assessment details.
    """
    try:
        # Get symbol info first since we need it for calculations
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return {
                "risk_reward_ratio": 0,
                "meets_min_ratio": False,
                "reason": "Symbol info not available"
            }
        
        # Input validation for buy orders
        if order_type == "BUY":
            if entry_price <= stop_loss:
                return {
                    "risk_reward_ratio": 0,
                    "meets_min_ratio": False,
                    "reason": f"Invalid stop loss: must be below entry price for BUY orders"
                }
            if take_profit <= entry_price:
                return {
                    "risk_reward_ratio": 0,
                    "meets_min_ratio": False,
                    "reason": f"Invalid take profit: must be above entry price for BUY orders"
                }
            risk_pips = entry_price - stop_loss
            reward_pips = take_profit - entry_price
        # Input validation for sell orders
        else:  # "SELL"
            if stop_loss <= entry_price:
                return {
                    "risk_reward_ratio": 0,
                    "meets_min_ratio": False,
                    "reason": f"Invalid stop loss: must be above entry price for SELL orders"
                }
            if entry_price <= take_profit:
                return {
                    "risk_reward_ratio": 0,
                    "meets_min_ratio": False,
                    "reason": f"Invalid take profit: must be below entry price for SELL orders"
                }
            risk_pips = stop_loss - entry_price
            reward_pips = entry_price - take_profit
        
        # Additional validation to prevent zero or negative values
        if risk_pips <= 0:
            return {
                "risk_reward_ratio": 0,
                "meets_min_ratio": False,
                "reason": f"Invalid risk: {risk_pips} pips"
            }
        
        if reward_pips <= 0:
            return {
                "risk_reward_ratio": 0,
                "meets_min_ratio": False,
                "reason": f"Invalid reward: {reward_pips} pips"
            }
        
        # Calculate risk-reward ratio
        risk_reward_ratio = reward_pips / risk_pips
        
        point = symbol_info.point
        
        # Calculate risk and reward in points
        risk_points = risk_pips / point
        reward_points = reward_pips / point
        
        # Safe property access
        if not hasattr(symbol_info, 'trade_tick_value') or not hasattr(symbol_info, 'trade_tick_size'):
            return {
                "risk_reward_ratio": risk_reward_ratio,
                "meets_min_ratio": risk_reward_ratio >= MIN_RISK_REWARD_RATIO,
                "reason": "Cannot calculate profit: symbol info incomplete"
            }
        
        # Calculate minimum viable profit check
        pip_value = symbol_info.trade_tick_value / symbol_info.trade_tick_size
        lot_size = 0.01  # Default minimum lot size
        expected_profit = reward_pips * pip_value * lot_size
        min_viable_profit = calculate_minimum_viable_profit(symbol, lot_size)
        
        # ENHANCED: Adjust minimum viable profit for super-extreme volatility
        volatility_level, atr_percent = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
        if volatility_level == "super-extreme":
            # Reduce minimum viable profit requirement for super-extreme volatility
            min_viable_profit *= 0.8  # 20% reduction
        
        # Check if expected profit meets minimum viable
        profit_sufficient = expected_profit >= min_viable_profit
        
        # Add this volatility-based adjustment before checking if risk-reward meets minimum
        adjusted_min_ratio = MIN_RISK_REWARD_RATIO
        
        # ENHANCED: Adjust minimum required ratio based on volatility
        if volatility_level == "high":
            adjusted_min_ratio = MIN_RISK_REWARD_RATIO * 0.8  # 20% reduction for high volatility
        elif volatility_level == "extreme":
            adjusted_min_ratio = MIN_RISK_REWARD_RATIO * 0.6  # 40% reduction for extreme volatility
        elif volatility_level == "super-extreme":
            adjusted_min_ratio = MIN_RISK_REWARD_RATIO * 0.5  # 50% reduction for super-extreme volatility
            
        # ENHANCED: Add special handling for Bitcoin
        is_bitcoin = symbol.startswith("BTC") or "BTC" in symbol
        if is_bitcoin and volatility_level in ["extreme", "super-extreme"]:
            # Further reduce minimum RR ratio for Bitcoin in high volatility
            adjusted_min_ratio *= 0.9  # Additional 10% reduction
        
        # Then modify the meets_min_ratio calculation to use the adjusted ratio
        meets_min_ratio = risk_reward_ratio >= adjusted_min_ratio and profit_sufficient
        
        result = {
            "risk_pips": risk_pips,
            "reward_pips": reward_pips,
            "risk_points": risk_points,
            "reward_points": reward_points,
            "risk_reward_ratio": risk_reward_ratio,
            "expected_profit": expected_profit,
            "min_viable_profit": min_viable_profit,
            "profit_sufficient": profit_sufficient,
            "meets_min_ratio": meets_min_ratio,
            "adjusted_min_ratio": adjusted_min_ratio,
            "volatility_level": volatility_level
        }
        
        if not profit_sufficient:
            result["reason"] = f"Expected profit ({expected_profit:.2f}) below minimum viable ({min_viable_profit:.2f})"
        elif not meets_min_ratio:
            result["reason"] = f"Risk-reward ratio {risk_reward_ratio:.2f} below adjusted minimum {adjusted_min_ratio:.2f}"
            
        return result
    except Exception as e:
        logging.error(f"Error evaluating risk-reward: {e}")
        return {
            "risk_reward_ratio": 0,
            "meets_min_ratio": False,
            "reason": f"Error: {str(e)}"
        }
    
def detect_sudden_price_movement(symbol, time_window=5):
    """
    Detect sudden price movements that might indicate market shocks.
    Uses asset-specific thresholds.
    """
    try:
        # Get candle data for the last 'time_window' minutes
        df = get_candle_data(symbol, mt5.TIMEFRAME_M1, num_bars=time_window + 10)
        if df is None or len(df) < time_window:
            return False, 0
        
        # Calculate price range within time window
        recent_bars = df.iloc[-time_window:]
        price_high = recent_bars['high'].max()
        price_low = recent_bars['low'].min()
        
        # Calculate range as percentage of current price
        current_price = df['close'].iloc[-1]
        price_range_pct = (price_high - price_low) / current_price * 100
        
        # Select the appropriate threshold based on the asset type
        if symbol.startswith("BTC") or symbol.startswith("ETH"):
            movement_threshold = CRYPTO_MOVEMENT_THRESHOLD
        elif symbol.startswith("US") or symbol.startswith("DE"):
            movement_threshold = INDICES_MOVEMENT_THRESHOLD
        else:
            movement_threshold = FOREX_MOVEMENT_THRESHOLD
        
        # Detect if current range is much larger than threshold
        is_sudden_movement = price_range_pct > movement_threshold
        
        return is_sudden_movement, price_range_pct
    except Exception as e:
        logging.error(f"Error detecting sudden price movement: {e}")
        return False, 0

def calculate_consolidated_signal(symbol):
    """
    Calculate a consolidated signal from multiple indicators with improved filtering
    and dynamic thresholds.
    """
    try:
        # At the beginning of calculate_consolidated_signal
        buy_strength = 0
        sell_strength = 0
        pattern_data = {}  # Initialize this too if needed
        with state.locks['signals']:
            # Get individual strategy signals
            trend_signal = trend_following_strategy(symbol)
            breakout_signal = breakout_strategy(symbol)
            reversal_signal = reversal_strategy(symbol)
            
            # Get market structure and momentum
            market_state = analyze_market_structure_enhanced(symbol)
            momentum_data = calculate_multi_timeframe_momentum(symbol)
            correlations = analyze_currency_correlations(symbol)

            # Get advanced chart patterns
            pattern_data = detect_advanced_chart_patterns(symbol)
            
            # Get market regime for adaptive filtering
            market_regime, regime_strength = detect_market_regime(symbol)
            
            # Get session liquidity
            high_liquidity = is_high_liquidity_session(symbol)
            
            # Check for sudden price movements
            sudden_movement, price_range_pct = detect_sudden_price_movement(symbol)
            
            # Get volatility level
            volatility_level = market_state.get("volatility", "normal")
            
            # Set base threshold based on market regime
            if market_regime == "STRONG_TREND":
                base_threshold = 1.3  # Lower threshold for strong trends
            elif market_regime == "CHOPPY_VOLATILE":
                base_threshold = 1.8  # Higher threshold for choppy markets
            elif market_regime == "RANGE_BOUND":
                # MODIFIED: Lower threshold for range-bound markets
                base_threshold = 1.4  # Reduced from 1.5
            else:
                base_threshold = 1.5  # Default threshold

            
            # Adjust for volatility
            if volatility_level == "super-extreme":
                # MODIFIED: Reduced threshold for super-extreme volatility
                threshold = base_threshold * 1.3  # Reduced from 1.6
                buy_threshold_multiplier = 1.1  # Reduced from 1.15
            elif volatility_level == "extreme":
                threshold = base_threshold * 1.4  # Higher threshold for extreme volatility
                buy_threshold_multiplier = 1.1  # Make BUY signals require 10% more strength
            else:
                threshold = base_threshold
                buy_threshold_multiplier = 1.0
            
            # Initialize signal strength values
            buy_strength = 0
            sell_strength = 0

            # Add special case for Bitcoin SELL signals in high volatility
            is_bitcoin = symbol.startswith("BTC") or "BTC" in symbol
            volatility_level, _ = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])

            # Check if there are head_shoulders patterns which often signal reversals
            has_reversal_patterns = len(set(pattern_data.get("patterns_found", {}).keys()).intersection(
                {"head_shoulders", "double_formation", "triple_formation"})) > 0

            if sell_strength > buy_strength and is_bitcoin and volatility_level in ["high", "extreme", "super-extreme"]:
                if has_reversal_patterns:
                    # Much lower threshold for SELL signals with reversal patterns in volatile Bitcoin
                    base_threshold *= 0.7  # 30% reduction for reversal patterns
                    logging.info(f"Reducing SELL threshold by 30% due to reversal patterns and {volatility_level} volatility")
                        
            
            # Add adaptive scoring based on market regime
            regime_multiplier = {
                "STRONG_TREND": 1.5,    # Boost trend signals in trending markets
                "RANGE_BOUND": 0.7,     # Reduce trend signals in ranging markets
                "CHOPPY_VOLATILE": 0.9, # Slightly reduce all signals in choppy markets
            }.get(market_regime, 1.0)
            
            # Add trend signal contribution with regime adaptation
            if trend_signal["signal"] == "BUY":
                buy_strength += 3 * trend_signal.get("trend_strength", 0.5) * regime_multiplier
            elif trend_signal["signal"] == "SELL":
                sell_strength += 3 * trend_signal.get("trend_strength", 0.5) * regime_multiplier
            
            # Add breakout signal contribution
            if breakout_signal["signal"] == "BUY":
                # Enhance breakouts in ranging markets
                breakout_multiplier = 1.5 if market_regime == "RANGE_BOUND" else 1.0
                buy_strength += 3 * breakout_multiplier
            elif breakout_signal["signal"] == "SELL":
                breakout_multiplier = 1.5 if market_regime == "RANGE_BOUND" else 1.0
                sell_strength += 3 * breakout_multiplier
            
            # Add reversal signal contribution
            if reversal_signal["signal"] == "BUY":
                # Enhance reversals in volatile markets
                reversal_multiplier = 1.3 if volatility_level in ["high", "extreme"] else 1.0
                buy_strength += 1.5 * reversal_multiplier
            elif reversal_signal["signal"] == "SELL":
                reversal_multiplier = 1.3 if volatility_level in ["high", "extreme"] else 1.0
                sell_strength += 1.5 * reversal_multiplier
            
            # Check candlestick patterns for additional confirmation
            candlestick_patterns = detect_candlestick_patterns(get_candle_data(symbol, MT5_TIMEFRAMES["PRIMARY"]))
            
            if candlestick_patterns.get("bullish_count", 0) >= 2:
                buy_strength += 1.5
            if candlestick_patterns.get("bearish_count", 0) >= 2:
                sell_strength += 1.5
            
            # Add market structure contribution
            if market_state.get("swing_trend") == "bullish":
                buy_strength += 1.2
            elif market_state.get("swing_trend") == "bearish":
                sell_strength += 1.2

            # Add chart pattern contribution
            pattern_direction = pattern_data.get("direction", "neutral")
            pattern_strength = pattern_data.get("strength", 0)
            
            # ENHANCED: Increase pattern contribution for super-extreme volatility
            pattern_volatility_multiplier = 1.5 if volatility_level == "super-extreme" else 1.0
            
            # Add pattern strength only if it exceeds a minimum threshold
            if pattern_strength > 0.3:
                if pattern_direction == "bullish":
                    buy_strength += pattern_strength * 3 * pattern_volatility_multiplier
                    logging.info(f"Adding bullish pattern strength: {pattern_strength * 3 * pattern_volatility_multiplier:.2f}")
                elif pattern_direction == "bearish":
                    # Increase sell strength for head and shoulders pattern - common in tops
                    has_head_shoulders = 'head_shoulders' in patterns_found
                    bearish_bonus = 1.5 if has_head_shoulders else 1.0
                    
                    # Add extra strength for Bitcoin in volatile conditions
                    if is_bitcoin and (volatility_level in ["high", "extreme", "super-extreme"]):
                        bearish_bonus *= 1.5  # 50% more sell strength for Bitcoin in volatile market
                        
                    # Apply enhanced multipliers
                    enhanced_sell_strength = pattern_strength * 3 * pattern_volatility_multiplier * bearish_bonus
                    sell_strength += enhanced_sell_strength
                    logging.info(f"Adding enhanced bearish pattern strength: {enhanced_sell_strength:.2f} (with bonus: {bearish_bonus:.1f})")
            
            # Log found patterns for debugging
            patterns_found = pattern_data.get("patterns_found", {})
            if patterns_found:
                pattern_names = list(patterns_found.keys())
                logging.info(f"Patterns found for {symbol}: {pattern_names}")

            # Fix the has_head_shoulders check
            has_head_shoulders = 'head_shoulders' in patterns_found  # Now patterns_found is defined
        
            # Add momentum contribution with enhanced weighting
            momentum_consensus = momentum_data.get("consensus", {}).get("momentum", "neutral")
            momentum_strength = momentum_data.get("consensus", {}).get("strength", 0)
            
            if momentum_consensus == "bullish":
                buy_strength += momentum_strength * 2.5
            elif momentum_consensus == "bearish":
                sell_strength += momentum_strength * 2.5
            
            # Apply correlation factor
            avg_correlation = correlations.get("average_correlation", 0)
            if abs(avg_correlation) > 0.7:
                # High correlation - consider the impact
                if avg_correlation > 0:
                    # Positive correlation - check for conflicting signals
                    if buy_strength > sell_strength and correlations.get("correlation_signal", "neutral") == "bearish":
                        buy_strength *= 0.7  # Reduce buy strength
                    elif sell_strength > buy_strength and correlations.get("correlation_signal", "neutral") == "bullish":
                        sell_strength *= 0.7  # Reduce sell strength
                else:
                    # Negative correlation - inverse relationship
                    if buy_strength > sell_strength and correlations.get("correlation_signal", "neutral") == "bullish":
                        buy_strength *= 0.7  # Reduce buy strength
                    elif sell_strength > buy_strength and correlations.get("correlation_signal", "neutral") == "bearish":
                        sell_strength *= 0.7  # Reduce sell strength
            
            # Reduce signal strength in low liquidity conditions
            if not high_liquidity:
                buy_strength *= 0.7
                sell_strength *= 0.7
                logging.info(f"Reduced signal strength due to low liquidity session")
            
            # Reduce signal strength during sudden price movements
            if sudden_movement:
                buy_strength *= 0.5
                sell_strength *= 0.5
                logging.info(f"Reduced signal strength due to sudden price movement: {price_range_pct:.2f}%")
            
            # Signal consistency check
            if hasattr(state, 'current_signals') and symbol in state.current_signals:
                prev_signal = state.current_signals[symbol]
                if isinstance(prev_signal, dict):
                    prev_signal_type = prev_signal.get("signal", "NONE")
                    prev_signal_strength = prev_signal.get("strength", 0)
                    
                    # Require much stronger signal to flip direction
                    if prev_signal_type == "BUY" and sell_strength > buy_strength:
                        required_strength = prev_signal_strength * (2.2 if volatility_level in ["high", "extreme"] else 2.0)
                        if sell_strength / 10 < required_strength:
                            logging.info(f"Rejecting SELL signal - insufficient strength to flip from recent BUY")
                            sell_strength = 0
                    
                    elif prev_signal_type == "SELL" and buy_strength > sell_strength:
                        required_strength = prev_signal_strength * (2.2 if volatility_level in ["high", "extreme"] else 2.0)
                        if buy_strength / 10 < required_strength:
                            logging.info(f"Rejecting BUY signal - insufficient strength to flip from recent SELL")
                            buy_strength = 0
            
            # Apply margin health check
            account_info = get_account_info()
            if account_info and account_info.margin_level is not None:
                if account_info.margin_level < 1000:  # Below 1000% margin level
                    threshold *= 1.2  # Require 20% stronger signals
            
            # Determine final signal
            signal = "NONE"
            
            # ENHANCED: Special handling for Bitcoin in extreme volatility
            is_bitcoin = symbol.startswith("BTC") or "BTC" in symbol

            # Get price data for reversal checks
            optimized_tf = optimize_timeframe_for_volatility(symbol)
            price_df = get_candle_data(symbol, optimized_tf)

            # Check for distinct chart patterns that indicate reversals
            has_reversal_patterns = any(pattern in pattern_data.get("patterns_found", {}) 
                                    for pattern in ["head_shoulders", "double_formation", "triple_formation"])

            # Check for recent price action that might indicate a sell opportunity
            price_falling = False
            if price_df is not None and len(price_df) >= 3:
                recent_prices = price_df['close'].iloc[-3:].values
                price_falling = recent_prices[0] > recent_prices[1] > recent_prices[2]

            if is_bitcoin and volatility_level in ["extreme", "super-extreme"]:
                # More relaxed conditions for Bitcoin in extreme volatility
                if buy_strength > (threshold * 0.85) and buy_strength > sell_strength * 1.3:
                    signal = "BUY"
                    strength = buy_strength / 10
                elif (sell_strength > threshold * 0.75 and sell_strength > buy_strength * 1.2) or \
                    (has_reversal_patterns and sell_strength > threshold * 0.65 and price_falling):
                    # Much more sensitive SELL detection with pattern confirmation
                    signal = "SELL"
                    strength = sell_strength / 10
                    logging.info(f"Enhanced SELL signal detected with strength: {strength:.2f}, patterns: {list(pattern_data.get('patterns_found', {}).keys())}")
                else:
                    signal = "NONE"
                    strength = 0
            elif buy_strength > (threshold * buy_threshold_multiplier) and buy_strength > sell_strength * 1.65:
                signal = "BUY"
                strength = buy_strength / 10
            elif sell_strength > threshold and sell_strength > buy_strength * 1.5:
                signal = "SELL"
                strength = sell_strength / 10
            # Add special case for SELL with reversal patterns
            elif has_reversal_patterns and sell_strength > threshold * 0.8 and sell_strength > buy_strength and price_falling:
                signal = "SELL"
                strength = sell_strength / 10
                logging.info(f"Pattern-based SELL signal with strength: {strength:.2f}, patterns: {list(pattern_data.get('patterns_found', {}).keys())}")
            else:
                signal = "NONE"
                strength = 0

            
            # Check if signal passes minimum profitability requirements
            if signal != "NONE":
                # Get current market conditions
                current_spread = get_current_spread(symbol)
                transaction_costs = estimate_transaction_costs(symbol)
                
                # Calculate expected profit potential
                if signal == "BUY":
                    base_signal = trend_signal if trend_signal["signal"] == "BUY" else breakout_signal if breakout_signal["signal"] == "BUY" else reversal_signal
                else:  # SELL
                    base_signal = trend_signal if trend_signal["signal"] == "SELL" else breakout_signal if breakout_signal["signal"] == "SELL" else reversal_signal
                    
                # CRITICAL IMPROVEMENT: Estimate expected profit vs costs with enhanced thresholds
                expected_profit_factor = estimate_profit_potential(symbol, signal, base_signal)
                
                # MODIFIED: Reduced the threshold for profit vs costs evaluation
                # For Bitcoin in super-extreme volatility, use an even lower threshold
                profit_cost_multiplier = 1.0 if (is_bitcoin and volatility_level == "super-extreme") else 1.2  # Reduced from 1.5
                
                # Log detailed profit vs. cost information for better analysis
                logging.info(f"Profit potential analysis for {symbol} {signal}: " +
                            f"Expected profit factor: {expected_profit_factor:.2f}, " +
                            f"Transaction costs: {transaction_costs:.5f}, " +
                            f"Spread: {current_spread:.1f} pips, " +
                            f"Threshold multiplier: {profit_cost_multiplier:.1f}")
                
                # Only proceed if expected profit exceeds costs by significant margin
                if expected_profit_factor < (profit_cost_multiplier * (transaction_costs + current_spread * 0.0001)):  # Added conversion factor for spread
                    logging.info(f"Rejecting {signal} signal - insufficient profit potential vs costs")
                    return {"signal": "NONE", "reason": "Insufficient profit potential vs costs"}
            
            # If we have a signal, calculate entry, SL, and TP
            if signal != "NONE":
                # Entry price
                entry_price = get_optimal_entry_price(symbol, signal)
                
                # Calculate stop loss and take profit with enhanced key level integration
                stop_loss, take_profit = calculate_advanced_stop_loss_take_profit(symbol, signal, entry_price, market_state)
                
                # Add null check for stop_loss and take_profit
                if stop_loss is None or take_profit is None:
                    logging.warning(f"SL/TP calculation failed for {symbol}, using default values")
                    # Set default values based on order type
                    if signal == "BUY":
                        stop_loss = entry_price * 0.98  # 2% below entry
                        take_profit = entry_price * 1.04  # 4% above entry
                    else:  # SELL
                        stop_loss = entry_price * 1.02  # 2% above entry
                        take_profit = entry_price * 0.96  # 4% below entry
                        
                # Calculate optimal lot size with enhanced risk adjustment
                dynamic_risk = calculate_dynamic_risk_percentage(symbol, market_regime)
                
                # Get symbol info for lot size calculation
                symbol_info = get_symbol_info(symbol)
                if symbol_info is None:
                    return {"signal": "NONE", "reason": "Symbol info not available"}

                # Calculate stop loss distance in pips
                if signal == "BUY":
                    stop_loss_pips = (entry_price - stop_loss) / symbol_info.point
                else:  # "SELL"
                    stop_loss_pips = (stop_loss - entry_price) / symbol_info.point
                
                # Calculate lot size with advanced risk management
                lot_size = calculate_risk_adjusted_lot_size(symbol, dynamic_risk, stop_loss_pips, market_regime)
                
                # Evaluate risk-reward with minimum viable profit check
                risk_reward = evaluate_risk_reward(symbol, signal, entry_price, stop_loss, take_profit)
                
                # Add detailed signal information
                signal_data = {
                    "signal": signal,
                    "strength": strength,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "lot_size": lot_size,
                    "risk_reward": risk_reward,
                    "buy_strength": buy_strength / 10,
                    "sell_strength": sell_strength / 10,
                    "source": "consolidated",
                    "market_regime": market_regime,
                    "volatility": volatility_level,
                    "momentum": momentum_consensus,
                    "patterns": candlestick_patterns.get("overall_sentiment", "neutral"),
                    "high_liquidity": high_liquidity,
                    "sudden_movement": sudden_movement,
                    "chart_patterns": pattern_data
                }

                return signal_data
            else:
                # Add more informative rejection reason
                buy_score = buy_strength / 10 if buy_strength > 0 else 0
                sell_score = sell_strength / 10 if sell_strength > 0 else 0
                
                logging.info(f"No signal for {symbol}: Buy strength: {buy_score:.2f}, " +
                        f"Sell strength: {sell_score:.2f}, Threshold: {threshold/10:.2f}, " +
                        f"Volatility: {volatility_level}, Regime: {market_regime}")
                
                # Create a proper signal_data object for NONE signals
                signal_data = {
                    "signal": "NONE", 
                    "strength": 0, 
                    "reason": "Insufficient signal strength",
                    "buy_strength": buy_score,
                    "sell_strength": sell_score
                }
                
                # Store current signal with locking
                state.current_signals[symbol] = signal_data
                
                return signal_data

    except Exception as e:
        logging.error(f"Error calculating consolidated signal: {e}")
        return {"signal": "NONE", "reason": f"Error: {str(e)}"}
    
# ======================================================
# MT5 CONNECTION FUNCTIONS
# ======================================================

def connect_to_mt5():
    """
    Connect to MetaTrader 5 platform.
    Retries if connection fails.
    """
    for attempt in range(MAX_RETRIES):
        try:
            # Close any existing connection
            if mt5.terminal_info() is not None:
                mt5.shutdown()
                time.sleep(1)
            
            # Initialize connection
            if mt5.initialize():
                logging.info("Successfully connected to MT5")
                return True
            
            logging.error(f"Failed to initialize MT5. Error: {mt5.last_error()}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            logging.error(f"Exception during MT5 connection: {e}")
            time.sleep(RETRY_DELAY)
    
    logging.error("Failed to connect to MT5 after multiple attempts. Exiting script.")
    return False

def login_to_mt5():
    """
    Login to MetaTrader 5 account.
    Uses account credentials from config or environment variables.
    """
    for attempt in range(MAX_RETRIES):
        try:
            # If using environment variables
            if USE_ENV_VARS:
                account_number = int(os.getenv("MT5_ACCOUNT_NUMBER", ACCOUNT_NUMBER))
                password = os.getenv("MT5_PASSWORD", PASSWORD)
                server = os.getenv("MT5_SERVER", SERVER)
            else:
                account_number = int(ACCOUNT_NUMBER)
                password = PASSWORD
                server = SERVER
            
            # Attempt login
            if mt5.login(account_number, password, server):
                logging.info(f"Successfully logged in to MT5 account {account_number} on {server}")
                return True
            
            logging.error(f"MT5 login failed. Error code: {mt5.last_error()}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            logging.error(f"Exception during MT5 login: {e}")
            time.sleep(RETRY_DELAY)
    
    logging.error("Failed to log in to MT5 after multiple attempts. Exiting script.")
    return False

def check_mt5_connection():
    """
    Check if MT5 is connected and reconnect if needed.
    Returns True if connected or reconnected successfully.
    """
    try:
        if mt5.terminal_info() is None:
            logging.warning("MT5 connection lost. Attempting to reconnect...")
            if connect_to_mt5() and login_to_mt5():
                logging.info("Successfully reconnected to MT5")
                return True
            else:
                logging.error("Failed to reconnect to MT5")
                return False
        return True
    except Exception as e:
        logging.error(f"Error checking MT5 connection: {e}")
        return False

def get_account_info():
    """
    Get account information from MT5.
    Returns account info or None if failed.
    """
    if not check_mt5_connection():
        return None
    
    for attempt in range(MAX_RETRIES):
        try:
            account_info = mt5.account_info()
            if account_info:
                return account_info
            
            logging.error(f"Failed to get account info. Error: {mt5.last_error()}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            logging.error(f"Exception getting account info: {e}")
            time.sleep(RETRY_DELAY)
    
    return None

def get_symbol_info(symbol):
    """
    Get symbol information from MT5 with improved retry logic.
    Returns symbol info or None if failed.
    """
    if not check_mt5_connection():
        return None
    
    for attempt in range(MAX_RETRIES):
        try:
            # First try standard symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                return symbol_info
            
            # If not found, try to enable symbol and retry
            if attempt == 0:
                logging.info(f"Symbol {symbol} not found, attempting to enable it")
                if mt5.symbol_select(symbol, True):
                    symbol_info = mt5.symbol_info(symbol)
                    if symbol_info:
                        return symbol_info
            
            logging.error(f"Failed to get symbol info for {symbol}. Error: {mt5.last_error()}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            logging.error(f"Exception getting symbol info for {symbol}: {e}")
            time.sleep(RETRY_DELAY)
    
    return None

def get_candle_data(symbol, timeframe, num_bars=500):
    """
    Fetch historical candle data for the given symbol and timeframe.
    Returns DataFrame with OHLCV data or None if failed.
    """
    if not check_mt5_connection():
        return None
    
    for attempt in range(MAX_RETRIES):
        try:
            # Check if symbol is valid
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logging.error(f"Invalid symbol: {symbol}. Cannot fetch candle data.")
                return None
            
            # Fetch historical data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
            if rates is None or len(rates) == 0:
                logging.error(f"Failed to fetch candle data for {symbol} on timeframe {timeframe}. Error: {mt5.last_error()}")
                time.sleep(RETRY_DELAY)
                continue
            
            # Convert rates to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Ensure 'volume' column exists
            if 'volume' not in df.columns or df['volume'].isnull().all() or (df['volume'] == 0).all():
                df['volume'] = 1.0  # Default volume
                # Reduce logging noise - only log this once per symbol
                if symbol not in state.volume_warning_logged:
                    logging.warning(f"Volume data not available for {symbol}. Using default value.")
                    state.volume_warning_logged.add(symbol)
            
            return df
        
        except Exception as e:
            logging.error(f"Exception fetching candle data for {symbol} on timeframe {timeframe}: {e}")
            time.sleep(RETRY_DELAY)
    
    logging.error(f"Failed to fetch candle data for {symbol} after multiple attempts.")
    return None

def get_positions(symbol=None):
    """
    Get open positions from MT5.
    If symbol is provided, returns positions for that symbol only.
    """
    if not check_mt5_connection():
        return []
    
    for attempt in range(MAX_RETRIES):
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if positions is None:
                logging.error(f"Failed to get positions. Error: {mt5.last_error()}")
                time.sleep(RETRY_DELAY)
                continue
            
            return positions
        except Exception as e:
            logging.error(f"Exception getting positions: {e}")
            time.sleep(RETRY_DELAY)
    
    return []

def get_orders(symbol=None):
    """
    Get pending orders from MT5.
    If symbol is provided, returns orders for that symbol only.
    """
    if not check_mt5_connection():
        return []
    
    for attempt in range(MAX_RETRIES):
        try:
            if symbol:
                orders = mt5.orders_get(symbol=symbol)
            else:
                orders = mt5.orders_get()
            
            if orders is None:
                logging.error(f"Failed to get orders. Error: {mt5.last_error()}")
                time.sleep(RETRY_DELAY)
                continue
            
            return orders
        except Exception as e:
            logging.error(f"Exception getting orders: {e}")
            time.sleep(RETRY_DELAY)
    
    return []

# ======================================================
# TECHNICAL INDICATOR FUNCTIONS
# ======================================================

def calculate_atr(df, period=None):
    """
    Calculate the Average True Range (ATR) for the given period.
    Returns the latest ATR value as a float.
    """
    if period is None:
        period = ATR_PERIOD
    
    try:
        df = df.copy()
        # Handle NaN values
        if df.isnull().values.any():
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate true range
        df['high-low'] = df['high'] - df['low']
        df['high-prev_close'] = abs(df['high'] - df['close'].shift(1))
        df['low-prev_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['tr'].rolling(window=period).mean()
        
        # Return the latest ATR value
        atr = df['atr'].iloc[-1]
        
        # Check if ATR is valid
        if np.isnan(atr) or atr <= 0:
            logging.warning("Invalid ATR value. Using alternative calculation.")
            # Fallback calculation
            atr = (df['high'] - df['low']).rolling(window=period).mean().iloc[-1]
        
        return atr
    except Exception as e:
        logging.error(f"Error calculating ATR: {e}")
        # Last resort fallback
        return (df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.5

def calculate_atr_percent(df, period=None, symbol=None):
    """
    Calculate ATR as a percentage of price.
    More consistent across different price ranges.
    """
    if period is None:
        period = ATR_PERIOD
    
    try:
        atr = calculate_atr(df, period)
        current_price = df['close'].iloc[-1]
        atr_percent = (atr / current_price) * 100
        return atr_percent
    except Exception as e:
        logging.error(f"Error calculating ATR percent: {e}")
        return 0.2  # Default fallback

def calculate_rsi(df, period=None):
    """
    Calculate Relative Strength Index (RSI).
    Returns the latest RSI value as a float.
    """
    if period is None:
        period = RSI_PERIOD
    
    try:
        # Handle NaN values
        close_prices = df['close'].ffill()
        
        # Calculate RSI using talib
        rsi = talib.RSI(close_prices, timeperiod=period)
        
        # Get latest RSI value
        latest_rsi = rsi.iloc[-1]
        
        # Check if RSI is valid
        if np.isnan(latest_rsi):
            logging.warning("Invalid RSI value. Using alternative calculation.")
            # Manual calculation as fallback
            delta = close_prices.diff()
            gain = delta.mask(delta < 0, 0).fillna(0)
            loss = -delta.mask(delta > 0, 0).fillna(0)
            avg_gain = gain.rolling(window=period).mean().iloc[-1]
            avg_loss = loss.rolling(window=period).mean().iloc[-1]
            
            if avg_loss == 0:
                latest_rsi = 100
            else:
                rs = avg_gain / avg_loss
                latest_rsi = 100 - (100 / (1 + rs))
        
        return latest_rsi
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}")
        return 50  # Neutral RSI as fallback

def calculate_bollinger_bands(df, period=None, std_dev=None):
    """
    Calculate Bollinger Bands.
    Returns upper band, middle band, lower band, and bandwidth as floats.
    """
    if period is None:
        period = BB_PERIOD
    if std_dev is None:
        std_dev = BB_STD_DEV
    
    try:
        # Handle NaN values
        close_prices = df['close'].ffill()
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            close_prices, 
            timeperiod=period, 
            nbdevup=std_dev, 
            nbdevdn=std_dev, 
            matype=0  # Simple Moving Average
        )
        
        # Calculate bandwidth
        bandwidth = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
        
        # Return latest values
        return upper.iloc[-1], middle.iloc[-1], lower.iloc[-1], bandwidth
    except Exception as e:
        logging.error(f"Error calculating Bollinger Bands: {e}")
        # Fallback calculation
        sma = df['close'].rolling(window=period).mean().iloc[-1]
        std = df['close'].rolling(window=period).std().iloc[-1]
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bandwidth = (upper_band - lower_band) / sma
        return upper_band, sma, lower_band, bandwidth

def calculate_ema(df, period):
    """
    Calculate Exponential Moving Average (EMA).
    Returns the latest EMA value as a float.
    """
    try:
        # Handle NaN values
        close_prices = df['close'].ffill()
        
        # Calculate EMA using talib
        ema = talib.EMA(close_prices, timeperiod=period)
        
        # Return latest value
        return ema.iloc[-1]
    except Exception as e:
        logging.error(f"Error calculating EMA: {e}")
        # Fallback to SMA
        return df['close'].rolling(window=period).mean().iloc[-1]

def calculate_macd(df, fast_period=None, slow_period=None, signal_period=None):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    Returns MACD line, signal line, and histogram as floats.
    """
    if fast_period is None:
        fast_period = MACD_FAST_PERIOD
    if slow_period is None:
        slow_period = MACD_SLOW_PERIOD
    if signal_period is None:
        signal_period = MACD_SIGNAL_PERIOD
    
    try:
        # Handle NaN values
        close_prices = df['close'].ffill()
        
        # Calculate MACD using talib
        macd_line, signal_line, histogram = talib.MACD(
            close_prices,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )
        
        # Return latest values
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    except Exception as e:
        logging.error(f"Error calculating MACD: {e}")
        # Fallback calculation
        ema_fast = talib.EMA(close_prices, timeperiod=fast_period)
        ema_slow = talib.EMA(close_prices, timeperiod=slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = talib.EMA(macd_line, timeperiod=signal_period)
        histogram = macd_line - signal_line
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

def calculate_adx(df, period=None):
    """
    Calculate Average Directional Index (ADX).
    Returns ADX, +DI, and -DI as floats.
    """
    if period is None:
        period = ADX_PERIOD
    
    try:
        # Calculate ADX using talib
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=period)
        plus_di = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=period)
        minus_di = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=period)
        
        # Return latest values
        return adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]
    except Exception as e:
        logging.error(f"Error calculating ADX: {e}")
        # Return default values
        return 20.0, 20.0, 20.0

def calculate_stochastic(df, k_period=14, d_period=3, slowing=3):
    """
    Calculate Stochastic Oscillator.
    Returns %K and %D as floats.
    """
    try:
        # Calculate Stochastic using talib
        k, d = talib.STOCH(
            df['high'], 
            df['low'], 
            df['close'],
            fastk_period=k_period,
            slowk_period=slowing,
            slowk_matype=0,
            slowd_period=d_period,
            slowd_matype=0
        )
        
        # Return latest values
        return k.iloc[-1], d.iloc[-1]
    except Exception as e:
        logging.error(f"Error calculating Stochastic: {e}")
        # Return default values
        return 50.0, 50.0

def calculate_ichimoku(df):
    """
    Calculate Ichimoku Cloud components.
    Returns Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, and Chikou Span.
    """
    try:
        high_prices = df['high']
        low_prices = df['low']
        close_prices = df['close']
        
        # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = high_prices.rolling(window=9).max()
        period9_low = low_prices.rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = high_prices.rolling(window=26).max()
        period26_low = low_prices.rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        # Calculate Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = high_prices.rolling(window=52).max()
        period52_low = low_prices.rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
        
        # Calculate Chikou Span (Lagging Span): Close price shifted backwards 26 periods
        chikou_span = close_prices.shift(-26)
        
        # Return latest available values
        return (
            tenkan_sen.iloc[-1], 
            kijun_sen.iloc[-1], 
            senkou_span_a.iloc[-1] if not np.isnan(senkou_span_a.iloc[-1]) else (tenkan_sen.iloc[-1] + kijun_sen.iloc[-1]) / 2,
            senkou_span_b.iloc[-1] if not np.isnan(senkou_span_b.iloc[-1]) else (period52_high.iloc[-1] + period52_low.iloc[-1]) / 2,
            chikou_span.iloc[-1] if len(chikou_span) > 26 and not np.isnan(chikou_span.iloc[-1]) else close_prices.iloc[-1]
        )
    except Exception as e:
        logging.error(f"Error calculating Ichimoku: {e}")
        # Return default values
        current_price = df['close'].iloc[-1]
        return current_price, current_price, current_price, current_price, current_price

def calculate_vwma(df, period=None):
    if period is None:
        period = VWMA_PERIOD
    
    try:
        # Check if volume data is available and valid
        if 'volume' not in df.columns or df['volume'].isnull().all() or (df['volume'] == 1.0).all():
            logging.warning("Valid volume data not available. Falling back to SMA.")
            return df['close'].rolling(window=period).mean().iloc[-1]
        
        # Calculate VWMA
        vwma = (df['close'] * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        
        # Return latest value
        return vwma.iloc[-1]
    except Exception as e:
        logging.error(f"Error calculating VWMA: {e}")
        # Fallback to SMA
        return df['close'].rolling(window=period).mean().iloc[-1]

def calculate_pivot_points(df, method='standard'):
    """
    Calculate pivot points.
    Methods: 'standard', 'fibonacci', 'camarilla', 'woodie'
    Returns pivot point and support/resistance levels.
    """
    try:
        # Get high, low, close from previous day/period
        prev_high = df['high'].iloc[-2]
        prev_low = df['low'].iloc[-2]
        prev_close = df['close'].iloc[-2]
        
        # Standard pivot points
        if method == 'standard':
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = (2 * pivot) - prev_low
            s1 = (2 * pivot) - prev_high
            r2 = pivot + (prev_high - prev_low)
            s2 = pivot - (prev_high - prev_low)
            r3 = r1 + (prev_high - prev_low)
            s3 = s1 - (prev_high - prev_low)
        
        # Fibonacci pivot points
        elif method == 'fibonacci':
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = pivot + 0.382 * (prev_high - prev_low)
            s1 = pivot - 0.382 * (prev_high - prev_low)
            r2 = pivot + 0.618 * (prev_high - prev_low)
            s2 = pivot - 0.618 * (prev_high - prev_low)
            r3 = pivot + 1.0 * (prev_high - prev_low)
            s3 = pivot - 1.0 * (prev_high - prev_low)
        
        # Camarilla pivot points
        elif method == 'camarilla':
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
            s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
            r2 = prev_close + 1.1 * (prev_high - prev_low) / 6
            s2 = prev_close - 1.1 * (prev_high - prev_low) / 6
            r3 = prev_close + 1.1 * (prev_high - prev_low) / 4
            s3 = prev_close - 1.1 * (prev_high - prev_low) / 4
        
        # Woodie pivot points
        elif method == 'woodie':
            pivot = (prev_high + prev_low + 2 * prev_close) / 4
            r1 = (2 * pivot) - prev_low
            s1 = (2 * pivot) - prev_high
            r2 = pivot + (prev_high - prev_low)
            s2 = pivot - (prev_high - prev_low)
            r3 = r1 + (prev_high - prev_low)
            s3 = s1 - (prev_high - prev_low)
        
        else:
            # Default to standard
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = (2 * pivot) - prev_low
            s1 = (2 * pivot) - prev_high
            r2 = pivot + (prev_high - prev_low)
            s2 = pivot - (prev_high - prev_low)
            r3 = r1 + (prev_high - prev_low)
            s3 = s1 - (prev_high - prev_low)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    except Exception as e:
        logging.error(f"Error calculating pivot points: {e}")
        current_price = df['close'].iloc[-1]
        # Return default values
        return {
            'pivot': current_price,
            'r1': current_price * 1.01, 'r2': current_price * 1.02, 'r3': current_price * 1.03,
            's1': current_price * 0.99, 's2': current_price * 0.98, 's3': current_price * 0.97
        }

def detect_candlestick_patterns(df):
    """
    Detect common candlestick patterns.
    Returns a dictionary of detected patterns.
    """
    try:
        # Ensure the DataFrame has required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logging.error("Missing required OHLC columns for candlestick pattern detection.")
            return {}
        
        patterns = {}
        
        # Bullish patterns
        patterns['bullish_engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']).iloc[-1] > 0
        patterns['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close']).iloc[-1] > 0
        patterns['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close']).iloc[-1] > 0
        patterns['piercing_line'] = talib.CDLPIERCING(df['open'], df['high'], df['low'], df['close']).iloc[-1] > 0
        patterns['bullish_harami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close']).iloc[-1] > 0
        
        # Bearish patterns
        patterns['bearish_engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']).iloc[-1] < 0
        patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close']).iloc[-1] < 0
        patterns['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close']).iloc[-1] < 0
        patterns['dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close']).iloc[-1] < 0
        patterns['bearish_harami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close']).iloc[-1] < 0
        
        # Neutral patterns
        patterns['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close']).iloc[-1] != 0
        
        # Calculate pattern strength
        bullish_count = sum(1 for k, v in patterns.items() if 'bullish' in k and v)
        bearish_count = sum(1 for k, v in patterns.items() if 'bearish' in k and v)
        neutral_count = sum(1 for k, v in patterns.items() if 'bullish' not in k and 'bearish' not in k and v)
        
        patterns['bullish_count'] = bullish_count
        patterns['bearish_count'] = bearish_count
        patterns['neutral_count'] = neutral_count
        
        if bullish_count > bearish_count:
            patterns['overall_sentiment'] = 'bullish'
        elif bearish_count > bullish_count:
            patterns['overall_sentiment'] = 'bearish'
        else:
            patterns['overall_sentiment'] = 'neutral'
        
        return patterns
    except Exception as e:
        logging.error(f"Error detecting candlestick patterns: {e}")
        return {
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'overall_sentiment': 'neutral'
        }

def detect_price_action(df):
    """
    Detect price action patterns and characteristics.
    Returns a dictionary of price action insights.
    """
    try:
        price_action = {}
        
        # Calculate basic stats
        last_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        # Trend direction based on recent closes
        price_action['short_term_trend'] = 'up' if last_close > prev_close else 'down'
        
        # Recent price movement strength
        price_action['price_change_percent'] = ((last_close - prev_close) / prev_close) * 100
        
        # Volatility
        atr = calculate_atr(df)
        atr_percent = (atr / last_close) * 100
        price_action['volatility'] = atr_percent
        
        if atr_percent < LOW_VOLATILITY_THRESHOLD:
            price_action['volatility_state'] = 'low'
        elif atr_percent < NORMAL_VOLATILITY_THRESHOLD:
            price_action['volatility_state'] = 'normal'
        elif atr_percent < HIGH_VOLATILITY_THRESHOLD:
            price_action['volatility_state'] = 'high'
        else:
            price_action['volatility_state'] = 'extreme'
        
        # Trend persistence
        recent_closes = df['close'].tail(10)
        up_days = sum(1 for i in range(1, len(recent_closes)) if recent_closes.iloc[i] > recent_closes.iloc[i-1])
        price_action['trend_persistence'] = up_days / (len(recent_closes) - 1)
        
        # Range analysis
        recent_highs = df['high'].tail(20)
        recent_lows = df['low'].tail(20)
        price_action['near_high'] = (recent_highs.max() - last_close) / recent_highs.max() < 0.02
        price_action['near_low'] = (last_close - recent_lows.min()) / last_close < 0.02
        
        # Candle characteristics
        last_candle = df.iloc[-1]
        candle_range = last_candle['high'] - last_candle['low']
        body_size = abs(last_candle['close'] - last_candle['open'])
        price_action['candle_size'] = 'large' if body_size > atr else ('small' if body_size < atr/2 else 'medium')
        
        # Upper and lower wicks
        if last_candle['close'] >= last_candle['open']:  # Bullish candle
            upper_wick = last_candle['high'] - last_candle['close']
            lower_wick = last_candle['open'] - last_candle['low']
        else:  # Bearish candle
            upper_wick = last_candle['high'] - last_candle['open']
            lower_wick = last_candle['close'] - last_candle['low']
        
        price_action['upper_wick_ratio'] = upper_wick / candle_range if candle_range > 0 else 0
        price_action['lower_wick_ratio'] = lower_wick / candle_range if candle_range > 0 else 0
        
        # Breakout detection
        n_periods = 20
        highest_high = df['high'].rolling(window=n_periods).max().iloc[-2]  # Highest high excluding current candle
        lowest_low = df['low'].rolling(window=n_periods).min().iloc[-2]  # Lowest low excluding current candle
        
        price_action['breakout_up'] = last_candle['close'] > highest_high
        price_action['breakout_down'] = last_candle['close'] < lowest_low
        
        return price_action
    except Exception as e:
        logging.error(f"Error detecting price action: {e}")
        return {
            'short_term_trend': 'neutral',
            'volatility_state': 'normal',
            'breakout_up': False,
            'breakout_down': False
        }

def detect_support_resistance(df, n_levels=3):
    """
    Detect support and resistance levels using price clustering.
    Returns a dictionary with support and resistance levels.
    """
    try:
        # Extract price data
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Create bins for price clustering
        all_prices = np.concatenate([highs, lows, closes])
        min_price, max_price = all_prices.min(), all_prices.max()
        price_range = max_price - min_price
        
        # Use more bins for larger price ranges
        n_bins = min(100, max(20, int(price_range / (price_range * 0.005))))
        
        # Create histogram of prices
        hist, bin_edges = np.histogram(all_prices, bins=n_bins)
        
        # Find peaks (areas of price clustering)
        peak_indices = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peak_indices.append(i)
        
        # Sort peaks by histogram value (frequency)
        sorted_peaks = sorted([(hist[i], bin_edges[i]) for i in peak_indices], reverse=True)
        
        # Get the current price
        current_price = closes[-1]
        
        # Separate into support and resistance levels
        support_levels = []
        resistance_levels = []
        
        for _, price_level in sorted_peaks:
            if price_level < current_price:
                support_levels.append(price_level)
            else:
                resistance_levels.append(price_level)
        
        # Sort levels by distance from current price
        support_levels.sort(reverse=True)  # Highest support first
        resistance_levels.sort()  # Lowest resistance first
        
        # Limit to n_levels
        support_levels = support_levels[:n_levels]
        resistance_levels = resistance_levels[:n_levels]
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }
    except Exception as e:
        logging.error(f"Error detecting support/resistance: {e}")
        current_price = df['close'].iloc[-1]
        
        # Return default levels
        return {
            'support_levels': [current_price * 0.99, current_price * 0.98, current_price * 0.97],
            'resistance_levels': [current_price * 1.01, current_price * 1.02, current_price * 1.03]
        }

# ======================================================
# ANALYSIS FUNCTIONS
# ======================================================

def initialize_market_constants():
    """
    Initialize market constants from configuration to avoid hard-coding.
    Adds these to CONFIG object for centralized management.
    """
    try:
        # Get existing config
        config = configparser.ConfigParser()
        if os.path.exists('config.ini'):
            config.read('config.ini')
            
        # Make sure VolatilityThresholds section exists
        if 'VolatilityThresholds' not in config:
            config.add_section('VolatilityThresholds')
            
        # Default volatility thresholds if not in config
        if not config.has_option('VolatilityThresholds', 'low_volatility_threshold'):
            config.set('VolatilityThresholds', 'low_volatility_threshold', "0.0020")
        if not config.has_option('VolatilityThresholds', 'normal_volatility_threshold'):
            config.set('VolatilityThresholds', 'normal_volatility_threshold', "0.0050")
        if not config.has_option('VolatilityThresholds', 'high_volatility_threshold'):
            config.set('VolatilityThresholds', 'high_volatility_threshold', "0.0100")
        if not config.has_option('VolatilityThresholds', 'extreme_volatility_threshold'):
            config.set('VolatilityThresholds', 'extreme_volatility_threshold', "0.0200")
        if not config.has_option('VolatilityThresholds', 'super_extreme_volatility_threshold'):
            config.set('VolatilityThresholds', 'super_extreme_volatility_threshold', "0.0300")
            
        # Add a MarketRegimes section
        if 'MarketRegimes' not in config:
            config.add_section('MarketRegimes')
            
        # Add market regime thresholds
        if not config.has_option('MarketRegimes', 'trend_threshold'):
            config.set('MarketRegimes', 'trend_threshold', "0.6")
        if not config.has_option('MarketRegimes', 'range_threshold'):
            config.set('MarketRegimes', 'range_threshold', "0.6")
        if not config.has_option('MarketRegimes', 'choppiness_threshold'):
            config.set('MarketRegimes', 'choppiness_threshold', "50")
            
        # Add risk multipliers
        if 'RiskMultipliers' not in config:
            config.add_section('RiskMultipliers')
            
        if not config.has_option('RiskMultipliers', 'strong_trend_multiplier'):
            config.set('RiskMultipliers', 'strong_trend_multiplier', "1.5")
        if not config.has_option('RiskMultipliers', 'range_bound_multiplier'):
            config.set('RiskMultipliers', 'range_bound_multiplier', "0.7")
        if not config.has_option('RiskMultipliers', 'choppy_volatile_multiplier'):
            config.set('RiskMultipliers', 'choppy_volatile_multiplier', "0.9")
            
        # Write the updated config
        with open('config.ini', 'w') as f:
            config.write(f)
            
        # Now reload the configuration with the new values
        load_config()
        
        # Create a centralized CONSTANTS dictionary
        if not hasattr(state, 'CONSTANTS'):
            state.CONSTANTS = {
                'VOLATILITY': {
                    'LOW': LOW_VOLATILITY_THRESHOLD,
                    'NORMAL': NORMAL_VOLATILITY_THRESHOLD,
                    'HIGH': HIGH_VOLATILITY_THRESHOLD,
                    'EXTREME': EXTREME_VOLATILITY_THRESHOLD,
                    'SUPER_EXTREME': config.getfloat('VolatilityThresholds', 'super_extreme_volatility_threshold')
                },
                'MARKET_REGIME': {
                    'TREND_THRESHOLD': config.getfloat('MarketRegimes', 'trend_threshold'),
                    'RANGE_THRESHOLD': config.getfloat('MarketRegimes', 'range_threshold'),
                    'CHOPPINESS_THRESHOLD': config.getfloat('MarketRegimes', 'choppiness_threshold')
                },
                'RISK_MULTIPLIERS': {
                    'STRONG_TREND': config.getfloat('RiskMultipliers', 'strong_trend_multiplier'),
                    'RANGE_BOUND': config.getfloat('RiskMultipliers', 'range_bound_multiplier'),
                    'CHOPPY_VOLATILE': config.getfloat('RiskMultipliers', 'choppy_volatile_multiplier')
                }
            }
            
        logging.info("Market constants initialized from configuration")
        
    except Exception as e:
        logging.error(f"Error initializing market constants: {e}")

def calculate_market_volatility(symbol, timeframe):
    """
    Calculate market volatility using ATR percentage with configurable thresholds.
    ENHANCED: Better handling for super-extreme volatility, especially for Bitcoin.
    """
    try:
        # Add fallback constants if initialization failed
        if not hasattr(state, 'CONSTANTS'):
            # Create default CONSTANTS as fallback
            state.CONSTANTS = {
                'VOLATILITY': {
                    'LOW': 0.0020,
                    'NORMAL': 0.0050,
                    'HIGH': 0.0100,
                    'EXTREME': 0.0200,
                    'SUPER_EXTREME': 0.0300
                },
                'MARKET_REGIME': {
                    'TREND_THRESHOLD': 0.6,
                    'RANGE_THRESHOLD': 0.6,
                    'CHOPPINESS_THRESHOLD': 50
                },
                'RISK_MULTIPLIERS': {
                    'STRONG_TREND': 1.5,
                    'RANGE_BOUND': 0.7,
                    'CHOPPY_VOLATILE': 0.9
                }
            }
            logging.warning("Created fallback CONSTANTS - proper initialization may have failed")
            
        # if timeframes is None:
        #     timeframes = [MT5_TIMEFRAMES["TREND"], MT5_TIMEFRAMES["PRIMARY"]]
        
        # Get data
        df = get_candle_data(symbol, timeframe)
        if df is None:
            return "normal", 0.0
        
        # Calculate ATR
        atr = calculate_atr(df)
        current_price = df['close'].iloc[-1]
        
        # Calculate ATR as percentage of price
        atr_percent = 0.0
        if current_price > 0:  # Prevent division by zero
            atr_percent = (atr / current_price) * 100
        
        # Get constants from centralized management
        volatility_thresholds = state.CONSTANTS['VOLATILITY']
        
        # ENHANCED: Special handling for cryptocurrencies
        is_crypto = symbol.startswith("BTC") or symbol.startswith("ETH") or "BTC" in symbol or "ETH" in symbol
        
        if is_crypto:
            # ENHANCED: Improved crypto-specific adjustment factor
            crypto_adjustment = 1.9  # Increased from 1.75 for better crypto volatility handling
            
            # ENHANCED: Specific handling for Bitcoin which tends to have higher volatility
            is_bitcoin = symbol.startswith("BTC") or "BTC" in symbol
            if is_bitcoin:
                # ENHANCED: More refined Bitcoin-specific thresholds
                if atr_percent > (volatility_thresholds['EXTREME'] * crypto_adjustment * 1.3):
                    # Log detailed super-extreme conditions for tracking
                    logging.info(f"SUPER-EXTREME volatility detected for {symbol}: {atr_percent:.4f}% " +
                               f"(Threshold: {volatility_thresholds['EXTREME'] * crypto_adjustment * 1.3:.4f}%)")
                    return "super-extreme", atr_percent
                
            # Determine volatility level using adjusted thresholds
            if atr_percent < (volatility_thresholds['LOW'] * crypto_adjustment):
                return "low", atr_percent
            elif atr_percent < (volatility_thresholds['NORMAL'] * crypto_adjustment):
                return "normal", atr_percent
            elif atr_percent < (volatility_thresholds['HIGH'] * crypto_adjustment):
                return "high", atr_percent
            elif atr_percent < (volatility_thresholds['SUPER_EXTREME'] * crypto_adjustment):
                return "extreme", atr_percent
            else:
                return "super-extreme", atr_percent
                
        # Standard volatility calculation for non-crypto
        if atr_percent < volatility_thresholds['LOW']:
            return "low", atr_percent
        elif atr_percent < volatility_thresholds['NORMAL']:
            return "normal", atr_percent
        elif atr_percent < volatility_thresholds['HIGH']:
            return "high", atr_percent
        elif atr_percent < volatility_thresholds['SUPER_EXTREME']:
            return "extreme", atr_percent
        else:
            return "super-extreme", atr_percent
            
    except Exception as e:
        logging.error(f"Error calculating market volatility: {e}")
        return "normal", 0.0
    
def analyze_market_structure(symbol):
    """
    Analyze market structure to determine if market is trending, ranging, or in transition.
    Uses multiple timeframes and indicators.
    """
    try:
        market_state = {
            "structure": "unknown",
            "trend_strength": 0,
            "volatility": "normal",
            "momentum": "neutral",
            "support_resistance": {},
            "key_levels": []
        }
        
        # Get data for primary and trend timeframes
        optimized_tf = optimize_timeframe_for_volatility(symbol)
        primary_df = get_candle_data(symbol, optimized_tf)
        trend_df = get_candle_data(symbol, MT5_TIMEFRAMES["TREND"])
        
        if primary_df is None or trend_df is None:
            return market_state
        
        # Calculate indicators
        adx, plus_di, minus_di = calculate_adx(trend_df)
        bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(primary_df)
        volatility_level, atr_percent = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
        rsi = calculate_rsi(primary_df)
        macd_line, signal_line, histogram = calculate_macd(primary_df)
        
        # Get additional data
        supp_res = detect_support_resistance(trend_df)
        price_action = detect_price_action(primary_df)
        patterns = detect_candlestick_patterns(primary_df)
        
        # Trend determination
        ema_short = calculate_ema(trend_df, EMA_SHORT_PERIOD)
        ema_long = calculate_ema(trend_df, EMA_LONG_PERIOD)
        trend_direction = "up" if ema_short > ema_long else "down" if ema_short < ema_long else "neutral"
        
        # Check for ranging market
        ranging_indicators = [
            adx < ADX_THRESHOLD,                         # Low ADX indicates ranging
            bb_width < 0.03,                             # Narrow Bollinger Bands
            abs(rsi - 50) < 15,                          # RSI near middle
            abs(macd_line) < 0.2 * primary_df['close'].mean() * 0.01,  # MACD near zero
            abs(plus_di - minus_di) < 10                 # DI lines close together
        ]
        
        # Check for trending market
        trending_indicators = [
            adx > ADX_THRESHOLD,                         # High ADX indicates trend
            abs(rsi - 50) > 15,                          # RSI away from middle
            abs(macd_line) > 0.2 * primary_df['close'].mean() * 0.01,  # MACD away from zero
            bb_width > 0.03,                             # Wider Bollinger Bands
            abs(plus_di - minus_di) > 10                 # DI lines separated
        ]
        
        # Calculate scores
        ranging_score = sum(ranging_indicators) / len(ranging_indicators)
        trending_score = sum(trending_indicators) / len(trending_indicators)
        
        # Determine market structure
        if trending_score > 0.7:
            market_state["structure"] = "trending"
            market_state["trend_direction"] = trend_direction
            market_state["trend_strength"] = adx / 100  # Normalize to 0-1
        elif ranging_score > 0.7:
            market_state["structure"] = "ranging"
            market_state["range_width"] = bb_width
        else:
            market_state["structure"] = "transitioning"
        
        # Add other market state information
        market_state["volatility"] = volatility_level
        market_state["atr_percent"] = atr_percent
        market_state["momentum"] = "bullish" if histogram > 0 and rsi > 50 else "bearish" if histogram < 0 and rsi < 50 else "neutral"
        market_state["support_resistance"] = supp_res
        market_state["price_action"] = price_action
        market_state["patterns"] = patterns
        market_state["adx"] = adx
        market_state["rsi"] = rsi
        
        # Identify key levels (combine pivot points with support/resistance)
        pivot_data = calculate_pivot_points(trend_df)
        key_levels = []
        
        # Add pivot levels
        for level_name, level_value in pivot_data.items():
            key_levels.append({
                "price": level_value,
                "type": level_name,
                "strength": 0.8 if level_name == "pivot" else 0.6
            })
        
        # Add support/resistance levels
        for level in supp_res.get("support_levels", []):
            key_levels.append({
                "price": level,
                "type": "support",
                "strength": 0.7
            })
        
        for level in supp_res.get("resistance_levels", []):
            key_levels.append({
                "price": level,
                "type": "resistance",
                "strength": 0.7
            })
        
        # Sort key levels by price
        key_levels.sort(key=lambda x: x["price"])
        market_state["key_levels"] = key_levels
        
        return market_state
    except Exception as e:
        logging.error(f"Error analyzing market structure: {e}")
        return {
            "structure": "unknown",
            "trend_strength": 0,
            "volatility": "normal",
            "momentum": "neutral",
            "support_resistance": {},
            "key_levels": []
        }

def analyze_trend(symbol, timeframes=None):
    if timeframes is None:
        timeframes = ["PRIMARY", "TREND"] + [f"SECONDARY_{i}" for i in range(len(SECONDARY_TIMEFRAMES))]
    
    try:
        trend_analysis = {}
        
        for tf in timeframes:
            if tf not in MT5_TIMEFRAMES:
                continue
                
            df = get_candle_data(symbol, MT5_TIMEFRAMES[tf])
            if df is None:
                continue
            
            # Calculate indicators
            ema_short = calculate_ema(df, EMA_SHORT_PERIOD)
            ema_long = calculate_ema(df, EMA_LONG_PERIOD)
            adx, plus_di, minus_di = calculate_adx(df)
            macd_line, signal_line, histogram = calculate_macd(df)
            rsi = calculate_rsi(df)
            
            # Determine trend direction and strength
            ema_trend = "bullish" if ema_short > ema_long else "bearish"
            macd_trend = "bullish" if macd_line > signal_line else "bearish"
            di_trend = "bullish" if plus_di > minus_di else "bearish"
            
            # Trend agreement
            trend_agreement = (
                (ema_trend == "bullish" and macd_trend == "bullish" and di_trend == "bullish") or
                (ema_trend == "bearish" and macd_trend == "bearish" and di_trend == "bearish")
            )
            
            # Trend strength based on indicators
            trend_strength = (adx / 100) * min(1.0, abs(plus_di - minus_di) / 30)
            
            # Overall trend direction
            if ema_trend == macd_trend == di_trend:
                trend_direction = ema_trend
            elif ema_trend == macd_trend:
                trend_direction = ema_trend
            elif ema_trend == di_trend:
                trend_direction = ema_trend
            elif macd_trend == di_trend:
                trend_direction = macd_trend
            else:
                trend_direction = "mixed"
            
            # Calculate momentum - FIX: Use Series operations correctly
            if isinstance(histogram, pd.Series):
                prev_histogram = histogram.shift(1)
                momentum = "increasing" if histogram.iloc[-1] > 0 and histogram.iloc[-1] > prev_histogram.iloc[-1] else \
                          "decreasing" if histogram.iloc[-1] < 0 and histogram.iloc[-1] < prev_histogram.iloc[-1] else \
                          "neutral"
            else:
                # Fallback if histogram is not a Series
                momentum = "neutral"
            
            # Determine if price is in an extreme condition
            extreme_condition = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "none"
            
            # Store analysis for this timeframe
            trend_analysis[tf] = {
                "direction": trend_direction,
                "strength": trend_strength,
                "momentum": momentum,
                "extreme_condition": extreme_condition,
                "agreement": trend_agreement,
                "indicators": {
                    "ema_trend": ema_trend,
                    "macd_trend": macd_trend,
                    "di_trend": di_trend,
                    "adx": adx,
                    "rsi": rsi
                }
            }
            
        # Add a multi-timeframe weighted consensus
        if trend_analysis:
            # Weighted voting system
            weights = {
                "TREND": 0.4,
                "PRIMARY": 0.3,
                "SECONDARY_0": 0.2,
                "SECONDARY_1": 0.1
            }
            
            bullish_score = 0
            total_weight = 0
            
            for tf, analysis in trend_analysis.items():
                if tf in weights:
                    weight = weights[tf]
                    direction_score = 1 if analysis["direction"] == "bullish" else -1 if analysis["direction"] == "bearish" else 0
                    strength_score = direction_score * analysis["strength"]
                    bullish_score += strength_score * weight
                    total_weight += weight
            
            if total_weight > 0:
                consensus_score = bullish_score / total_weight
                consensus = {
                    "direction": "bullish" if consensus_score > 0.3 else "bearish" if consensus_score < -0.3 else "neutral",
                    "strength": abs(consensus_score),
                    "score": consensus_score
                }
                trend_analysis["consensus"] = consensus
        
        return trend_analysis
    except Exception as e:
        logging.error(f"Error analyzing trend: {e}")
        return {}

def check_news_events(symbol, hours=24):
    """
    Check for high-impact news events that could affect the symbol.
    Returns True if high-impact news found within time window.
    """
    if not USE_NEWS_FILTER or not NEWS_API_KEY:
        return False
    
    try:
        # Extract currency codes from symbol (e.g., EURUSD -> EUR, USD)
        currencies = []
        if len(symbol) == 6 and symbol.isalpha():
            currencies = [symbol[:3], symbol[3:]]
        
        # Build query
        query = " OR ".join(currencies) if currencies else symbol
        
        # API request
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            current_time = datetime.now(timezone.utc)
            
            for article in articles:
                try:
                    published_at = datetime.strptime(article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
                    published_at = published_at.replace(tzinfo=timezone.utc)
                    
                    # Check if article is within time window
                    if (current_time - published_at).total_seconds() < hours * 3600:
                        # Check for high-impact keywords
                        high_impact_keywords = [
                            "rate decision", "interest rate", "monetary policy", "central bank",
                            "economic crisis", "market crash", "flash crash", "recession",
                            "unemployment", "inflation", "GDP", "non-farm payroll", "NFP"
                        ]
                        
                        title_lower = article["title"].lower()
                        description = article.get("description", "").lower()
                        
                        if any(keyword in title_lower or keyword in description for keyword in high_impact_keywords):
                            logging.warning(f"High-impact news detected: {article['title']}")
                            return True
                except Exception as e:
                    logging.error(f"Error processing news article: {e}")
            
            return False
        else:
            logging.warning(f"News API request failed: {response.status_code}")
            return False
    
    except Exception as e:
        logging.error(f"Error checking news events: {e}")
        return False

# ======================================================
# TRADE MANAGEMENT FUNCTIONS
# ======================================================

def calculate_lot_size(symbol, risk_percentage, stop_loss_pips):
    """
    Calculate appropriate lot size based on account balance, risk percentage, and stop loss.
    Ensures lot size is within symbol's volume limits.
    """
    try:
        # Get account info
        account_info = get_account_info()
        if account_info is None:
            return 0.01  # Default minimum
        
        balance = account_info.balance
        
        # Get symbol info
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return 0.01  # Default minimum
        
        # Get symbol-specific values
        point = symbol_info.point
        contract_size = symbol_info.trade_contract_size
        price = mt5.symbol_info_tick(symbol).ask
        
        # Currency conversion if needed (for cross pairs)
        currency_profit = symbol_info.currency_profit
        account_currency = account_info.currency
        
        conversion_rate = 1.0
        if currency_profit != account_currency:
            conversion_pair = f"{currency_profit}{account_currency}"
            conversion_info = get_symbol_info(conversion_pair)
            if conversion_info:
                conversion_rate = mt5.symbol_info_tick(conversion_pair).ask
        
        # Calculate risk amount
        risk_amount = balance * (risk_percentage / 100)
        
        # Calculate pip value
        pip_value = (point * contract_size) / price
        
        # Convert pip value to account currency
        pip_value_in_account_currency = pip_value * conversion_rate
        
        # Calculate lot size
        stop_loss_amount = stop_loss_pips * pip_value_in_account_currency
        lot_size = risk_amount / stop_loss_amount if stop_loss_amount > 0 else 0.01
        
        # Ensure lot size is within symbol's limits
        lot_size = max(lot_size, symbol_info.volume_min)
        lot_size = min(lot_size, symbol_info.volume_max)
        
        # Round to nearest step
        lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
        
        # Check if margin is sufficient
        margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, lot_size, price)
        if margin and margin > account_info.margin_free:
            # Reduce lot size to fit available margin
            max_affordable_lot = (account_info.margin_free / margin) * lot_size
            max_affordable_lot = max(max_affordable_lot, symbol_info.volume_min)
            max_affordable_lot = min(max_affordable_lot, symbol_info.volume_max)
            max_affordable_lot = round(max_affordable_lot / symbol_info.volume_step) * symbol_info.volume_step
            lot_size = max_affordable_lot
        
        logging.info(f"Calculated lot size for {symbol}: {lot_size} (Risk: {risk_percentage}%, Stop loss: {stop_loss_pips} pips)")
        return lot_size
    
    except Exception as e:
        logging.error(f"Error calculating lot size: {e}")
        return 0.01  # Default minimum

def calculate_stop_loss_take_profit(symbol, order_type, entry_price=None, atr_value=None, key_levels=None):
    """
    Calculate dynamic stop loss and take profit levels with improved volatility adjustment.
    Uses ATR, key levels, and market structure for optimal placement.
    """
    try:
        # Get current price if entry price not provided
        if entry_price is None:
            if order_type == "BUY":
                entry_price = mt5.symbol_info_tick(symbol).ask
            else:  # "SELL"
                entry_price = mt5.symbol_info_tick(symbol).bid
        
        # Get symbol info
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return None, None
        
        point = symbol_info.point
        min_stop_level_points = symbol_info.trade_stops_level

        # NEW: Check market regime for more ambitious targets
        market_regime, regime_strength = detect_market_regime(symbol)
        
        # NEW: Set more ambitious targets in strong trends
        if market_regime in ["STRONG_UPTREND", "STRONG_DOWNTREND"] and regime_strength > 0.7:
            if (market_regime == "STRONG_UPTREND" and order_type == "BUY") or \
               (market_regime == "STRONG_DOWNTREND" and order_type == "SELL"):
                # In aligned strong trends, be much more ambitious with targets
                tp_multiplier *= 1.5  # 50% larger take profit target
                logging.info(f"Setting ambitious profit target (50% larger) due to strong {market_regime}")
                
        # NEW: Expand targets based on chart patterns
        if hasattr(state, 'current_signals') and symbol in state.current_signals:
            signal_data = state.current_signals[symbol]
            if "chart_patterns" in signal_data:
                pattern_data = signal_data["chart_patterns"]
                if pattern_data.get("detected", False):
                    # Use pattern targets for more precise profit objectives
                    pattern_target = pattern_data.get("target")
                    if pattern_target is not None:
                        # Calculate potential profit based on pattern target
                        if order_type == "BUY" and pattern_target > entry_price:
                            pattern_profit = pattern_target - entry_price
                            # Only use pattern target if it's more ambitious
                            if pattern_profit > tp_distance:
                                tp_distance = pattern_profit
                                logging.info(f"Using more ambitious pattern-based target: {pattern_profit:.2f}")
                        elif order_type == "SELL" and pattern_target < entry_price:
                            pattern_profit = entry_price - pattern_target
                            # Only use pattern target if it's more ambitious
                            if pattern_profit > tp_distance:
                                tp_distance = pattern_profit
                                logging.info(f"Using more ambitious pattern-based target: {pattern_profit:.2f}")                                
        
        # Cryptocurrency-specific handling
        if symbol.startswith("BTC") or symbol.startswith("ETH") or "BTC" in symbol or "ETH" in symbol:
            # Use improved percentage-based stops for cryptocurrencies that vary by volatility
            volatility_level, atr_percent = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
            
            is_bitcoin = symbol.startswith("BTC") or "BTC" in symbol
    
            # Base percentages - IMPROVED: Better volatility adjustment
            if is_bitcoin:
                if volatility_level == "super-extreme":
                    sl_percent = 0.02  # 2.5% default stop loss for super-extreme - INCREASED for better protection
                    tp_percent = 0.045  # 4.5% take profit - INCREASED for better reward
                elif volatility_level == "extreme":
                    sl_percent = 0.020  # 2.0% stop loss (was 1.8%) - INCREASED for better protection
                    tp_percent = 0.035  # 3.5% take profit - INCREASED for better reward
                else:
                    sl_percent = 0.016  # 1.6% for lower volatility (was 1.5%) - INCREASED for better protection
                    tp_percent = 0.030  # 3.0% take profit - INCREASED for better reward
            else:
                # Base percentages for other crypto
                sl_percent = 0.013  # 1.3% default stop loss (was 1.2%) - INCREASED for better protection
                tp_percent = 0.023  # 2.3% default take profit (was 2.2%) - INCREASED for better reward
                
                # Adjust percentages based on volatility - NEW: More aggressive adjustment
                if volatility_level == "high":
                    sl_percent = 0.017  # 1.7% during high volatility (was 1.6%) - INCREASED for better protection
                    tp_percent = 0.030  # 3.0% during high volatility - INCREASED for better reward
                elif volatility_level == "extreme":
                    sl_percent = 0.020  # 2.0% during extreme volatility (was 1.8%) - INCREASED for better protection
                    tp_percent = 0.035  # 3.5% during extreme volatility - INCREASED for better reward

            # ENHANCED: Maximum SL distance cap based on ATR to prevent excessive risk
            # Use a multiple of ATR as a cap, but ensure it's not too tight
            if atr_value is not None:
                atr_sl_percent = (atr_value * 6) / entry_price  # Use 6x ATR as maximum SL distance
                # Take the smaller of percentage-based or ATR-based SL
                sl_percent = min(sl_percent, atr_sl_percent)
                
                # Ensure risk-reward ratio remains favorable after adjustments
                min_rr_ratio = 1.5  # Minimum risk-reward ratio
                min_tp_percent = sl_percent * min_rr_ratio
                tp_percent = max(tp_percent, min_tp_percent)  # Ensure TP is at least min_rr_ratio times SL
            
            # Calculate stop loss and take profit based on adjusted percentages
            if order_type == "BUY":
                stop_loss = entry_price * (1 - sl_percent)
                take_profit = entry_price * (1 + tp_percent)
            else:  # "SELL"
                stop_loss = entry_price * (1 + sl_percent)
                take_profit = entry_price * (1 - tp_percent)
            
            # Round to symbol precision
            digits = symbol_info.digits
            stop_loss = round(stop_loss, digits)
            take_profit = round(take_profit, digits)
            
            # Ensure SL and TP are within broker's minimum distance requirements
            min_distance = min_stop_level_points * point
            
            if order_type == "BUY":
                if entry_price - stop_loss < min_distance:
                    stop_loss = entry_price - min_distance
                if take_profit - entry_price < min_distance:
                    take_profit = entry_price + min_distance
            else:  # "SELL"
                if stop_loss - entry_price < min_distance:
                    stop_loss = entry_price + min_distance
                if entry_price - take_profit < min_distance:
                    take_profit = entry_price - min_distance
                    
            return stop_loss, take_profit
        
        # Standard calculation for non-crypto assets
        # Get ATR if not provided
        if atr_value is None:
            optimized_tf = optimize_timeframe_for_volatility(symbol)
            df = get_candle_data(symbol, optimized_tf)
            if df is None:
                return None, None
            atr_value = calculate_atr(df)
        
        # IMPROVED: Better volatility-based multiplier with enhanced scalability
        volatility_level, atr_percent = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
        
        # NEW: More responsive volatility adjustment
        if volatility_level == "low":
            sl_multiplier = 1.6  # INCREASED from 1.5 for better protection
            tp_multiplier = 2.5  # INCREASED from 2.3 for better reward
        elif volatility_level == "normal":
            sl_multiplier = 1.9  # INCREASED from 1.8 for better protection
            tp_multiplier = 2.9  # INCREASED from 2.7 for better reward
        elif volatility_level == "high":
            sl_multiplier = 2.2  # INCREASED from 2.0 for better protection
            tp_multiplier = 3.3  # INCREASED from 3.0 for better reward
        else:  # "extreme"
            sl_multiplier = 2.5  # INCREASED from 2.2 for better protection
            tp_multiplier = 3.8  # INCREASED from 3.3 for better reward
        
        # NEW: Apply scaling based on symbol characteristics
        if "JPY" in symbol:  # JPY pairs typically need wider stops due to higher point value
            sl_multiplier *= 1.2
            tp_multiplier *= 1.2
        
        # Calculate initial SL and TP based on ATR
        sl_distance = atr_value * sl_multiplier
        tp_distance = atr_value * tp_multiplier
        
        # Calculate stop loss and take profit levels
        if order_type == "BUY":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # "SELL"
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        # NEW: Apply emergency protection for stops that are too far
        # Cap maximum SL distance to protect capital
        max_sl_percent = 0.04  # Maximum SL at 4% from entry
        if order_type == "BUY":
            max_allowed_sl = entry_price * (1 - max_sl_percent)
            stop_loss = max(stop_loss, max_allowed_sl)
        else:  # "SELL"
            max_allowed_sl = entry_price * (1 + max_sl_percent)
            stop_loss = min(stop_loss, max_allowed_sl)

        
        # Refine with key levels if available
        if key_levels:
            # For buy orders, find nearest support for SL and nearest resistance for TP
            if order_type == "BUY":
                # Find support levels below entry
                suitable_supports = [level["price"] for level in key_levels 
                                     if level["type"] in ["support", "s1", "s2", "s3"] 
                                     and level["price"] < entry_price]
                
                # Find resistance levels above entry
                suitable_resistances = [level["price"] for level in key_levels 
                                        if level["type"] in ["resistance", "r1", "r2", "r3"] 
                                        and level["price"] > entry_price]
                
                # Use nearest support that's not too close to entry
                if suitable_supports:
                    nearest_support = max(suitable_supports)
                    if entry_price - nearest_support >= sl_distance * 0.8:  # At least 80% of ATR-based distance
                        stop_loss = nearest_support
                
                # Use nearest resistance that's not too close to entry
                if suitable_resistances:
                    nearest_resistance = min(suitable_resistances)
                    if nearest_resistance - entry_price >= tp_distance * 0.8:  # At least 80% of ATR-based distance
                        take_profit = nearest_resistance
            
            # For sell orders, find nearest resistance for SL and nearest support for TP
            else:  # "SELL"
                # Find resistance levels above entry
                suitable_resistances = [level["price"] for level in key_levels 
                                        if level["type"] in ["resistance", "r1", "r2", "r3"] 
                                        and level["price"] > entry_price]
                
                # Find support levels below entry
                suitable_supports = [level["price"] for level in key_levels 
                                     if level["type"] in ["support", "s1", "s2", "s3"] 
                                     and level["price"] < entry_price]
                
                # Use nearest resistance that's not too close to entry
                if suitable_resistances:
                    nearest_resistance = min(suitable_resistances)
                    if nearest_resistance - entry_price >= sl_distance * 0.8:  # At least 80% of ATR-based distance
                        stop_loss = nearest_resistance
                
                # Use nearest support that's not too close to entry
                if suitable_supports:
                    nearest_support = max(suitable_supports)
                    if entry_price - nearest_support >= tp_distance * 0.8:  # At least 80% of ATR-based distance
                        take_profit = nearest_support
        
        # Ensure SL and TP are within broker's minimum distance requirements
        min_distance = min_stop_level_points * point
        
        if order_type == "BUY":
            if entry_price - stop_loss < min_distance:
                stop_loss = entry_price - min_distance
            if take_profit - entry_price < min_distance:
                take_profit = entry_price + min_distance
        else:  # "SELL"
            if stop_loss - entry_price < min_distance:
                stop_loss = entry_price + min_distance
            if entry_price - take_profit < min_distance:
                take_profit = entry_price - min_distance

        # ADDED: Apply maximum SL distance to prevent excessive risk
        max_atr_multiplier = 4.0  # Maximum ATR multiple allowed for SL
        max_sl_distance = atr_value * max_atr_multiplier

        # Apply the cap
        if order_type == "BUY":
            min_allowed_sl = entry_price - max_sl_distance
            stop_loss = max(stop_loss, min_allowed_sl)
        else:  # "SELL"
            max_allowed_sl = entry_price + max_sl_distance
            stop_loss = min(stop_loss, max_allowed_sl)

        # Log the SL adjustment for monitoring
        if order_type == "BUY":
            original_sl_distance = entry_price - stop_loss
        elif order_type == "SELL":
            original_sl_distance = stop_loss - entry_price
        else:
            original_sl_distance = 0

        logging.info(f"SL distance for {symbol} {order_type}: {original_sl_distance:.2f} points, " +
                    f"Max allowed: {max_sl_distance:.2f} points (ATR: {atr_value:.2f})")
        
        return stop_loss, take_profit
    
    except Exception as e:
        logging.error(f"Error calculating SL/TP: {e}")
        return None, None
    
def apply_trailing_stop(symbol, ticket, current_price, position_type, entry_price, atr_value=None):
    """
    Apply intelligent trailing stop based on price action and volatility.
    Returns the new stop loss level.
    """
    try:
        # Get current position
        positions = get_positions(symbol)
        position = next((p for p in positions if p.ticket == ticket), None)
        
        if position is None:
            logging.warning(f"Position {ticket} not found. Cannot apply trailing stop.")
            return None
        
        # Get current stop loss
        current_sl = position.sl
        
        # Get ATR if not provided
        if atr_value is None:
            optimized_tf = optimize_timeframe_for_volatility(symbol)
            df = get_candle_data(symbol, optimized_tf)
            if df is None:
                return current_sl
            atr_value = calculate_atr(df)
        
        # Get profit in points
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return current_sl
        
        point = symbol_info.point
        
        if position_type == mt5.ORDER_TYPE_BUY:
            profit_points = (current_price - entry_price) / point
        else:  # mt5.ORDER_TYPE_SELL
            profit_points = (entry_price - current_price) / point
        
        # Only activate trailing stop after certain profit threshold
        activation_points = atr_value / point * 2  # 2 ATR
        
        if profit_points < activation_points:
            return current_sl
        
        # Calculate profit percentage
        profit_percent = profit_points / (entry_price / point) * 100
        
        # Determine trail factor based on profit zones
        trail_factor = 0.0
        
        # Find the appropriate profit lock level
        for i, level in enumerate(TSL_PROFIT_LOCK_LEVELS):
            if profit_percent >= level:
                trail_factor = TSL_PROFIT_LOCK_PERCENTS[i]
            else:
                break
        
        # Calculate trail distance based on ATR
        trail_distance = atr_value * (1.0 - trail_factor)
        
        # Calculate new stop loss
        if position_type == mt5.ORDER_TYPE_BUY:
            new_sl = current_price - trail_distance
            
            # Only move stop loss up
            if new_sl <= current_sl:
                return current_sl
        else:  # mt5.ORDER_TYPE_SELL
            new_sl = current_price + trail_distance
            
            # Only move stop loss down
            if new_sl >= current_sl:
                return current_sl
        
        # Round to symbol precision
        digits = symbol_info.digits
        new_sl = round(new_sl, digits)
        
        return new_sl
    
    except Exception as e:
        logging.error(f"Error applying trailing stop: {e}")
        return None

def update_position_stops(symbol, ticket, new_sl=None, new_tp=None):
    """
    Update stop loss and/or take profit for an open position.
    Returns True if update was successful.
    """
    try:
        if not check_mt5_connection():
            return False
        
        # Get current position
        positions = get_positions(symbol)
        position = next((p for p in positions if p.ticket == ticket), None)
        
        if position is None:
            logging.warning(f"Position {ticket} not found. Cannot update stops.")
            return False
        
        # Use current values if not provided
        if new_sl is None:
            new_sl = position.sl
        
        if new_tp is None:
            new_tp = position.tp
        
        # Check if update is needed
        if new_sl == position.sl and new_tp == position.tp:
            return True  # No change needed
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "sl": new_sl,
            "tp": new_tp,
            "position": ticket
        }
        
        # Send request
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Updated stops for position {ticket} - New SL: {new_sl}, New TP: {new_tp}")
            return True
        else:
            logging.error(f"Failed to update stops for position {ticket}: {result.comment if result else 'Unknown error'}")
            return False
    
    except Exception as e:
        logging.error(f"Error updating position stops: {e}")
        return False

def place_market_order(symbol, order_type, lot_size, sl, tp, comment=""):
    """
    Place a market order with specified parameters.
    Enhanced version with broker-specific handling and multiple fallback strategies.
    Returns ticket number if successful, None otherwise.
    """
    try:
        if not check_mt5_connection():
            return None
        
        # Check if this is a Bitcoin order
        is_bitcoin = symbol.startswith("BTC") or "BTC" in symbol
        
        # ADDED: Safety check for unreasonable stop loss distances
        if order_type == "BUY" and sl is not None:
            price = mt5.symbol_info_tick(symbol).ask
            sl_distance = price - sl
            
            # Get ATR for reference
            df = get_candle_data(symbol, MT5_TIMEFRAMES["PRIMARY"])
            # Get volatility level
            volatility_level, _ = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
            if df is not None:
                atr = calculate_atr(df)
                # MODIFIED: Increased ATR multiplier for Bitcoin
                if is_bitcoin:
                    # Adjust multiplier based on volatility level
                    if volatility_level == "super-extreme":
                        max_reasonable_sl = atr * 25  # Increased from 18 to 25 for super-extreme volatility
                    else:
                        max_reasonable_sl = atr * 18  # Increased from 5 to 18 for Bitcoin
                elif symbol.startswith("ETH") or "ETH" in symbol:
                    max_reasonable_sl = atr * 12  # Increased for Ethereum
                else:
                    max_reasonable_sl = atr * 5  # Original for other instruments
                # Add detailed logging for SL safety check
                logging.info(f"SL safety check for {symbol}: SL distance={sl_distance:.2f}, " +
                            f"max allowed={max_reasonable_sl:.2f}, ATR={atr:.2f}, " +
                            f"volatility={volatility_level}")
                if sl_distance > max_reasonable_sl:
                    logging.warning(f"Safety check failed: SL distance ({sl_distance:.2f}) exceeds maximum reasonable distance ({max_reasonable_sl:.2f})")
                    return None
        
        elif order_type == "SELL" and sl is not None:
            price = mt5.symbol_info_tick(symbol).bid
            sl_distance = sl - price
            
            # Get ATR for reference
            df = get_candle_data(symbol, MT5_TIMEFRAMES["PRIMARY"])
            if df is not None:
                atr = calculate_atr(df)
                # MODIFIED: Increased ATR multiplier for Bitcoin
                if is_bitcoin:
                    max_reasonable_sl = atr * 18  # Increased from 5 to 18 for Bitcoin
                elif symbol.startswith("ETH") or "ETH" in symbol:
                    max_reasonable_sl = atr * 12  # Increased for Ethereum
                else:
                    max_reasonable_sl = atr * 5  # Original for other instruments
                
                if sl_distance > max_reasonable_sl:
                    logging.warning(f"Safety check failed: SL distance ({sl_distance:.2f}) exceeds maximum reasonable distance ({max_reasonable_sl:.2f})")
                    return None
        
        # IMPROVED: Multiple verification checks for existing positions
        # Check twice with delay to ensure we're not missing any positions still being processed
        existing_positions = get_positions(symbol)
        if len(existing_positions) > 0:
            logging.warning(f"Position already exists for {symbol}. Skipping order placement.")
            return None
            
        # Wait a moment and check again to catch any positions that might be in process
        time.sleep(0.3)  
        existing_positions = get_positions(symbol)
        if len(existing_positions) > 0:
            logging.warning(f"Position detected on second check for {symbol}. Skipping order placement.")
            return None
        
        # Get symbol info
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return None
        
        # Check account info
        account_info = get_account_info()
        if account_info is None:
            return None
            
        # Only check margin level if margin is actually being used
        if account_info.margin > 0:
            if account_info.margin_level is not None and account_info.margin_level < 200:
                logging.warning(f"Margin level too low ({account_info.margin_level:.1f}%). Cannot place order.")
                return None
                
            # Check margin usage percentage
            margin_usage = (account_info.margin / account_info.equity) * 100 if account_info.equity > 0 else 0
            if margin_usage > 70:
                logging.warning(f"Margin usage too high ({margin_usage:.1f}%). Cannot place order.")
                return None
        
        # Verify symbol trading is enabled
        if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
            logging.error(f"Trading is not enabled for {symbol}. Mode: {symbol_info.trade_mode}")
            return None
            
        # Prepare request
        if order_type == "BUY":
            price = mt5.symbol_info_tick(symbol).ask
            mt5_order_type = mt5.ORDER_TYPE_BUY
        else:  # "SELL"
            price = mt5.symbol_info_tick(symbol).bid
            mt5_order_type = mt5.ORDER_TYPE_SELL
        
        # Round to symbol precision
        digits = symbol_info.digits
        price = round(price, digits)
        sl = round(sl, digits) if sl is not None else 0.0
        tp = round(tp, digits) if tp is not None else 0.0
        
        # Verify lot size is within allowed range
        lot_size = max(lot_size, symbol_info.volume_min)
        lot_size = min(lot_size, symbol_info.volume_max)
        lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
        
        # Define symbol filling mode constants
        SYMBOL_FILLING_FOK = 1
        SYMBOL_FILLING_IOC = 2
        
        # Default filling mode
        filling_mode = mt5.ORDER_FILLING_IOC  # Use IOC (Immediate or Cancel) as default
        
        # Check if the filling_mode attribute exists and handle accordingly
        if hasattr(symbol_info, 'filling_mode'):
            # Check symbol filling mode safely without using potentially missing MT5 constants
            if symbol_info.filling_mode & SYMBOL_FILLING_FOK:
                filling_mode = mt5.ORDER_FILLING_FOK
            elif symbol_info.filling_mode & SYMBOL_FILLING_IOC:
                filling_mode = mt5.ORDER_FILLING_IOC
            
            logging.info(f"Symbol {symbol} filling mode: {symbol_info.filling_mode}, using {filling_mode}")
        else:
            logging.info(f"Symbol {symbol} has no filling_mode attribute, using default filling mode")
        
        # Print additional debug info
        logging.info(f"Order parameters - Symbol: {symbol}, Type: {order_type}, Volume: {lot_size}, Price: {price}, SL: {sl}, TP: {tp}")
        logging.info(f"Symbol properties - Min Vol: {symbol_info.volume_min}, Max Vol: {symbol_info.volume_max}, Step: {symbol_info.volume_step}")
        
        # Calculate margin required for this order
        margin_check = mt5.order_calc_margin(mt5_order_type, symbol, lot_size, price)
        if margin_check:
            logging.info(f"Margin required for order: {margin_check}, Available: {account_info.margin_free}")
            if margin_check > account_info.margin_free:
                logging.warning(f"Insufficient margin: Required {margin_check}, Available {account_info.margin_free}")
                # Adjust lot size to fit available margin
                adjusted_lot = (account_info.margin_free / margin_check) * lot_size * 0.9  # Use 90% of available
                adjusted_lot = max(adjusted_lot, symbol_info.volume_min)
                adjusted_lot = min(adjusted_lot, symbol_info.volume_max)
                adjusted_lot = round(adjusted_lot / symbol_info.volume_step) * symbol_info.volume_step
                logging.info(f"Adjusting lot size from {lot_size} to {adjusted_lot}")
                lot_size = adjusted_lot
        
        # Define common error descriptions for logging
        error_descriptions = {
            10004: "Requote",
            10006: "Order rejected",
            10007: "Order canceled by client",
            10008: "Order already executed",
            10009: "Order already exists",
            10010: "Order conditions not met",
            10011: "Too many orders",
            10012: "Trade disabled",
            10013: "Market closed",
            10014: "Not enough money",
            10015: "Price changed",
            10016: "Price off",
            10017: "Invalid expiration",
            10018: "Order locked",
            10019: "Buy only allowed",
            10020: "Sell only allowed",
            10021: "Too many requests",
            10022: "Trade timeout",
            10023: "Invalid price",
            10024: "Invalid stops",
            10025: "Invalid trade volume"
        }
        
        # ------------ STRATEGIC ORDER PLACEMENT APPROACHES ------------
        
        # Try multiple order placement strategies in sequence
        
        # STRATEGY 1: Standard order with all parameters
        # NEW: Use higher deviation for Bitcoin orders
        deviation = 50 if is_bitcoin else 10
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5_order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,  # Higher deviation for Bitcoin
            "magic": 12345,
            "comment": comment or (f"BTC Enhanced {order_type}" if is_bitcoin else f"MT5 Trading Bot {order_type}"),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode
        }
        
        logging.info(f"STRATEGY 1: Sending order request with standard parameters: {request}")
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            ticket = result.order
            logging.info(f"Market order placed successfully with STRATEGY 1: {order_type} {symbol} {lot_size} lots at {price}")
            write_trade_notes(f"Opened {order_type} position on {symbol}: {lot_size} lots at {price}, SL: {sl}, TP: {tp}")
            
            # IMPROVED: More robust position verification with extended attempts
            verified = False
            for verification_attempt in range(5):  # Increased from 3 to 5 attempts
                time.sleep(0.5)  # Wait for position to register
                positions = get_positions(symbol)
                position = next((p for p in positions if p.ticket == ticket), None)
                if position:
                    logging.info(f"Position {ticket} verified with SL={position.sl}, TP={position.tp} on attempt {verification_attempt+1}")
                    verified = True
                    break
                time.sleep(0.5)  # Additional wait between attempts
                
            if not verified:
                logging.warning(f"Could not verify position {ticket} was created - please check manually")
                
            return ticket
            
        # Log error details for first attempt with improved description
        error_code = result.retcode if result else -1
        error_message = f"STRATEGY 1 failed: {result.comment if result else 'Unknown error'}"
        error_details = f"(Code: {error_code})"
        
        if error_code in error_descriptions:
            error_details += f", Description: {error_descriptions[error_code]}"
            
        logging.warning(f"{error_message} {error_details}")
            
        # STRATEGY 2: Try without stop loss and take profit (add them later)
        basic_request = request.copy()
        basic_request.pop("sl", None)
        basic_request.pop("tp", None)
        # NEW: Higher deviation for Bitcoin in all strategies
        basic_request["deviation"] = 60 if is_bitcoin else 30  # Allow more slippage
        
        logging.info(f"STRATEGY 2: Sending order without SL/TP: {basic_request}")
        result = mt5.order_send(basic_request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            ticket = result.order
            logging.info(f"Market order placed successfully with STRATEGY 2: {order_type} {symbol} {lot_size} lots at {price}")
            
            # Now try to modify the position to add SL/TP
            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": ticket,
                "sl": sl,
                "tp": tp
            }
            
            modify_result = mt5.order_send(modify_request)
            if modify_result and modify_result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"Successfully added SL/TP to position {ticket}")
            else:
                logging.warning(f"Failed to add SL/TP to position {ticket}")
                
            write_trade_notes(f"Opened {order_type} position on {symbol}: {lot_size} lots at {price}, SL: {sl}, TP: {tp}")
            
            # IMPROVED: More robust position verification
            verified = False
            for verification_attempt in range(5):  # Try 5 times
                time.sleep(0.5)  # Wait for position to be registered
                positions = get_positions(symbol)
                position = next((p for p in positions if p.ticket == ticket), None)
                if position:
                    logging.info(f"Position {ticket} verified with SL={position.sl}, TP={position.tp} on attempt {verification_attempt+1}")
                    verified = True
                    break
                time.sleep(0.5)  # Additional wait between attempts
                
            if not verified:
                logging.warning(f"Could not verify position {ticket} was created - please check manually")
                
            return ticket
            
        # STRATEGY 3: Try removing filling mode entirely
        if "type_filling" in basic_request:
            no_filling_request = basic_request.copy()
            del no_filling_request["type_filling"]
            
            logging.info(f"STRATEGY 3: Sending order without filling type: {no_filling_request}")
            result = mt5.order_send(no_filling_request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                ticket = result.order
                logging.info(f"Market order placed successfully with STRATEGY 3: {order_type} {symbol} {lot_size} lots at {price}")
                
                # Try to add SL/TP
                modify_request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": ticket,
                    "sl": sl,
                    "tp": tp
                }
                
                mt5.order_send(modify_request)
                write_trade_notes(f"Opened {order_type} position on {symbol}: {lot_size} lots at {price}, SL: {sl}, TP: {tp}")
                
                # IMPROVED: Verify position was created with multiple attempts
                verified = False
                for verification_attempt in range(5):  # Try 5 times
                    time.sleep(0.5)
                    positions = get_positions(symbol)
                    position = next((p for p in positions if p.ticket == ticket), None)
                    if position:
                        logging.info(f"Position {ticket} verified with SL={position.sl}, TP={position.tp} on attempt {verification_attempt+1}")
                        verified = True
                        break
                    time.sleep(0.5)
                
                if not verified:
                    logging.warning(f"Could not verify position {ticket} was created - please check manually")
                    
                return ticket
                
        # STRATEGY 4: Try with aggressive slippage and FOK filling
        fok_request = request.copy()
        fok_request["type_filling"] = mt5.ORDER_FILLING_FOK
        # NEW: Higher deviation for Bitcoin
        fok_request["deviation"] = 150 if is_bitcoin else 100  # Allow significant slippage
        
        logging.info(f"STRATEGY 4: Sending order with FOK filling and high deviation: {fok_request}")
        result = mt5.order_send(fok_request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            ticket = result.order
            logging.info(f"Market order placed successfully with STRATEGY 4: {order_type} {symbol} {lot_size} lots at {price}")
            write_trade_notes(f"Opened {order_type} position on {symbol}: {lot_size} lots at {price}, SL: {sl}, TP: {tp}")
            
            # IMPROVED: Verify position was created with multiple attempts
            verified = False
            for verification_attempt in range(5):
                time.sleep(0.5)
                positions = get_positions(symbol)
                position = next((p for p in positions if p.ticket == ticket), None)
                if position:
                    logging.info(f"Position {ticket} verified with SL={position.sl}, TP={position.tp} on attempt {verification_attempt+1}")
                    verified = True
                    break
                time.sleep(0.5)
            
            if not verified:
                logging.warning(f"Could not verify position {ticket} was created - please check manually")
                
            return ticket
            
        # STRATEGY 5: Try as a pending order
        pending_type = mt5.ORDER_TYPE_BUY_LIMIT if mt5_order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_SELL_LIMIT
        # NEW: Smaller price adjustment for Bitcoin to increase execution likelihood
        price_adjustment = 0.00005 * price if is_bitcoin else 0.0001 * price
        
        if mt5_order_type == mt5.ORDER_TYPE_BUY:
            limit_price = price - price_adjustment  # Buy limit slightly below current price
        else:
            limit_price = price + price_adjustment  # Sell limit slightly above current price
            
        limit_price = round(limit_price, digits)
        
        pending_request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": lot_size,
            "type": pending_type,
            "price": limit_price,
            "sl": sl,
            "tp": tp,
            "deviation": 50 if is_bitcoin else 10,
            "magic": 12345,
            "comment": comment or (f"BTC Enhanced Pending {order_type}" if is_bitcoin else f"MT5 Trading Bot Pending {order_type}"),
            "type_time": mt5.ORDER_TIME_GTC,
            "expiration": int(time.time()) + 60  # Expire in 60 seconds
        }
        
        logging.info(f"STRATEGY 5: Sending as pending order: {pending_request}")
        result = mt5.order_send(pending_request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            ticket = result.order
            logging.info(f"Pending order placed successfully with STRATEGY 5: {order_type} {symbol} {lot_size} lots at {limit_price}")
            write_trade_notes(f"Opened pending {order_type} on {symbol}: {lot_size} lots at {limit_price}, SL: {sl}, TP: {tp}")
            
            # IMPROVED: Verify order was created with multiple attempts
            verified = False
            for verification_attempt in range(5):
                time.sleep(0.5)
                orders = get_orders(symbol)
                order = next((o for o in orders if o.ticket == ticket), None)
                if order:
                    logging.info(f"Pending order {ticket} verified with SL={order.sl}, TP={order.tp} on attempt {verification_attempt+1}")
                    verified = True
                    break
                time.sleep(0.5)
            
            if not verified:
                logging.warning(f"Could not verify pending order {ticket} was created - please check manually")
                
            return ticket
        
        # STRATEGY 6: Try ultra-minimal approach (for crypto trading)
        minimal_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5_order_type,
            "price": price,
            # NEW: Add deviation for Bitcoin
            "deviation": 80 if is_bitcoin else 30
        }
        
        logging.info(f"STRATEGY 6: Sending ultra-minimal order: {minimal_request}")
        result = mt5.order_send(minimal_request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            ticket = result.order
            logging.info(f"Market order placed successfully with STRATEGY 6: {order_type} {symbol} {lot_size} lots at {price}")
            
            # Wait slightly before attempting to add SL/TP
            time.sleep(0.5)
            
            # Try to add SL/TP
            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": ticket,
                "sl": sl,
                "tp": tp
            }
            
            modify_result = mt5.order_send(modify_request)
            if modify_result and modify_result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"Successfully added SL/TP to position {ticket}")
            else:
                logging.warning(f"Failed to add SL/TP to position {ticket}")
            
            write_trade_notes(f"Opened {order_type} position on {symbol}: {lot_size} lots at {price}, SL: {sl}, TP: {tp}")
            
            # IMPROVED: Verify position was created with multiple attempts
            verified = False
            for verification_attempt in range(5):
                time.sleep(0.5)
                positions = get_positions(symbol)
                position = next((p for p in positions if p.ticket == ticket), None)
                if position:
                    logging.info(f"Position {ticket} verified with SL={position.sl}, TP={position.tp} on attempt {verification_attempt+1}")
                    verified = True
                    break
                time.sleep(0.5)
            
            if not verified:
                logging.warning(f"Could not verify position {ticket} was created - please check manually")
                
            return ticket
        
        # STRATEGY 7: Try with market order type
        market_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5_order_type,
            "price": price,
            "type_filling": mt5.ORDER_FILLING_RETURN,  # Try return filling mode
            # NEW: Add deviation for Bitcoin
            "deviation": 100 if is_bitcoin else 50
        }
        
        logging.info(f"STRATEGY 7: Sending with RETURN filling mode: {market_request}")
        result = mt5.order_send(market_request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            ticket = result.order
            logging.info(f"Market order placed successfully with STRATEGY 7: {order_type} {symbol} {lot_size} lots at {price}")
            
            # Try to add SL/TP
            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": ticket,
                "sl": sl,
                "tp": tp
            }
            
            mt5.order_send(modify_request)
            write_trade_notes(f"Opened {order_type} position on {symbol}: {lot_size} lots at {price}, SL: {sl}, TP: {tp}")
            
            # IMPROVED: Verify position was created with multiple attempts
            verified = False
            for verification_attempt in range(5):
                time.sleep(0.5)
                positions = get_positions(symbol)
                position = next((p for p in positions if p.ticket == ticket), None)
                if position:
                    logging.info(f"Position {ticket} verified with SL={position.sl}, TP={position.tp} on attempt {verification_attempt+1}")
                    verified = True
                    break
                time.sleep(0.5)
            
            if not verified:
                logging.warning(f"Could not verify position {ticket} was created - please check manually")
                
            return ticket
        
        # All strategies failed, run diagnostics
        logging.error(f"All order placement strategies failed for {order_type} {symbol}")
        diagnose_trading_issues(symbol)
        
        return None
    
    except Exception as e:
        logging.error(f"Error placing market order: {e}")
        logging.error(traceback.format_exc())
        return None
    
def close_position(ticket, partial_lots=None):
    """
    Close an open position by ticket number.
    Can partially close if partial_lots is specified.
    Returns True if successful.
    """
    try:
        if not check_mt5_connection():
            return False
        
        # Get position info
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logging.warning(f"Position {ticket} not found. Cannot close.")
            return False
        
        position = positions[0]
        symbol = position.symbol
        position_type = position.type
        
        # Determine price for closing
        if position_type == mt5.ORDER_TYPE_BUY:
            price = mt5.symbol_info_tick(symbol).bid
        else:  # mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).ask
        
        # Determine volume to close
        volume = position.volume
        if partial_lots is not None and partial_lots < volume:
            volume = partial_lots
        
        # Prepare close request
        close_type = mt5.ORDER_TYPE_SELL if position_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 12345,
            "comment": f"Close position {ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK
        }
        
        # Send request
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            # Calculate profit
            entry_price = position.price_open
            
            if position_type == mt5.ORDER_TYPE_BUY:
                profit_points = (price - entry_price) / mt5.symbol_info(symbol).point
            else:
                profit_points = (entry_price - price) / mt5.symbol_info(symbol).point
            
            logging.info(f"Closed position {ticket}: {volume} lots at {price} (Profit: {profit_points} points)")
            
            # Record for statistics
            trade_result = {
                "ticket": ticket,
                "symbol": symbol,
                "type": "BUY" if position_type == mt5.ORDER_TYPE_BUY else "SELL",
                "entry_price": entry_price,
                "exit_price": price,
                "profit": position.profit * (volume / position.volume),
                "profit_pips": profit_points,
                "entry_time": datetime.fromtimestamp(position.time),
                "exit_time": datetime.now(),
                "reason": "Manual close" if partial_lots is None else "Partial close"
            }
            
            # NEW: Collect trade results for ML training
            manage_completed_trade(trade_result)
            
            return True
        else:
            error_message = f"Close failed: {result.comment}" if result else "Close failed: Unknown error"
            logging.error(f"Failed to close position {ticket}: {error_message}")
            return False
    
    except Exception as e:
        logging.error(f"Error closing position: {e}")
        return False
    
def close_all_positions(symbol=None, only_profit=False, only_loss=False):
    """
    Close all open positions for a symbol or all symbols.
    Can filter to close only profitable or only losing positions.
    Returns the number of positions closed.
    """
    try:
        if not check_mt5_connection():
            return 0
        
        # Get positions
        positions = get_positions(symbol)
        
        if not positions:
            return 0
        
        closed_count = 0
        
        for position in positions:
            # Apply filters
            if only_profit and position.profit <= 0:
                continue
            
            if only_loss and position.profit >= 0:
                continue
            
            # Close position
            if close_position(position.ticket):
                closed_count += 1
        
        return closed_count
    
    except Exception as e:
        logging.error(f"Error closing all positions: {e}")
        return 0

def check_drawdown(symbol=None):
    """
    Check if current drawdown exceeds maximum allowed levels.
    Returns True if drawdown limit is reached.
    """
    try:
        # Get account info
        account_info = get_account_info()
        if account_info is None:
            return False
        
        # Calculate daily drawdown
        daily_drawdown = (state.initial_balance - account_info.balance) / state.initial_balance
        
        if daily_drawdown >= MAX_DRAWDOWN_PER_DAY:
            logging.warning(f"Maximum daily drawdown reached: {daily_drawdown*100:.2f}% > {MAX_DRAWDOWN_PER_DAY*100:.2f}%")
            return True
        
        # If symbol is specified, check trade-specific drawdown
        if symbol:
            positions = get_positions(symbol)
            
            for position in positions:
                entry_balance = state.initial_balance
                current_balance = account_info.balance
                trade_drawdown = (entry_balance - current_balance) / entry_balance
                
                if trade_drawdown >= MAX_DRAWDOWN_PER_TRADE:
                    logging.warning(f"Maximum trade drawdown reached for {symbol}: {trade_drawdown*100:.2f}% > {MAX_DRAWDOWN_PER_TRADE*100:.2f}%")
                    return True
        
        return False
    
    except Exception as e:
        logging.error(f"Error checking drawdown: {e}")
        return False

# ======================================================
# TRADING STRATEGY FUNCTIONS
# ======================================================

def trend_following_strategy(symbol):
    try:
        # Get market structure and trend analysis
        market_state = analyze_market_structure(symbol)
        trend_analysis = analyze_trend(symbol)
        
        # Get data for primary timeframe
        optimized_tf = optimize_timeframe_for_volatility(symbol)
        df = get_candle_data(symbol, optimized_tf)
        if df is None:
            return {"signal": "NONE"}
        
        # Calculate indicators
        adx, plus_di, minus_di = calculate_adx(df)
        rsi = calculate_rsi(df)
        macd_line, signal_line, histogram = calculate_macd(df)
        atr = calculate_atr(df)
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Default: no signal
        signal = "NONE"
        
        # Lower the requirements for trend strength
        min_trend_strength = 0.3  # Reduced from 0.5
        min_adx = 20.0  # Reduced from ADX_THRESHOLD (likely 25)
        
        # Check for trending market with less strict conditions
        if "consensus" in trend_analysis:
            consensus = trend_analysis["consensus"]
            trend_strength = consensus["strength"]
            
            # Only trade if trend is strong enough (with reduced threshold)
            if trend_strength >= min_trend_strength and adx >= min_adx:
                # Buy signal with wider RSI range
                if consensus["direction"] == "bullish" and plus_di > minus_di:
                    # Additional confirmation with wider RSI range
                    if rsi > 35 and rsi < 75:  # Was 40-70
                        signal = "BUY"
                
                # Sell signal with wider RSI range
                elif consensus["direction"] == "bearish" and minus_di > plus_di:
                    # Additional confirmation with wider RSI range
                    if rsi < 65 and rsi > 25:  # Was 60-30
                        signal = "SELL"
        
        # Calculate entry, stop loss, and take profit
        if signal == "BUY":
            entry_price = mt5.symbol_info_tick(symbol).ask
            stop_loss, take_profit = calculate_stop_loss_take_profit(
                symbol, "BUY", entry_price, atr, market_state.get("key_levels")
            )
        elif signal == "SELL":
            entry_price = mt5.symbol_info_tick(symbol).bid
            stop_loss, take_profit = calculate_stop_loss_take_profit(
                symbol, "SELL", entry_price, atr, market_state.get("key_levels")
            )
        else:
            return {"signal": "NONE"}
        
        # Validate stop loss and take profit
        if stop_loss is None or take_profit is None:
            return {"signal": "NONE"}
        
        # Evaluate risk-reward
        risk_reward = evaluate_risk_reward(symbol, signal, entry_price, stop_loss, take_profit)
        
        # Slightly reduce minimum risk-reward requirement to generate more signals
        min_risk_reward = MIN_RISK_REWARD_RATIO * 0.9  # 90% of the configured value
        
        if risk_reward["risk_reward_ratio"] < min_risk_reward:
            logging.info(f"Signal {signal} for {symbol} rejected: Risk-reward ratio {risk_reward['risk_reward_ratio']:.2f} below minimum {min_risk_reward}")
            return {"signal": "NONE"}
        
        return {
            "signal": signal,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": atr,
            "risk_reward": risk_reward,
            "adx": adx,
            "rsi": rsi,
            "trend_strength": trend_analysis["consensus"]["strength"] if "consensus" in trend_analysis else 0,
            "strategy": "trend_following"
        }
    
    except Exception as e:
        logging.error(f"Error in trend following strategy: {e}")
        return {"signal": "NONE"}

def breakout_strategy(symbol):
    """
    Breakout strategy based on support/resistance levels and volatility.
    Returns a dictionary with signal, entry price, stop loss, and take profit.
    """
    try:
        # Get market data
        optimized_tf = optimize_timeframe_for_volatility(symbol)
        df = get_candle_data(symbol, optimized_tf)
        if df is None:
            return {"signal": "NONE"}
        
        # Get market structure
        market_state = analyze_market_structure(symbol)
        
        # Allow breakout trades in all market structures, not just ranging
        
        # Calculate volatility
        atr = calculate_atr(df)
        volatility_level = market_state["volatility"]

        # Modify the volatility-based multipliers section:
        if volatility_level == "low":
            sl_multiplier = 1.5
            tp_multiplier = 2.5
        elif volatility_level == "normal":
            sl_multiplier = 1.8
            tp_multiplier = 2.8
        elif volatility_level == "high":
            sl_multiplier = 2.0  # Modified: was 2.5, make it less conservative
            tp_multiplier = 2.8  # Modified: was 3.5, slightly more balanced
        else:  # "extreme"
            sl_multiplier = 2.2  # Modified: was 3.0, less conservative 
            tp_multiplier = 3.0  # Modified: was 4.0, more balanced
        
        # Identify support and resistance levels
        supp_res = detect_support_resistance(df)
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Get price action insights
        price_action = detect_price_action(df)
        
        # Check for breakouts
        signal = "NONE"
        entry_price = None
        
        # Enhance breakout detection - look for near-breakouts too
        near_breakout_threshold = 0.0015  # 0.15% from key level
        
        # Check for actual or near breakouts
        if price_action.get("breakout_up", False) or price_action.get("near_high", False):
            # Less strict confirmation
            if price_action.get("short_term_trend") == "up":
                signal = "BUY"
                entry_price = mt5.symbol_info_tick(symbol).ask
        
        elif price_action.get("breakout_down", False) or price_action.get("near_low", False):
            # Less strict confirmation
            if price_action.get("short_term_trend") == "down":
                signal = "SELL"
                entry_price = mt5.symbol_info_tick(symbol).bid
        
        # Calculate stop loss and take profit
        if signal in ["BUY", "SELL"]:
            stop_loss, take_profit = calculate_stop_loss_take_profit(
                symbol, signal, entry_price, atr, market_state.get("key_levels")
            )
            
            # Add this to boost signal strength during high volatility:
            if (volatility_level == "high" or volatility_level == "extreme"):
                # For breakouts during high volatility, slightly improve risk-reward by adjusting stop loss
                if signal == "BUY":
                    stop_loss = entry_price - (atr * sl_multiplier * 0.9)  # Tighter stop for better R:R
                else:  # "SELL"
                    stop_loss = entry_price + (atr * sl_multiplier * 0.9)  # Tighter stop for better R:R
                
                # Recalculate take profit based on adjusted stop loss for consistent R:R
                if signal == "BUY":
                    risk_distance = entry_price - stop_loss
                    take_profit = entry_price + (risk_distance * tp_multiplier / sl_multiplier)
                else:  # "SELL"
                    risk_distance = stop_loss - entry_price
                    take_profit = entry_price - (risk_distance * tp_multiplier / sl_multiplier)
            
            # Validate stop loss and take profit
            if stop_loss is None or take_profit is None:
                return {"signal": "NONE"}
            
            # Evaluate risk-reward
            risk_reward = evaluate_risk_reward(symbol, signal, entry_price, stop_loss, take_profit)
            
            # Add more detailed logging for better analysis
            risk_reward_ratio = risk_reward.get("risk_reward_ratio", 0)
            logging.info(f"BREAKOUT SIGNAL DETAILS - {symbol}: {signal}, Entry: {entry_price}, " +
                     f"SL: {stop_loss}, TP: {take_profit}, " +
                     f"Risk-Reward: {risk_reward_ratio:.2f}, " +
                     f"Volatility: {volatility_level}")
            
            if not risk_reward["meets_min_ratio"]:
                adjusted_min_ratio = risk_reward.get("adjusted_min_ratio", MIN_RISK_REWARD_RATIO)
                logging.info(f"Breakout signal {signal} for {symbol} rejected: Risk-reward ratio {risk_reward_ratio:.2f} below adjusted minimum {adjusted_min_ratio:.2f}")
                return {"signal": "NONE"}
            
            return {
                "signal": signal,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": atr,
                "risk_reward": risk_reward,
                "breakout_type": "upward" if signal == "BUY" else "downward",
                "volatility": volatility_level
            }
        
        return {"signal": "NONE"}
    
    except Exception as e:
        logging.error(f"Error in breakout strategy: {e}")
        return {"signal": "NONE"}

def reversal_strategy(symbol):
    """
    Reversal strategy based on oversold/overbought conditions and candlestick patterns.
    Returns a dictionary with signal, entry price, stop loss, and take profit.
    """
    try:
        # Get market data
        optimized_tf = optimize_timeframe_for_volatility(symbol)
        df = get_candle_data(symbol, optimized_tf)
        if df is None:
            return {"signal": "NONE"}
        
        # Get market structure
        market_state = analyze_market_structure(symbol)
        
        # Calculate indicators
        rsi = calculate_rsi(df)
        stoch_k, stoch_d = calculate_stochastic(df)
        macd_line, signal_line, histogram = calculate_macd(df)
        atr = calculate_atr(df)
        
        # Detect candlestick patterns
        patterns = detect_candlestick_patterns(df)
        
        # Get price action insights
        price_action = detect_price_action(df)
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Default: no signal
        signal = "NONE"
        
        # Check for reversal conditions
        
        # Bullish reversal
        bullish_reversal_conditions = [
            rsi < 30,  # Oversold RSI
            stoch_k < 20 and stoch_d < 20,  # Oversold Stochastic
            patterns.get("bullish_count", 0) >= 1,  # Bullish candlestick pattern
            price_action.get("near_low", False)  # Near recent low
        ]
        
        # Bearish reversal
        bearish_reversal_conditions = [
            rsi > 70,  # Overbought RSI
            stoch_k > 80 and stoch_d > 80,  # Overbought Stochastic
            patterns.get("bearish_count", 0) >= 1,  # Bearish candlestick pattern
            price_action.get("near_high", False)  # Near recent high
        ]
        
        # Count confirmed conditions
        bullish_count = sum(bullish_reversal_conditions)
        bearish_count = sum(bearish_reversal_conditions)
        
        # Generate signal based on reversal conditions
        if bullish_count >= 3:
            signal = "BUY"
            entry_price = mt5.symbol_info_tick(symbol).ask
        elif bearish_count >= 3:
            signal = "SELL"
            entry_price = mt5.symbol_info_tick(symbol).bid
        
        # Calculate stop loss and take profit
        if signal in ["BUY", "SELL"]:
            stop_loss, take_profit = calculate_stop_loss_take_profit(
                symbol, signal, entry_price, atr, market_state.get("key_levels")
            )
            
            # Validate stop loss and take profit
            if stop_loss is None or take_profit is None:
                return {"signal": "NONE"}
            
            # Evaluate risk-reward
            risk_reward = evaluate_risk_reward(symbol, signal, entry_price, stop_loss, take_profit)
            
            if not risk_reward["meets_min_ratio"]:
                logging.info(f"Reversal signal {signal} for {symbol} rejected: Risk-reward ratio {risk_reward['risk_reward_ratio']:.2f} below minimum {MIN_RISK_REWARD_RATIO}")
                return {"signal": "NONE"}
            
            return {
                "signal": signal,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": atr,
                "risk_reward": risk_reward,
                "rsi": rsi,
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
                "patterns": patterns
            }
        
        return {"signal": "NONE"}
    
    except Exception as e:
        logging.error(f"Error in reversal strategy: {e}")
        return {"signal": "NONE"}

def get_trade_signal(symbol):
    """
    Get trade signal for a symbol using the enhanced consolidated signal approach.
    """
    try:
        # These checks remain the same as your original function
        if check_news_events(symbol):
            logging.info(f"No signal for {symbol}: High-impact news detected")
            return {"signal": "NONE", "reason": "High-impact news detected"}
        
        # Check for maximum open positions
        positions = get_positions(symbol)
        if len(positions) >= MAX_OPEN_POSITIONS:
            logging.info(f"No signal for {symbol}: Maximum open positions reached")
            return {"signal": "NONE", "reason": "Maximum open positions reached"}
        
        # Check for maximum drawdown
        if check_drawdown(symbol):
            logging.info(f"No signal for {symbol}: Maximum drawdown reached")
            return {"signal": "NONE", "reason": "Maximum drawdown reached"}
        
        # This is the key change - use the consolidated signal calculation
        # instead of evaluating strategies separately
        return calculate_consolidated_signal(symbol)
    
    except Exception as e:
        logging.error(f"Error getting trade signal: {e}")
        return {"signal": "NONE", "reason": f"Error: {str(e)}"}
    
def optimize_timeframe_for_volatility(symbol):
    """
    Dynamically adjust timeframes based on current market volatility.
    """
    try:
        volatility_level, atr_percent = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
        
        if volatility_level == "extreme":
            return mt5.TIMEFRAME_H1  # Higher timeframe for extreme volatility
        elif volatility_level == "high":
            return mt5.TIMEFRAME_M30
        elif volatility_level == "low":
            return mt5.TIMEFRAME_M5  # Lower timeframe for low volatility
        else:
            return mt5.TIMEFRAME_M15  # Default for normal volatility
    except Exception as e:
        logging.error(f"Error optimizing timeframe: {e}")
        return MT5_TIMEFRAMES["PRIMARY"]  # Return default primary timeframe

# ======================================================
# MAIN TRADING LOGIC
# ======================================================


def execute_trading_cycle_enhanced(symbol):
    """
    Enhanced trading cycle that implements all the improvements.
    ENHANCED: Better handling for super-extreme volatility and range-bound markets
    """
    try:
        logging.info(f"Starting enhanced trading cycle for {symbol}")

        # Initialize ML validation system if needed
        if not hasattr(state, 'signal_validation_models'):
            setup_ml_validation_system()

        # Weekly maintenance routine
        optimize_crypto_parameters()
        
        # Check if symbol is in cooldown period after margin errors
        current_time = time.time()
        if hasattr(state, 'margin_error_cooldown') and symbol in state.margin_error_cooldown:
            if current_time < state.margin_error_cooldown[symbol]:
                cooldown_minutes = (state.margin_error_cooldown[symbol] - current_time) / 60
                logging.info(f"Skipping {symbol}: In margin error cooldown ({cooldown_minutes:.1f} minutes remaining)")
                return
            else:
                # Cooldown expired
                del state.margin_error_cooldown[symbol]
                logging.info(f"Margin error cooldown expired for {symbol}")
        
        # Get account info
        account_info = get_account_info()
        if account_info is None:
            logging.error("Failed to get account info. Skipping trading cycle.")
            return
        
        # More robust position check - make multiple attempts to verify position status
        positions = None
        for attempt in range(3):
            positions = get_positions(symbol)
            if positions is not None:
                break
            time.sleep(0.2)  # Short delay between attempts
            
        if positions is None:
            logging.error(f"Failed to get positions for {symbol}. Skipping trading cycle.")
            return
            
        # Strict limit - only allow ONE position per symbol
        if len(positions) > 0:
            logging.info(f"Already have {len(positions)} open position(s) for {symbol}. Skipping.")
            return
            
        # Check if we have too many positions overall
        all_positions = get_positions()
        if len(all_positions) >= MAX_OPEN_POSITIONS:
            logging.info(f"Maximum overall positions reached ({len(all_positions)}/{MAX_OPEN_POSITIONS}). Skipping.")
            return
        
        # Only check margin level if we have open positions
        if len(all_positions) > 0 and account_info.margin > 0:
            # Hard safety check - prevent trading in critical margin conditions
            if account_info.margin_level is not None and account_info.margin_level < 200:
                logging.warning(f"Skipping {symbol}: Margin level too low ({account_info.margin_level:.1f}%)")
                return
            
            # Check margin usage
            margin_usage = (account_info.margin / account_info.equity) * 100 if account_info.equity > 0 else 0
            if margin_usage > 70:
                logging.warning(f"Skipping {symbol}: Margin usage too high ({margin_usage:.1f}%)")
                return
        
        # Check if account balance is sufficient
        if account_info.balance < MIN_BALANCE:
            logging.warning(f"Account balance ({account_info.balance}) below minimum ({MIN_BALANCE}). Skipping trading.")
            return
        
        # Update balance statistics
        if state.initial_balance == 0:
            state.initial_balance = account_info.balance
        
        # Check if symbol is available for trading
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            logging.error(f"Symbol {symbol} is not available. Skipping.")
            return
        
        # Check if symbol is enabled for trading
        if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
            logging.warning(f"Symbol {symbol} is not enabled for trading. Skipping.")
            return
        
        # Check for high-impact news events
        if check_news_events(symbol):
            logging.info(f"No signal for {symbol}: High-impact news detected")
            return
        
        # Check for maximum drawdown
        if check_drawdown(symbol):
            logging.info(f"No signal for {symbol}: Maximum drawdown reached")
            return
        
        # ENHANCED: More flexible in allowing trades during low liquidity for Bitcoin in super-extreme volatility
        is_bitcoin = symbol.startswith("BTC") or "BTC" in symbol
        volatility_level, _ = calculate_market_volatility(symbol, MT5_TIMEFRAMES["PRIMARY"])
        
        # Check if we're in a high liquidity session - with special handling for Bitcoin
        if not is_high_liquidity_session(symbol):
            if is_bitcoin and volatility_level == "super-extreme":
                # For Bitcoin in super-extreme volatility, continue despite low liquidity
                logging.info(f"Continuing with {symbol} in low liquidity session due to super-extreme volatility")
            else:
                logging.info(f"Skipping {symbol}: Low liquidity session")
                # Try to get a potential signal despite the filter
                potential_signal = calculate_consolidated_signal(symbol)
                log_missed_opportunity(symbol, "Low liquidity session", potential_signal)
                return
        
        # Check for sudden price movements - with special handling for Bitcoin
        sudden_movement, price_range_pct = detect_sudden_price_movement(symbol)
        if sudden_movement:
            # For Bitcoin in super-extreme volatility, only skip on extremely large movements
            if is_bitcoin and volatility_level == "super-extreme":
                if price_range_pct > 5.0:  # Only skip for movements > 5%
                    logging.warning(f"Skipping {symbol}: Extreme sudden price movement detected ({price_range_pct:.2f}%)")
                    return
                else:
                    logging.info(f"Allowing {symbol} despite sudden price movement ({price_range_pct:.2f}%) due to super-extreme volatility")
            else:
                logging.warning(f"Skipping {symbol}: Sudden price movement detected ({price_range_pct:.2f}%)")
                return
        
        # Double check for positions again right before getting signal
        positions = get_positions(symbol)
        if len(positions) > 0:
            logging.info(f"Position opened for {symbol} during cycle. Skipping.")
            return
            
        # Get consolidated trading signal with improved debouncing
        signal_data = calculate_consolidated_signal(symbol)
        signal = signal_data.get("signal", "NONE")

        # Add chart pattern analysis results to signal data
        pattern_data = detect_advanced_chart_patterns(symbol)
        signal_data["chart_patterns"] = pattern_data
        
        # Store current signal
        state.current_signals[symbol] = signal_data
        
        # Check if we have a valid signal
        if signal == "NONE":
            reason = signal_data.get("reason", "No valid signal")
            logging.info(f"No trading signal for {symbol}: {reason}")
            return
        
        # Apply ML validation if available - with special handling for Bitcoin in super-extreme volatility
        valid_signal, confidence = validate_trading_signal(symbol, signal_data)
        
        # ENHANCED: Lower validation threshold for Bitcoin in super-extreme volatility
        if not valid_signal and is_bitcoin and volatility_level == "super-extreme":
            if confidence >= 0.55:  # Lower threshold (from default of 0.65)
                logging.info(f"Allowing {symbol} signal despite lower ML confidence ({confidence:.2f}) due to super-extreme volatility")
                valid_signal = True
        
        if not valid_signal:
            logging.info(f"ML validation rejected {signal} signal for {symbol}")
            return
        
        # Extract signal data
        entry_price = signal_data.get("entry_price")
        stop_loss = signal_data.get("stop_loss")
        take_profit = signal_data.get("take_profit")
        lot_size = signal_data.get("lot_size", 0.01)
        
        # Advanced signal verification for volatile conditions
        market_state = analyze_market_structure(symbol)
        strength = signal_data.get("strength", 0)

        # ENHANCED: Relaxed verification for Bitcoin in super-extreme volatility
        if signal == "BUY":
            # Only check strength for non-Bitcoin or lower volatility
            if volatility_level == "super-extreme" and not is_bitcoin and strength < 0.3:
                logging.info(f"Rejecting BUY signal for {symbol}: Insufficient strength ({strength:.2f}) for super-extreme volatility")
                return
            # For Bitcoin in super-extreme, use a lower threshold
            elif volatility_level == "super-extreme" and is_bitcoin and strength < 0.25:
                logging.info(f"Rejecting BUY signal for {symbol}: Insufficient strength ({strength:.2f}) for Bitcoin in super-extreme volatility")
                return
            elif volatility_level == "extreme" and strength < 0.35:
                logging.info(f"Rejecting BUY signal for {symbol}: Insufficient strength ({strength:.2f}) for extreme volatility")
                return
            elif volatility_level == "high" and strength < 0.33:
                logging.info(f"Rejecting BUY signal for {symbol}: Insufficient strength ({strength:.2f}) for high volatility")
                return
                
            # Verify reasonable stop loss distance
            if stop_loss is not None and entry_price is not None:
                sl_distance = entry_price - stop_loss
                # Verify SL is not too wide
                if sl_distance > 0:  # Only for valid SL values
                    expected_atr = calculate_atr(get_candle_data(symbol, MT5_TIMEFRAMES["PRIMARY"]))

                    # ENHANCED: Special handling for Bitcoin in super-extreme volatility
                    if is_bitcoin:
                        if volatility_level == "super-extreme":
                            # Use percentage-based SL for super-extreme volatility in Bitcoin
                            percentage_sl = entry_price * 0.025  # 2.5% stop loss
                            max_reasonable_sl = max(percentage_sl, expected_atr * 20)  # Increased from 18
                        else:
                            max_reasonable_sl = expected_atr * 18
                    elif symbol.startswith("ETH") or "ETH" in symbol:
                        max_reasonable_sl = expected_atr * 12
                    else:
                        max_reasonable_sl = expected_atr * 4
                        
                    if sl_distance > max_reasonable_sl:
                        logging.warning(f"Rejecting BUY signal for {symbol}: SL distance too wide ({sl_distance:.2f} vs max {max_reasonable_sl:.2f})")
                        return

        if entry_price is None or stop_loss is None or take_profit is None:
            logging.error(f"Invalid signal data for {symbol}. Skipping.")
            return
        
        # Validate with risk-reward check including minimum viable profit
        risk_reward = signal_data.get("risk_reward", {})
        
        # ENHANCED: Add special handling for Bitcoin in super-extreme volatility
        if not risk_reward.get("meets_min_ratio", False):
            # For Bitcoin in super-extreme volatility, allow slightly lower risk-reward
            if is_bitcoin and volatility_level == "super-extreme":
                # Manually check against a lower threshold
                rr_ratio = risk_reward.get("risk_reward_ratio", 0)
                if rr_ratio >= 1.3:  # Allow RR as low as 1.3 (vs normal 1.5)
                    logging.info(f"Allowing {symbol} trade with lower RR ratio ({rr_ratio:.2f}) due to super-extreme volatility")
                else:
                    reason = risk_reward.get("reason", "Risk-reward not favorable")
                    logging.info(f"Signal {signal} for {symbol} rejected: {reason}")
                    return
            else:
                reason = risk_reward.get("reason", "Risk-reward not favorable")
                logging.info(f"Signal {signal} for {symbol} rejected: {reason}")
                return
            
        # Final position check right before order placement
        final_positions = get_positions(symbol)
        if len(final_positions) > 0:
            logging.info(f"Position already opened for {symbol}. Skipping order placement.")
            return
        
        # ENHANCED: More detailed comment for super-extreme volatility trades
        comment = f"ML: {confidence:.2f}"
        if pattern_data.get("patterns_found"):
            patterns_str = ", ".join(list(pattern_data.get("patterns_found", {}).keys())[:3])  # Top 3 patterns
            comment += f", Patterns: {patterns_str}"
            
        if volatility_level == "super-extreme":
            comment += f", SuperX Volatility"
        
        # Place order with enhanced order placement function
        ticket = place_market_order(symbol, signal, lot_size, stop_loss, take_profit, comment=comment)
        
        if ticket:
            logging.info(f"Order placed: {signal} {symbol} at {entry_price} with SL={stop_loss}, TP={take_profit}, Lot={lot_size}")
            state.last_trade_time[symbol] = time.time()
            
            # Store the signal data for later ML training
            if not hasattr(state, 'pending_signals'):
                state.pending_signals = {}
            
            state.pending_signals[ticket] = signal_data
            
            write_trade_notes(f"Placed {signal} order on {symbol}: Entry={entry_price}, SL={stop_loss}, TP={take_profit}, Lot={lot_size}")
        else:
            logging.error(f"Failed to place order for {symbol}")
    
    except Exception as e:
        logging.error(f"Error in enhanced trading cycle for {symbol}: {e}")
        logging.error(traceback.format_exc())

def manage_open_positions_enhanced():
    """
    Enhanced position management with advanced trailing stop, partial closing and emergency protection.
    Includes optimizations for reduced logging and calculation frequency.
    """
    try:
        # Only log position management start every 5 cycles
        should_log_cycle = int(time.time()) % (5 * UPDATE_INTERVAL) < UPDATE_INTERVAL
        
        if should_log_cycle:
            logging.info("Starting position management cycle")
        
        # Check for emergency conditions first
        emergency, emergency_type, details = detect_emergency_conditions()
        if emergency:
            logging.warning(f"EMERGENCY CONDITION DETECTED: {emergency_type} - {details}")
            
            # Apply emergency protection based on type
            if emergency_type == "extreme_volatility":
                symbol = details.get("symbol")
                if symbol:
                    # Move stops to breakeven for the affected symbol
                    positions = get_positions(symbol)
                    for position in positions:
                        # Move stop loss to breakeven
                        entry_price = position.price_open
                        update_position_stops(symbol, position.ticket, entry_price)
                        logging.warning(f"EMERGENCY: Moved stop loss to breakeven for position {position.ticket} due to extreme volatility")
            
            elif emergency_type == "critical_margin":
                # Close worst performing positions to free up margin
                positions = get_positions()
                
                # Sort by profit (ascending)
                sorted_positions = sorted(positions, key=lambda p: p.profit)
                
                # Close the worst 50% of positions
                positions_to_close = sorted_positions[:len(sorted_positions)//2]
                for position in positions_to_close:
                    close_position(position.ticket)
                    logging.warning(f"EMERGENCY: Closed position {position.ticket} due to critical margin level")
            
            elif emergency_type == "extreme_drawdown":
                # Close all positions to prevent further drawdown
                close_all_positions()
                logging.warning(f"EMERGENCY: Closed all positions due to extreme drawdown")
                
            elif emergency_type == "connection_issues":
                # Protect all positions with safe stops
                positions = get_positions()
                for position in positions:
                    symbol = position.symbol
                    entry_price = position.price_open
                    position_type = position.type
                    
                    # Get current price
                    try:
                        current_price = mt5.symbol_info_tick(symbol).bid if position_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
                        
                        # Ensure position has a stop loss that's closer to breakeven
                        if position_type == mt5.ORDER_TYPE_BUY:
                            if position.sl is None or position.sl < entry_price:
                                # For profit positions, set stop to breakeven
                                if current_price > entry_price:
                                    new_sl = entry_price
                                else:
                                    # For loss positions, try to minimize loss - set stop 1% away
                                    new_sl = current_price * 0.99
                                    
                                update_position_stops(symbol, position.ticket, new_sl)
                                logging.warning(f"EMERGENCY: Added protection stop for position {position.ticket} due to connection issues")
                        
                        elif position_type == mt5.ORDER_TYPE_SELL:
                            if position.sl is None or position.sl > entry_price:
                                # For profit positions, set stop to breakeven
                                if current_price < entry_price:
                                    new_sl = entry_price
                                else:
                                    # For loss positions, try to minimize loss - set stop 1% away
                                    new_sl = current_price * 1.01
                                    
                                update_position_stops(symbol, position.ticket, new_sl)
                                logging.warning(f"EMERGENCY: Added protection stop for position {position.ticket} due to connection issues")
                    except:
                        logging.error(f"Could not protect position {position.ticket} during connection issues")
                
                # Call shutdown handler to ensure proper position protection
                shutdown_handler()

        # Get all open positions
        positions = get_positions()
        
        if not positions:
            if should_log_cycle:
                logging.info("No open positions to manage")
            return
        
        if should_log_cycle:
            logging.info(f"Managing {len(positions)} open positions")
        
        # Initialize current_atr attribute if it doesn't exist
        if not hasattr(state, 'current_atr'):
            state.current_atr = {}
        
        for position in positions:
            symbol = position.symbol
            ticket = position.ticket
            position_type = position.type
            
            # Throttled logging for position management
            should_log_position = hasattr(globals(), 'throttled_log') and callable(globals()['throttled_log'])
            
            if should_log_position:
                should_log_position = throttled_log(f"manage_position_{ticket}", 
                                             f"Managing position {ticket} {symbol} ({position.volume} lots)",
                                             min_interval=60)
            else:
                # Fallback if throttled_log function doesn't exist
                should_log_position = should_log_cycle
                if should_log_position:
                    logging.info(f"Managing position {ticket} {symbol} ({position.volume} lots)")
            
            # Get symbol info
            symbol_info = get_symbol_info(symbol)
            if symbol_info is None:
                logging.warning(f"Symbol info not available for {symbol}. Skipping position {ticket}.")
                continue
            
            # Get current price
            current_price = mt5.symbol_info_tick(symbol).bid if position_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
            
            # Get entry price
            entry_price = position.price_open
            
            # Calculate profit in points
            symbol_point = symbol_info.point
            
            if position_type == mt5.ORDER_TYPE_BUY:
                profit_points = (current_price - entry_price) / symbol_point
            else:  # mt5.ORDER_TYPE_SELL
                profit_points = (entry_price - current_price) / symbol_point
            
            # Calculate profit percentage
            profit_percent = profit_points / (entry_price / symbol_point) * 100
            
            if should_log_position:
                logging.info(f"Position {ticket} {symbol}: Current profit: {profit_percent:.2f}%, " +
                         f"TSL activation threshold: {TSL_ACTIVATION_THRESHOLD * 100:.2f}%")
            
            # Get current ATR (with less frequent calculation)
            if int(time.time()) % 300 < 10 or symbol not in state.current_atr:  # Once every 5 minutes or if not calculated yet
                # Use optimized timeframe
                optimized_tf = optimize_timeframe_for_volatility(symbol)
                df = get_candle_data(symbol, optimized_tf)
                if df is not None:
                    atr = calculate_atr(df)
                    if should_log_position:
                        logging.info(f"Current ATR for {symbol}: {atr}")
                    
                    # Store ATR in state for reuse
                    state.current_atr[symbol] = atr
            else:
                # Use previously calculated ATR
                atr = state.current_atr.get(symbol, 0)
            
            # Ensure every position has a stop loss (emergency protection)
            if position.sl is None:
                # Calculate a reasonable stop based on ATR
                if position_type == mt5.ORDER_TYPE_BUY:
                    new_sl = entry_price - (atr * 2)  # 2 ATR below entry
                else:  # mt5.ORDER_TYPE_SELL
                    new_sl = entry_price + (atr * 2)  # 2 ATR above entry
                
                # Round to symbol precision
                digits = symbol_info.digits
                new_sl = round(new_sl, digits)
                
                # Set the stop loss
                update_success = update_position_stops(symbol, ticket, new_sl)
                if update_success:
                    logging.info(f"Added missing stop loss for position {ticket} at {new_sl}")
                else:
                    logging.warning(f"Failed to add stop loss for position {ticket}")
            
            # Check for partial closing opportunities
            manage_position_with_partial_close(symbol, ticket)
            
            # Apply enhanced trailing stop if position is in profit
            if profit_percent >= TSL_ACTIVATION_THRESHOLD * 100 and atr > 0:
                if should_log_position:
                    logging.info(f"Applying enhanced trailing stop for position {ticket}")
                
                new_sl = enhanced_trailing_stop(symbol, ticket, current_price, position_type, entry_price, atr)
                if new_sl is not None and new_sl != position.sl:
                    update_success = update_position_stops(symbol, ticket, new_sl)
                    if update_success:
                        logging.info(f"Updated trailing stop for {symbol} position {ticket} to {new_sl}")
                    else:
                        logging.warning(f"Failed to update trailing stop for position {ticket}")
                elif should_log_position:
                    logging.info(f"No TSL update needed for position {ticket}")
            
            # Check for maximum drawdown per trade
            if check_drawdown(symbol):
                # Close position if maximum drawdown reached
                logging.warning(f"Maximum drawdown reached for position {ticket}")
                close_position(ticket)
                logging.warning(f"Closed position {ticket} due to maximum drawdown reached")
            
            # Check for sudden market movements that might warrant closing
            sudden_movement, price_range_pct = detect_sudden_price_movement(symbol)
            if sudden_movement:
                # Special handling for Bitcoin vs other assets
                is_bitcoin = symbol.startswith("BTC") or "BTC" in symbol
                extreme_threshold = 5.0 if is_bitcoin else 3.0  # Higher threshold for Bitcoin
                
                if price_range_pct > extreme_threshold:  # Extreme movement
                    logging.warning(f"Extreme market movement detected for {symbol}: {price_range_pct:.2f}%")
                    if (position_type == mt5.ORDER_TYPE_BUY and current_price < entry_price) or \
                    (position_type == mt5.ORDER_TYPE_SELL and current_price > entry_price):
                        # Close position if in loss during extreme movement
                        close_position(ticket)
                        logging.warning(f"Closed position {ticket} due to extreme market movement")
        
        if should_log_cycle:
            logging.info("Completed position management cycle")
    
    except Exception as e:
        logging.error(f"Error managing open positions: {e}")
        logging.error(traceback.format_exc())
        # Attempt to protect positions in case of an error
        try:
            shutdown_handler()
        except:
            logging.error("Failed to run shutdown handler during error recovery")

def update_market_analysis_enhanced():
    """
    Enhanced market analysis with more advanced indicators and optimizations.
    This replaces the original update_market_analysis function.
    """
    try:
        for symbol in SYMBOLS:
            # Skip if symbol not available
            symbol_info = get_symbol_info(symbol)
            if symbol_info is None:
                continue
            
            # Analyze enhanced market structure
            market_state = analyze_market_structure_enhanced(symbol)
            
            # Calculate multi-timeframe momentum
            momentum_data = calculate_multi_timeframe_momentum(symbol)
            
            # Analyze currency correlations (for currency pairs)
            correlations = {}
            if len(symbol) >= 6 and symbol[:6].isalpha():
                correlations = analyze_currency_correlations(symbol)
            
            # Store in global state with all enhanced data
            state.market_state[symbol] = {
                "structure": market_state,
                "momentum": momentum_data,
                "correlations": correlations,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Periodically optimize strategy parameters (e.g., once per day)
            # This is resource-intensive, so we do it sparingly
            if random.random() < 0.01:  # ~1% chance on each cycle
                logging.info(f"Performing strategy parameter optimization for {symbol}")
                optimal_params = optimize_strategy_parameters(symbol)
                if optimal_params:
                    # Store optimized parameters in state for later use
                    if "optimized_params" not in state.performance_stats:
                        state.performance_stats["optimized_params"] = {}
                    state.performance_stats["optimized_params"][symbol] = {
                        "params": optimal_params,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
    
    except Exception as e:
        logging.error(f"Error updating enhanced market analysis: {e}")

def trading_worker_enhanced():
    """
    Enhanced trading worker with improved timing, robustness, and emergency protection.
    Modified to adaptively focus attention based on open positions with memory management.
    """
    try:
        logging.info("Starting enhanced trading worker")
        
        # Initialize state
        state.initial_balance = 0
        state.daily_loss = 0
        state.profit_streak = 0
        state.last_trade_time = {}
        state.error_count = 0
        state.safe_mode_until = 0
        
        # Initialize margin safety fields
        if not hasattr(state, 'margin_errors'):
            state.margin_errors = {}
        if not hasattr(state, 'margin_error_cooldown'):
            state.margin_error_cooldown = {}
        
        # Initialize memory management tracking
        if not hasattr(state, 'last_memory_cleanup'):
            state.last_memory_cleanup = time.time()
        
        # Initialize model saving tracking
        if not hasattr(state, 'last_model_save'):
            state.last_model_save = time.time()
        
        # Load trade statistics
        load_trade_stats()
        
        # Calculate cycle interval in seconds (10-15 seconds as recommended)
        update_interval = UPDATE_INTERVAL  # seconds
        
        # Set a longer interval for market analysis updates
        market_analysis_interval = 5 * 60  # 5 minutes
        last_market_analysis = 0
        
        # Risk management update interval (5 minutes)
        risk_management_interval = 5 * 60  # 5 minutes
        last_risk_update = 0
        
        # Memory management interval (10 minutes)
        memory_management_interval = 10 * 60  # 10 minutes
        
        # Model saving interval (30 minutes)
        model_save_interval = 30 * 60  # 30 minutes
        
        # NEW: Track new trade search frequency
        new_trade_search_interval = update_interval  # Start with normal interval
        last_new_trade_search = 0
        
        # NEW: Register our shutdown handler with atexit
        import atexit
        atexit.register(enhanced_shutdown_handler)
        
        # Trading loop
        while not state.exit_requested:
            start_time = time.time()  # Track cycle start time
            
            try:
                # Check for exit request early
                if state.exit_requested:
                    break
                
                # Check MT5 connection
                if not check_mt5_connection():
                    logging.error("MT5 not connected. Attempting to reconnect...")
                    if not connect_to_mt5() or not login_to_mt5():
                        logging.error("Failed to reconnect to MT5. Waiting before retry.")
                        time.sleep(60)
                        continue
                
                # Get account balance
                account_info = get_account_info()
                if account_info is None:
                    logging.error("Failed to get account info. Skipping cycle.")
                    time.sleep(30)
                    continue
                
                # Initialize initial balance if not set
                if state.initial_balance == 0:
                    state.initial_balance = account_info.balance
                    logging.info(f"Initial account balance set to {state.initial_balance}")

                account_info = get_account_info()
                logging.info(f"DEBUG: Margin level: {account_info.margin_level}, Margin: {account_info.margin}, Equity: {account_info.equity}")

                # NEW: Check for emergency conditions at the beginning of each cycle
                emergency, emergency_type, details = detect_emergency_conditions()
                if emergency:
                    logging.warning(f"EMERGENCY CONDITION DETECTED in main loop: {emergency_type}")
                    # Handle emergency specifically
                    if emergency_type == "extreme_drawdown" or emergency_type == "critical_margin":
                        # Stop trading for this cycle
                        logging.warning("Trading paused due to emergency conditions")
                        time.sleep(60)  # Wait longer during emergency
                        continue
                
                # NEW: Perform memory management periodically
                current_time = time.time()
                if current_time - state.last_memory_cleanup > memory_management_interval:
                    logging.info("Performing memory cleanup...")
                    
                    # Clean trade statistics - limit last_trades array
                    if hasattr(state, 'trade_stats') and 'last_trades' in state.trade_stats:
                        if len(state.trade_stats['last_trades']) > 100:  # Keep last 100 trades
                            state.trade_stats['last_trades'] = state.trade_stats['last_trades'][-100:]
                    
                    # Clean current signals cache - limit size
                    if hasattr(state, 'current_signals') and len(state.current_signals) > 50:
                        # Convert to list and keep most recent 50 items
                        items = list(state.current_signals.items())
                        state.current_signals = dict(items[-50:])
                    
                    # Clean missed opportunities
                    if hasattr(state, 'missed_opportunities') and len(state.missed_opportunities) > 200:
                        state.missed_opportunities = state.missed_opportunities[-200:]
                    
                    # Clean pending signals - remove for positions that no longer exist
                    if hasattr(state, 'pending_signals'):
                        active_tickets = [p.ticket for p in get_positions()]
                        keys_to_remove = [t for t in state.pending_signals if t not in active_tickets]
                        for ticket in keys_to_remove:
                            if ticket in state.pending_signals:
                                del state.pending_signals[ticket]
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    state.last_memory_cleanup = current_time
                    logging.info("Memory cleanup completed")
                
                # NEW: Save ML models periodically
                if current_time - state.last_model_save > model_save_interval:
                    if hasattr(state, 'signal_validation_models') and len(state.signal_validation_models) > 0:
                        logging.info("Saving ML models...")
                        
                        # Create models directory if needed
                        models_dir = 'ml_models'
                        if not os.path.exists(models_dir):
                            os.makedirs(models_dir)
                        
                        # Save each model
                        saved_count = 0
                        for symbol, model in state.signal_validation_models.items():
                            try:
                                # Create a safe filename
                                safe_symbol = symbol.replace('/', '_').replace('\\', '_')
                                model_path = os.path.join(models_dir, f"{safe_symbol}_model.pkl")
                                
                                # Save using pickle
                                import pickle
                                with open(model_path, 'wb') as f:
                                    pickle.dump(model, f)
                                saved_count += 1
                            except Exception as e:
                                logging.error(f"Error saving model for {symbol}: {e}")
                        
                        logging.info(f"Saved {saved_count} ML models")
                        state.last_model_save = current_time
                
                # Periodically update risk management
                current_time = time.time()
                if last_risk_update == 0 or (current_time - last_risk_update) > risk_management_interval:
                    logging.info("Updating risk management parameters...")
                    enhance_risk_management()
                    last_risk_update = current_time
                
                # Update market analysis periodically to reduce API load
                current_time = time.time()
                if current_time - last_market_analysis > market_analysis_interval:
                    logging.info("Updating market analysis...")
                    update_market_analysis_enhanced()
                    last_market_analysis = current_time
                
                # Always manage open positions on every cycle
                manage_open_positions_enhanced()
                
                # Get open positions count AFTER management
                positions = get_positions()
                open_position_count = len(positions)
                
                # NEW: Adaptively adjust new trade search frequency based on open positions
                current_time = time.time()
                
                # If we have open positions, reduce how often we look for new trades
                if open_position_count > 0:
                    # Exponentially increase search interval based on position count
                    # With 1 position: 2x normal interval
                    # With 2 positions: 4x normal interval
                    # With 3+ positions: 8x normal interval
                    position_factor = min(3, open_position_count)  # Cap at 3
                    new_trade_search_interval = update_interval * (2 ** position_factor)
                    
                    logging.debug(f"Adjusting new trade search interval: {new_trade_search_interval}s with {open_position_count} open positions")
                else:
                    # No positions - use normal interval
                    new_trade_search_interval = update_interval
                
                # Only search for new trades if enough time has passed since last search
                if current_time - last_new_trade_search > new_trade_search_interval:
                    logging.info(f"Searching for new trade opportunities (interval: {new_trade_search_interval}s)")
                    
                    # Execute trading cycle for each symbol with staggered timing
                    for i, symbol in enumerate(SYMBOLS):
                        # Check for exit request frequently
                        if state.exit_requested:
                            break
                        
                        # NEW: Check for emergency conditions for this symbol
                        symbol_emergency, emergency_type, _ = detect_emergency_conditions(symbol)
                        if symbol_emergency:
                            logging.warning(f"Skipping new trade search for {symbol} - emergency condition: {emergency_type}")
                            continue
                        
                        # Skip symbols with existing positions to focus on other opportunities
                        symbol_positions = get_positions(symbol)
                        if symbol_positions and len(symbol_positions) > 0:
                            logging.debug(f"Skipping new trade search for {symbol} - already have position")
                            continue
                        
                        # Stagger symbol processing to spread API calls
                        symbol_stagger_delay = i * 1.0  # 1 second between symbols
                        time.sleep(symbol_stagger_delay)
                        
                        execute_trading_cycle_enhanced(symbol)
                    
                    # Update last search timestamp
                    last_new_trade_search = current_time
                
                # Calculate time spent in this cycle
                cycle_time = time.time() - start_time
                
                # Sleep for the remainder of the update interval, if any
                sleep_time = max(0, update_interval - cycle_time)
                if sleep_time > 0:
                    # Sleep in shorter intervals to check for exit_requested more frequently
                    for _ in range(int(sleep_time)):
                        if state.exit_requested:
                            break
                        time.sleep(1)
            
            except Exception as e:
                logging.error(f"Error in trading cycle: {e}")
                logging.error(traceback.format_exc())
                
                # Track error frequency
                if not hasattr(state, 'error_count'):
                    state.error_count = 0
                state.error_count += 1
                
                # If too many errors in a short time, enter safe mode
                if state.error_count > 5:
                    state.safe_mode_until = time.time() + 3600  # 1 hour safe mode
                    state.error_count = 0
                    logging.warning("Entering safe mode for 1 hour due to repeated errors")
                    
                    # Protect positions in case of repeated errors
                    try:
                        shutdown_handler()
                    except:
                        logging.error("Failed to run shutdown handler during error recovery")
                
                time.sleep(60)  # Wait longer after error
        
        # Run shutdown handler before exiting
        shutdown_handler()
        logging.info("Enhanced trading worker stopped")
    
    except Exception as e:
        logging.error(f"Trading worker crashed: {e}")
        logging.error(traceback.format_exc())
        # Still try to protect positions on critical failure
        try:
            shutdown_handler()
        except:
            logging.error("Failed to run shutdown handler during critical failure")

def tsl_management_worker():
    """
    Dedicated worker thread that focuses exclusively on trailing stop management.
    This runs more frequently than the main trading cycle to ensure TSLs are updated promptly.
    """
    try:
        logging.info("Starting dedicated TSL management worker")
        
        # Use the configured TSL update interval instead of the main cycle interval
        update_interval = TSL_UPDATE_INTERVAL  # This should be around 1 second
        
        # Track positions that are in profit and need active management
        active_management_positions = {}
        
        while not state.exit_requested:
            start_time = time.time()
            
            try:
                # Check MT5 connection
                if not check_mt5_connection():
                    time.sleep(1)
                    continue
                
                # Get all open positions
                positions = get_positions()
                if not positions:
                    # No positions to manage, sleep for a bit
                    time.sleep(update_interval)
                    continue
                
                # For performance, only process positions that are in profit
                # and have met the TSL activation threshold
                positions_to_update = []
                
                for position in positions:
                    symbol = position.symbol
                    ticket = position.ticket
                    position_type = position.type
                    
                    # Get symbol info
                    symbol_info = get_symbol_info(symbol)
                    if symbol_info is None:
                        continue
                    
                    # Get current price
                    current_price = mt5.symbol_info_tick(symbol).bid if position_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
                    
                    # Calculate profit percentage
                    entry_price = position.price_open
                    symbol_point = symbol_info.point
                    
                    if position_type == mt5.ORDER_TYPE_BUY:
                        profit_points = (current_price - entry_price) / symbol_point
                    else:  # mt5.ORDER_TYPE_SELL
                        profit_points = (entry_price - current_price) / symbol_point
                    
                    profit_percent = profit_points / (entry_price / symbol_point) * 100
                    
                    # Check if position meets TSL activation threshold
                    # Using a relaxed threshold here since we'll let the enhanced_trailing_stop
                    # function make the detailed decision about activation
                    
                    # Calculate position age in hours for time-based threshold reduction
                    position_age_hours = (time.time() - position.time) / 3600  # Hours
                    
                    # Calculate adjusted threshold considering position age
                    # CHANGED: Increased hourly reduction rate from 0.1 to 0.2
                    base_activation = TSL_ACTIVATION_THRESHOLD * 100  # Convert to percentage 
                    adjusted_activation = max(base_activation * 0.5, base_activation - (position_age_hours * 0.2))
                    
                    # Add positions in profit to candidate list for TSL update
                    if profit_percent > 0:  # Any position in profit
                        positions_to_update.append({
                            'position': position,
                            'current_price': current_price,
                            'profit_percent': profit_percent,
                            'adjusted_threshold': adjusted_activation
                        })
                        
                        # Add to active management for more frequent updates
                        active_management_positions[ticket] = True
                    elif ticket in active_management_positions:
                        # Position was in active management but no longer meets criteria
                        del active_management_positions[ticket]
                
                # Process positions that need TSL updates
                for pos_data in positions_to_update:
                    position = pos_data['position']
                    current_price = pos_data['current_price']
                    profit_percent = pos_data['profit_percent']
                    adjusted_threshold = pos_data['adjusted_threshold']
                    
                    symbol = position.symbol
                    ticket = position.ticket
                    position_type = position.type
                    entry_price = position.price_open
                    
                    # Log TSL status periodically
                    # Only log if profit is close to or above the threshold
                    if profit_percent > adjusted_threshold * 0.8:  # Within 80% of threshold
                        logging.info(f"TSL Worker: Position {ticket} {symbol} profit {profit_percent:.2f}%, " +
                                    f"threshold {adjusted_threshold:.2f}%")
                    
                    # Get data for trailing stop calculation
                    optimized_tf = optimize_timeframe_for_volatility(symbol)
                    df = get_candle_data(symbol, optimized_tf)
                    if df is None:
                        continue
                    
                    # Calculate ATR
                    atr = calculate_atr(df)
                    
                    # Apply enhanced trailing stop
                    new_sl = enhanced_trailing_stop(symbol, ticket, current_price, position_type, entry_price, atr)
                    
                    if new_sl is not None and new_sl != position.sl:
                        # Update stop loss
                        success = update_position_stops(symbol, ticket, new_sl)
                        if success:
                            logging.info(f"TSL Worker: Updated trailing stop for {symbol} position {ticket} to {new_sl} ({profit_percent:.2f}% profit)")
                
                # Calculate time spent in this cycle
                cycle_time = time.time() - start_time
                
                # Sleep for the remainder of the update interval, if any
                sleep_time = max(0.1, update_interval - cycle_time)  # At least 0.1s sleep to prevent CPU hogging
                time.sleep(sleep_time)
                
            except Exception as e:
                logging.error(f"Error in TSL management cycle: {e}")
                logging.error(traceback.format_exc())
                time.sleep(1)  # Short sleep after error
        
        logging.info("TSL management worker stopped")
        
    except Exception as e:
        logging.error(f"TSL management worker crashed: {e}")
        logging.error(traceback.format_exc())
        # Try to protect positions if thread crashes
        try:
            shutdown_handler()
        except:
            logging.error("Failed to run shutdown handler during TSL worker crash")

# ======================================================
# MAIN FUNCTION
# ======================================================

def main_enhanced():
    """
    Enhanced main function with integrated improvements for robustness,
    performance monitoring, and error recovery.
    """
    try:
        logging.info("=== Starting Advanced MT5 Trading Bot ===")
        logging.info(f"Configured to trade: {', '.join(SYMBOLS)}")

        # Check and install required packages
        ensure_required_packages()
        
        # NEW: Initialize thread locks if not already present
        if not hasattr(state, 'locks'):
            state.locks = {
                'trade_stats': threading.RLock(),    # For trade statistics
                'positions': threading.RLock(),      # For position management
                'signals': threading.RLock(),        # For signal data
                'market_state': threading.RLock(),   # For market state data
                'ml_models': threading.RLock(),      # For ML models
                'risk_modifier': threading.RLock(),  # For risk adjustment
                'cache': threading.RLock()           # For caching operations
            }
            logging.info("Thread synchronization initialized")
        
        initialize_market_constants()
        logging.info("Market constants initialized")

        # NEW: Initialize memory management tracking
        if not hasattr(state, 'last_memory_cleanup'):
            state.last_memory_cleanup = time.time()
            logging.info("Memory management initialized")
        
        # NEW: Create models directory if it doesn't exist
        models_dir = 'ml_models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logging.info(f"Created models directory: {models_dir}")
        
        # Setup error handling and recovery mechanisms
        setup_error_handling()

        # NEW: Try to load previously saved ML models
        try:
            model_count = 0
            for filename in os.listdir(models_dir):
                if filename.endswith('_model.pkl'):
                    try:
                        symbol = filename.replace('_model.pkl', '').replace('_', '/')
                        model_path = os.path.join(models_dir, filename)
                        
                        import pickle
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                            
                        # Initialize model storage if not exists
                        if not hasattr(state, 'signal_validation_models'):
                            state.signal_validation_models = {}
                        
                        # Store the model
                        state.signal_validation_models[symbol] = model
                        model_count += 1
                    except Exception as e:
                        logging.error(f"Error loading model {filename}: {e}")
            
            if model_count > 0:
                logging.info(f"Loaded {model_count} ML models from disk")
        except Exception as e:
            logging.error(f"Error loading ML models: {e}")
        
        # Initialize ML validation system
        setup_ml_validation_system()
        
        # Reset state
        state.exit_requested = False
        
        # Connect to MT5
        if not connect_to_mt5():
            logging.error("Failed to connect to MT5. Exiting.")
            return
        
        # Login to MT5 account
        if not login_to_mt5():
            logging.error("Failed to login to MT5 account. Exiting.")
            return
        
        # Log account info
        account_info = get_account_info()
        if account_info:
            logging.info(f"Account: {account_info.login}, Balance: {account_info.balance} {account_info.currency}")
            if account_info.balance < MIN_BALANCE:
                logging.warning(f"Account balance ({account_info.balance}) below recommended minimum ({MIN_BALANCE}).")
                choice = input("Continue trading with low balance? (y/n): ")
                if choice.lower() != 'y':
                    logging.info("Exiting as requested.")
                    return
            
            # Initialize initial balance for tracking
            state.initial_balance = account_info.balance
        
        # Load trade statistics
        load_trade_stats()

        # Log initialization of advanced features
        logging.info("ML signal validation system initialized")
        logging.info("Advanced chart pattern recognition system initialized")
        
        # Start trading worker thread
        trading_thread = threading.Thread(target=trading_worker_enhanced, name="TradingWorker")
        trading_thread.daemon = True
        trading_thread.start()
        
        # Start dedicated TSL management thread
        tsl_thread = threading.Thread(target=tsl_management_worker, name="TSLManager")
        tsl_thread.daemon = True
        tsl_thread.start()
        logging.info("Dedicated TSL management thread started")
        
        # Main control loop
        try:
            print("\nAdvanced MT5 Trading Bot running. Commands:")
            print("q - Quit")
            print("s - Show status")
            print("p - Print performance report")
            print("c - Close all positions")
            print("r - Run system diagnostics")
            print("t - Test order placement")
            print("m - Save ML models")  # NEW: Added command to save models on demand
            print("Press Ctrl+C to exit")
            
            while (trading_thread.is_alive() or tsl_thread.is_alive()) and not state.exit_requested:
                try:
                    command = input("Command: ").lower().strip()
                    
                    if command == 'q':
                        logging.info("User requested exit. Stopping trading.")
                        state.exit_requested = True
                        # Run shutdown handler to protect positions
                        enhanced_shutdown_handler()
                        break
                        
                    elif command == 's':
                        # Show current status
                        account_info = get_account_info()
                        positions = get_positions()
                        
                        print(f"\nAccount Balance: {account_info.balance} {account_info.currency}")
                        print(f"Equity: {account_info.equity} {account_info.currency}")
                        
                        if account_info.margin > 0:
                            print(f"Margin Level: {account_info.margin_level:.1f}%")
                        
                        print(f"Open Positions: {len(positions)}")
                        
                        if positions:
                            print("\nOpen Positions:")
                            for pos in positions:
                                profit_pct = (pos.profit / pos.volume) / pos.price_open * 100
                                print(f"  {pos.symbol}: {'BUY' if pos.type == 0 else 'SELL'}, " +
                                     f"Lots: {pos.volume}, Profit: {pos.profit:.2f} ({profit_pct:.2f}%), " +
                                     f"SL: {pos.sl}, TP: {pos.tp}")
                        
                        # NEW: Add memory usage info
                        import sys
                        if hasattr(state, 'trade_stats') and 'last_trades' in state.trade_stats:
                            print(f"Trade history entries: {len(state.trade_stats['last_trades'])}")
                        if hasattr(state, 'signal_validation_models'):
                            print(f"ML models: {len(state.signal_validation_models)}")
                        
                        print(f"\nTrading Status: {'Running' if not state.exit_requested else 'Stopping'}")
                        
                    elif command == 'p':
                        # Show performance report
                        report = generate_performance_report()
                        print("\n" + report)
                        
                    elif command == 'c':
                        # Close all positions
                        confirm = input("Close ALL open positions? (y/n): ")
                        if confirm.lower() == 'y':
                            closed = close_all_positions()
                            print(f"Closed {closed} positions.")
                            
                    elif command == 'r':
                        # Run system diagnostics
                        symbol = input("Enter symbol to diagnose (or press Enter for all): ").upper().strip()
                        if symbol:
                            diagnose_trading_issues(symbol)
                            print(f"Diagnostics for {symbol} written to log file.")
                        else:
                            print("Running system-wide diagnostics...")
                            for s in SYMBOLS:
                                diagnose_trading_issues(s)
                            print("System diagnostics written to log file.")
                            
                    elif command == 't':
                        # Test order placement
                        symbol = input("Enter symbol for test order: ").upper().strip()
                        if symbol:
                            result = test_order_placement(symbol)
                            if result:
                                print(f"Test order for {symbol} was successful.")
                            else:
                                print(f"Test order for {symbol} failed. Check logs for details.")
                    
                    elif command == 'm':
                        # NEW: Save ML models on demand
                        if hasattr(state, 'signal_validation_models') and len(state.signal_validation_models) > 0:
                            # Create models directory if needed
                            models_dir = 'ml_models'
                            if not os.path.exists(models_dir):
                                os.makedirs(models_dir)
                            
                            # Save each model
                            saved_count = 0
                            for symbol, model in state.signal_validation_models.items():
                                try:
                                    # Create a safe filename
                                    safe_symbol = symbol.replace('/', '_').replace('\\', '_')
                                    model_path = os.path.join(models_dir, f"{safe_symbol}_model.pkl")
                                    
                                    # Save using pickle
                                    import pickle
                                    with open(model_path, 'wb') as f:
                                        pickle.dump(model, f)
                                    saved_count += 1
                                except Exception as e:
                                    print(f"Error saving model for {symbol}: {e}")
                            
                            print(f"Saved {saved_count} ML models to {models_dir}")
                        else:
                            print("No ML models to save")
                
                except KeyboardInterrupt:
                    raise  # Re-raise to handle at outer level
                except Exception as e:
                    print(f"Error processing command: {e}")
                    
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Stopping trading...")
            state.exit_requested = True
            # Run shutdown handler
            enhanced_shutdown_handler()
        
        # Wait for threads to finish (with timeout)
        print("Waiting for trading operations to complete...")
        
        # Set shorter timeout for better user experience
        timeout_counter = 0
        while (trading_thread.is_alive() or tsl_thread.is_alive()) and timeout_counter < 20:
            time.sleep(1)
            timeout_counter += 1
            if timeout_counter % 5 == 0:
                print(f"Still waiting... ({timeout_counter}s)")
        
        # NEW: Save ML models before exit
        if hasattr(state, 'signal_validation_models') and len(state.signal_validation_models) > 0:
            print("Saving ML models before exit...")
            try:
                models_dir = 'ml_models'
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)
                
                import pickle
                saved_count = 0
                for symbol, model in state.signal_validation_models.items():
                    try:
                        safe_symbol = symbol.replace('/', '_').replace('\\', '_')
                        model_path = os.path.join(models_dir, f"{safe_symbol}_model.pkl")
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)
                        saved_count += 1
                    except:
                        pass
                
                print(f"Saved {saved_count} ML models")
            except Exception as e:
                print(f"Error saving models: {e}")
        
        # Generate final performance report
        report = generate_performance_report()
        print("\nFinal Performance Report:\n" + report)
        
        # Disconnect from MT5
        mt5.shutdown()
        logging.info("Disconnected from MT5")
        
        logging.info("=== Advanced MT5 Trading Bot Stopped ===")
    
    except Exception as e:
        logging.error(f"Critical error in main function: {e}")
        logging.error(traceback.format_exc())
        
        # Run shutdown handler to protect positions
        enhanced_shutdown_handler()
        
        try:
            mt5.shutdown()
        except:
            pass

# if __name__ == "__main__":
#     main_enhanced
if __name__ == "__main__":
    try:
        print("Starting MetaTrader 5 Trading Bot...")
        main_enhanced()
    except KeyboardInterrupt:
        print("\nManually interrupted. Exiting gracefully...")
        try:
            # Ensure the exit flag is set
            if 'state' in globals() and hasattr(state, 'exit_requested'):
                state.exit_requested = True
            # Force shutdown MT5 connection
            mt5.shutdown()
            print("MT5 connection closed.")
        except Exception as shutdown_error:
            print(f"Error during shutdown: {shutdown_error}")
    except Exception as e:
        print(f"\nCritical error: {e}")
        import traceback
        traceback.print_exc()
        try:
            mt5.shutdown()
            print("MT5 connection closed.")
        except:
            pass
        
    print("Application terminated.")