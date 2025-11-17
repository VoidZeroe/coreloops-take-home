"""
CLI script for making revenue predictions.

Usage:
    python -m scripts.predict --customer C00042 --date 2024-10-06
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.predictor import RevenueForecast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Predict next-day revenue for a customer'
    )
    
    parser.add_argument(
        '--customer',
        type=str,
        required=True,
        help='Customer ID (e.g., C00042)'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        required=True,
        help='Prediction date in YYYY-MM-DD format (e.g., 2024-10-06)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='artifacts/model',
        help='Path to model directory (default: artifacts/model)'
    )
    
    parser.add_argument(
        '--metrics-path',
        type=str,
        default='artifacts/daily_customer_metrics.parquet',
        help='Path to metrics file (default: artifacts/daily_customer_metrics.parquet)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output result as JSON'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        # Parse date
        try:
            prediction_date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD", file=sys.stderr)
            return 1
        
        # Check paths exist
        model_dir = Path(args.model_dir)
        metrics_path = Path(args.metrics_path)
        
        if not model_dir.exists():
            print(f"Error: Model directory not found: {model_dir}", file=sys.stderr)
            print("Please run the pipeline first: python -m scripts.run_pipeline", file=sys.stderr)
            return 1
            
        if not metrics_path.exists():
            print(f"Error: Metrics file not found: {metrics_path}", file=sys.stderr)
            print("Please run the pipeline first: python -m scripts.run_pipeline", file=sys.stderr)
            return 1
        
        # Load model and make prediction
        logger.info(f"Loading model from {model_dir}")
        forecast = RevenueForecast(model_dir, metrics_path)
        
        logger.info(f"Making prediction for customer {args.customer} on {args.date}")
        result = forecast.predict_next_day(args.customer, prediction_date)
        
        # Output result
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            # Human-readable output
            print("\n" + "="*60)
            print("REVENUE PREDICTION")
            print("="*60)
            print(f"Customer ID:        {result['customer_id']}")
            print(f"Prediction Date:    {result['prediction_date']}")
            print(f"Predicted Revenue:  £{result['predicted_net_gbp']:.2f}")
            print(f"Confidence:         {result['confidence'].upper()}")
            
            if 'recent_avg_revenue' in result:
                print(f"\nRecent Activity (30 days):")
                print(f"  Average Revenue:  £{result['recent_avg_revenue']:.2f}")
                print(f"  Std Deviation:    £{result['recent_std_revenue']:.2f}")
                
            if 'last_transaction_date' in result:
                print(f"\nLast Transaction:   {result['last_transaction_date']}")
                print(f"Days Since Last:    {result['days_since_last_transaction']}")
                
            if 'message' in result:
                print(f"\nNote: {result['message']}")
                
            print("="*60)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())