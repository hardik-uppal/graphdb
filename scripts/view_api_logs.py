#!/usr/bin/env python3
"""
OpenAI API log viewer and analyzer.
"""

import json
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Any
import re

def parse_log_line(line: str) -> Dict[str, Any]:
    """Parse a single log line into structured data."""
    # Format: timestamp - logger_name - level - message
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (\w+) - (.+)'
    match = re.match(pattern, line)
    
    if not match:
        return None
    
    timestamp_str, logger_name, level, message = match.groups()
    
    # Parse timestamp
    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
    
    # Try to parse JSON message
    try:
        if message.startswith(('EMBEDDING_', 'CHAT_COMPLETION_')):
            # Extract the type and JSON data
            if ': ' in message:
                msg_type, json_data = message.split(': ', 1)
                data = json.loads(json_data)
                return {
                    'timestamp': timestamp,
                    'logger': logger_name,
                    'level': level,
                    'type': msg_type,
                    'data': data
                }
    except json.JSONDecodeError:
        pass
    
    return {
        'timestamp': timestamp,
        'logger': logger_name,
        'level': level,
        'message': message
    }

def load_logs(log_file: str) -> List[Dict[str, Any]]:
    """Load and parse log file."""
    logs = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parsed = parse_log_line(line)
                if parsed:
                    logs.append(parsed)
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return []
    
    return logs

def analyze_embedding_logs(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze embedding-related logs."""
    embedding_requests = []
    embedding_responses = []
    embedding_errors = []
    
    for log in logs:
        if log.get('type') == 'EMBEDDING_REQUEST':
            embedding_requests.append(log)
        elif log.get('type') == 'EMBEDDING_RESPONSE':
            embedding_responses.append(log)
        elif log.get('type') in ['EMBEDDING_ERROR', 'EMBEDDING_BATCH_ERROR']:
            embedding_errors.append(log)
    
    # Calculate statistics
    total_requests = len(embedding_requests)
    total_responses = len(embedding_responses)
    total_errors = len(embedding_errors)
    
    # Calculate token usage
    total_tokens = 0
    for resp in embedding_responses:
        usage = resp.get('data', {}).get('usage', {})
        if isinstance(usage.get('total_tokens'), int):
            total_tokens += usage['total_tokens']
    
    # Calculate success rate
    success_rate = (total_responses / total_requests * 100) if total_requests > 0 else 0
    
    return {
        'total_requests': total_requests,
        'total_responses': total_responses,
        'total_errors': total_errors,
        'success_rate': success_rate,
        'total_tokens': total_tokens,
        'recent_errors': embedding_errors[-5:] if embedding_errors else []
    }

def analyze_chat_logs(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze chat completion logs."""
    chat_requests = []
    chat_responses = []
    chat_errors = []
    
    for log in logs:
        if log.get('type') == 'CHAT_COMPLETION_REQUEST':
            chat_requests.append(log)
        elif log.get('type') == 'CHAT_COMPLETION_RESPONSE':
            chat_responses.append(log)
        elif log.get('type') == 'CHAT_COMPLETION_ERROR':
            chat_errors.append(log)
    
    # Calculate statistics
    total_requests = len(chat_requests)
    total_responses = len(chat_responses)
    total_errors = len(chat_errors)
    
    # Calculate token usage
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    
    for resp in chat_responses:
        usage = resp.get('data', {}).get('usage', {})
        if isinstance(usage.get('total_tokens'), int):
            total_tokens += usage['total_tokens']
        if isinstance(usage.get('prompt_tokens'), int):
            prompt_tokens += usage['prompt_tokens']
        if isinstance(usage.get('completion_tokens'), int):
            completion_tokens += usage['completion_tokens']
    
    # Function call statistics
    function_calls = []
    for resp in chat_responses:
        if resp.get('data', {}).get('has_tool_calls'):
            tool_calls = resp.get('data', {}).get('tool_calls', [])
            function_calls.extend([tc.get('function_name') for tc in tool_calls])
    
    function_usage = {}
    for func in function_calls:
        function_usage[func] = function_usage.get(func, 0) + 1
    
    success_rate = (total_responses / total_requests * 100) if total_requests > 0 else 0
    
    return {
        'total_requests': total_requests,
        'total_responses': total_responses,
        'total_errors': total_errors,
        'success_rate': success_rate,
        'total_tokens': total_tokens,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'function_calls': function_usage,
        'recent_errors': chat_errors[-5:] if chat_errors else []
    }

def print_summary(logs: List[Dict[str, Any]]):
    """Print summary of API usage."""
    if not logs:
        print("No logs found.")
        return
    
    print(f"=== OpenAI API Log Summary ===")
    print(f"Total log entries: {len(logs)}")
    print(f"Time range: {logs[0]['timestamp']} to {logs[-1]['timestamp']}")
    
    # Analyze embeddings
    embedding_stats = analyze_embedding_logs(logs)
    print(f"\n=== Embedding API Usage ===")
    print(f"Requests: {embedding_stats['total_requests']}")
    print(f"Responses: {embedding_stats['total_responses']}")
    print(f"Errors: {embedding_stats['total_errors']}")
    print(f"Success rate: {embedding_stats['success_rate']:.1f}%")
    print(f"Total tokens: {embedding_stats['total_tokens']}")
    
    # Analyze chat completions
    chat_stats = analyze_chat_logs(logs)
    print(f"\n=== Chat Completion API Usage ===")
    print(f"Requests: {chat_stats['total_requests']}")
    print(f"Responses: {chat_stats['total_responses']}")
    print(f"Errors: {chat_stats['total_errors']}")
    print(f"Success rate: {chat_stats['success_rate']:.1f}%")
    print(f"Total tokens: {chat_stats['total_tokens']}")
    print(f"Prompt tokens: {chat_stats['prompt_tokens']}")
    print(f"Completion tokens: {chat_stats['completion_tokens']}")
    
    if chat_stats['function_calls']:
        print(f"\n=== Function Calls ===")
        for func, count in sorted(chat_stats['function_calls'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {func}: {count}")
    
    # Show recent errors
    all_errors = embedding_stats['recent_errors'] + chat_stats['recent_errors']
    if all_errors:
        print(f"\n=== Recent Errors ===")
        for error in all_errors[-5:]:
            print(f"  {error['timestamp']}: {error.get('data', {}).get('error', 'Unknown error')}")

def print_detailed_logs(logs: List[Dict[str, Any]], log_type: str = None):
    """Print detailed log entries."""
    filtered_logs = logs
    
    if log_type:
        filtered_logs = [log for log in logs if log.get('type', '').startswith(log_type.upper())]
    
    for log in filtered_logs:
        print(f"\n=== {log['timestamp']} - {log.get('type', 'LOG')} ===")
        
        if 'data' in log:
            print(json.dumps(log['data'], indent=2, default=str))
        else:
            print(log.get('message', 'No message'))

def tail_logs(log_file: str, lines: int = 10):
    """Show last N lines of logs."""
    logs = load_logs(log_file)
    
    if not logs:
        print("No logs found.")
        return
    
    recent_logs = logs[-lines:]
    print(f"=== Last {len(recent_logs)} log entries ===")
    
    for log in recent_logs:
        timestamp = log['timestamp'].strftime('%H:%M:%S')
        log_type = log.get('type', 'LOG')
        
        if 'data' in log:
            # Show key information for each log type
            data = log['data']
            if log_type.startswith('EMBEDDING_'):
                if 'error' in data:
                    print(f"[{timestamp}] {log_type}: ERROR - {data['error']}")
                elif 'batch_size' in data:
                    print(f"[{timestamp}] {log_type}: Batch size {data['batch_size']}")
                else:
                    print(f"[{timestamp}] {log_type}: {data.get('input_text', '')[:50]}...")
            elif log_type.startswith('CHAT_'):
                if 'error' in data:
                    print(f"[{timestamp}] {log_type}: ERROR - {data['error']}")
                elif 'user_query' in data:
                    print(f"[{timestamp}] {log_type}: Query - {data['user_query']}")
                else:
                    print(f"[{timestamp}] {log_type}: Success")
        else:
            print(f"[{timestamp}] {log_type}: {log.get('message', '')}")

def main():
    parser = argparse.ArgumentParser(description="OpenAI API log analyzer")
    parser.add_argument("--log-file", default="openai_api.log", help="Log file to analyze")
    parser.add_argument("--summary", action="store_true", help="Show summary statistics")
    parser.add_argument("--detailed", action="store_true", help="Show detailed logs")
    parser.add_argument("--type", help="Filter by log type (embedding, chat)")
    parser.add_argument("--tail", type=int, help="Show last N log entries")
    
    args = parser.parse_args()
    
    # Default action is summary
    if not any([args.summary, args.detailed, args.tail]):
        args.summary = True
    
    logs = load_logs(args.log_file)
    
    if args.tail:
        tail_logs(args.log_file, args.tail)
    elif args.summary:
        print_summary(logs)
    elif args.detailed:
        print_detailed_logs(logs, args.type)

if __name__ == "__main__":
    main()
