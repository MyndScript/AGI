#!/usr/bin/env python3
"""
Quick test of monitor system formatting
"""
import sys
sys.path.append('.')
from monitor_system import *

print('🔍 AGI System Monitor - Test')
print('=' * 80)
print(f"{'Service':<20} {'Status':<12} {'Latency':<10} {'Port':<8} {'Process':<15} {'Details'}")
print('=' * 80)

# Test one service
name = 'Overseer Gateway'
config = SERVICES[name]
port_open, port_info = check_port(config['port'])

if port_open:
    status, latency, http_status, error_details = check_service(name, config)
    details = error_details if error_details else config['description']
    print(f"{name:<20} {status:<12} {latency:<10} {config['port']:<8} {port_info:<15} {details}")
else:
    print(f"{name:<20} {'❌ DOWN':<12} {'N/A':<10} {config['port']:<8} {port_info:<15} {'Port closed'}")
