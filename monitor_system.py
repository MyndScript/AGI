#!/usr/bin/env python3
"""
AGI System Monitor - Enhanced Version
Monitors all AGI services with detailed status information
"""
import psutil
import requests
import time
import sys
import socket
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

SERVICES = {
    "Overseer Gateway": {
        "url": "http://localhost:8010",
        "port": 8010,
        "description": "Central API router",
        "health_endpoint": "/health"
    },
    "Agent Server": {
        "url": "http://localhost:8000",
        "port": 8000,
        "description": "Python FastAPI agent service",
        "health_endpoint": "/health"
    },
    "Memory Server": {
        "url": "http://localhost:8001",
        "port": 8001,
        "description": "Go Gin memory service",
        "health_endpoint": "/health"
    },
    "Personality Server": {
        "url": "http://localhost:8002",
        "port": 8002,
        "description": "Python FastAPI personality service",
        "health_endpoint": "/health"
    },
    "Embedding Server": {
        "url": "http://localhost:8003",
        "port": 8003,
        "description": "Python SentenceTransformers embedding service",
        "health_endpoint": "/health"
    },
    "Legacy Embedding Server": {
        "url": "http://localhost:8004",
        "port": 8004,
        "description": "Go embedding service (deprecated)",
        "health_endpoint": "/health"
    },
    "UI Frontend": {
        "url": "http://localhost:3000",
        "port": 3000,
        "description": "React UI frontend",
        "health_endpoint": "/"
    }
}

def check_service(name, config):
    """Check if a service is responding with detailed information"""
    try:
        start_time = time.time()
        response = requests.get(f"{config['url']}{config['health_endpoint']}", timeout=5)
        latency = (time.time() - start_time) * 1000

        if response.status_code == 200:
            return "✅ UP", f"{latency:.1f}ms", f"HTTP {response.status_code}", None
        else:
            return f"⚠️  HTTP {response.status_code}", f"{latency:.1f}ms", f"HTTP {response.status_code}", None
    except requests.exceptions.Timeout:
        return "⏰ TIMEOUT", ">5000ms", "Timeout", "Service took too long to respond"
    except requests.exceptions.ConnectionError:
        return "❌ DOWN", "N/A", "Connection Error", "Cannot connect to service"
    except Exception as e:
        return f"❓ ERROR", "N/A", "Exception", f"{str(e)}"

def check_port(port):
    """Check if a port is open with detailed information"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()

        if result == 0:
            # Get process information for the port
            if PSUTIL_AVAILABLE:
                try:
                    for conn in psutil.net_connections():
                        if conn.laddr and conn.laddr.port == port and conn.status == 'LISTEN':
                            if conn.pid:
                                process = psutil.Process(conn.pid)
                                return True, f"PID {conn.pid} ({process.name()})"
                except Exception:
                    pass
            return True, "Active"
        else:
            return False, "Closed"
    except Exception as e:
        return False, f"Error: {str(e)}"

def get_system_info():
    """Get basic system information"""
    if PSUTIL_AVAILABLE:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            return f"CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%"
        except Exception:
            pass
    return "System info unavailable"

def main():
    print("🔍 AGI System Monitor - Enhanced")
    print("=" * 80)
    print(f"{'Service':<20} {'Status':<12} {'Latency':<10} {'Port':<8} {'Process':<15} {'Details'}")
    print("=" * 80)

    while True:
        print(f"\n📊 Status at {datetime.now().strftime('%H:%M:%S')} | {get_system_info()}")
        print("-" * 80)

        all_healthy = True
        service_count = len(SERVICES)
        healthy_count = 0

        for name, config in SERVICES.items():
            port_open, port_info = check_port(config['port'])

            if port_open:
                status, latency, http_status, error_details = check_service(name, config)
                if "✅" in status:
                    healthy_count += 1
                else:
                    all_healthy = False

                details = error_details if error_details else config['description']
                print(f"{name:<20} {status:<12} {latency:<10} {config['port']:<8} {port_info:<15} {details}")
            else:
                all_healthy = False
                print(f"{name:<20} {'❌ DOWN':<12} {'N/A':<10} {config['port']:<8} {port_info:<15} Port closed")

        print("-" * 80)
        if all_healthy:
            print(f"🎉 All {service_count} services operational! ({healthy_count}/{service_count} healthy)")
        else:
            print(f"⚠️  {healthy_count}/{service_count} services healthy - Some systems need attention.")

        print("\nPress Ctrl+C to exit...")
        time.sleep(10)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped.")
        sys.exit(0)
