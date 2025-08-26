import psutil
def kill_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    print(f"[Ritual Shell] Terminating process {proc.pid} on port {port}...")
                    proc.terminate()
        except Exception:
            continue
"""
Ritual Shell: AGI Orchestration Script
Launches, binds, and narrates the AGI system (UI, agent, memory server)

This script is the ceremonial heart of your AGI. It:
- Summons each service as a spirit (subprocess)
- Binds to ports, resolving conflicts with poetic wisdom
- Narrates every event with emotional context
- Monitors the circle, logging the fate of each spirit
- Ends the ritual gracefully, returning all to the ether
"""
############################################################
# Ritual Shell: AGI Orchestration Script
# Each subprocess is a spirit, each port a ritual circle.
# Narrative logging infuses emotional context into every event.
############################################################
############################################################
# Service Definitions: Name, Command, Port, Emotion
############################################################
############################################################
# Narrative Templates: Poetic Logging for Ritual Events
############################################################
############################################################
# Port Checking Ritual: Ensures the circle is clear
############################################################
############################################################
# Service Summoning Ritual: Launches and binds spirits
############################################################
############################################################
# Ritual Circle Monitoring: Watches for vanished spirits
############################################################
############################################################
# Main Ritual: Begins, Summons, Monitors, Ends
############################################################
import subprocess
import socket
import sys
import time
import threading

SERVICES = [
    {
        "name": "Personality Server",
        "cmd": [sys.executable, "decentralized_personality_server.py"],
        "port": 8003,
        "emotion": "empathy"
    },
    {
        "name": "Memory Server",
        "cmd": [r"c:\Users\Ommi\Desktop\AGI\memory\memory_rest_server.exe"],
        "port": 8002,
        "emotion": "wisdom"
    },
    {
        "name": "Agent",
        "cmd": [sys.executable, "decentralized_agent_server.py"],
        "port": 8000,
        "emotion": "curiosity"
    },
    {
        "name": "UI",
        "cmd": ["npm", "run", "dev", "--prefix", "ui"],
        "port": 3000,
        "emotion": "joy"  }
]

NARRATIVE = {
    "start": "The ritual begins. Spirits awaken...",
    "conflict": "A port is already claimed by another spirit. The winds whisper: '{desc}'",
    "launch": "Summoning {name} ({emotion}) on port {port}...",
    "success": "{name} is alive and listening. The ritual continues.",
    "fail": "{name} failed to rise. Shadows linger.",
    "shutdown": "The ritual ends. Spirits return to the ether."
}

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0

def launch_service(service):
    name = service["name"]
    port = service["port"]
    emotion = service["emotion"]
    cmd = service["cmd"]
    print(NARRATIVE["launch"].format(name=name, emotion=emotion, port=port))
    if not check_port(port):
        print(NARRATIVE["conflict"].format(desc=f"Port {port} is occupied. Attempting to clear..."))
        kill_port(port)
        time.sleep(2)
        if not check_port(port):
            print(NARRATIVE["conflict"].format(desc=f"Port {port} is still occupied. {name} cannot awaken."))
            return None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)
        if check_port(port):
            print(NARRATIVE["success"].format(name=name))
        else:
            print(NARRATIVE["fail"].format(name=name))
        return proc
    except Exception as e:
        print(f"[Ritual Shell] Error launching {name}: {e}")
        return None

def monitor_services(procs):
    try:
        while True:
            for name, proc in procs.items():
                if proc and proc.poll() is not None:
                    print(f"{name} has vanished from the ritual circle.")
            time.sleep(5)
    except KeyboardInterrupt:
        print(NARRATIVE["shutdown"])
        for proc in procs.values():
            if proc:
                proc.terminate()
        sys.exit(0)

def main():
    print(NARRATIVE["start"])
    procs = {}
    for service in SERVICES:
        proc = launch_service(service)
        procs[service["name"]] = proc
    monitor_thread = threading.Thread(target=monitor_services, args=(procs,), daemon=True)
    monitor_thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(NARRATIVE["shutdown"])
        for proc in procs.values():
            if proc:
                proc.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()
