# AGI System Launch & Management Guide

## 🎯 Best Practices Overview

This AGI system uses a **microservices architecture** with centralized orchestration through the **Overseer Gateway**. Here are the recommended approaches for launching and managing the system.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐
│   Overseer      │────│   Frontend/UI    │
│   Gateway       │    │   (Port 3000)    │
│   (Port 8010)   │    └──────────────────┘
└─────────────────┘             │
         │                      │
         └──────────────────────┼──────────────────────┐
                                │                       │
                   ┌────────────┴────────────┐ ┌────────┴────────┐
                   │    Agent Server        │ │ Memory Server   │
                   │    (Port 8000)         │ │ (Port 8001)      │
                   └─────────────────────────┘ └─────────────────┘
                                │                       │
                   ┌────────────┴────────────┐ ┌────────┴────────┐
                   │ Personality Server     │ │ Embedding       │
                   │ (Port 8002)            │ │ Server          │
                   └─────────────────────────┘ │ (Port 8004)     │
                                               └─────────────────┘
```

## 🚀 Launch Methods

### 1. **Production Launch** (Recommended)
```powershell
# Full production launch with monitoring
.\launch_production.ps1

# Options:
# -CleanStart    # Clear all ports before starting
# -NoUI         # Skip UI launch
# -Verbose      # Show detailed output
```

### 2. **Development Launch**
```powershell
# Simple development launch
.\launch_all.ps1
```

### 3. **Orchestrated Launch**
```bash
# Python-based orchestration with monitoring
python ritual_shell.py
```

## 📊 Monitoring

### System Health Monitor
```bash
python monitor_system.py
```

### Individual Service Checks
```bash
# Check specific service
curl http://localhost:8010/health  # Overseer
curl http://localhost:8000/docs    # Agent (FastAPI docs)
curl http://localhost:8001/health  # Memory
curl http://localhost:8002/docs    # Personality (FastAPI docs)
```

## 🔧 Service Configuration

### Environment Variables
```bash
# Core Services
AGI_AGENT_PORT=8000
AGI_MEMORY_PORT=8001
AGI_PERSONALITY_PORT=8002
AGI_OVERSEER_PORT=8010

# Optional
PYTHONPATH=/path/to/agi
```

### Port Assignments
- **8010**: Overseer Gateway (Central Router)
- **8000**: Agent Server (FastAPI)
- **8001**: Memory Server (Go/Gin)
- **8002**: Personality Server (FastAPI)
- **8004**: Embedding Server (Go)
- **3000**: UI (React/esbuild)

## 🏆 Recommended Approach

### For **Production**:
1. Use `launch_production.ps1` - combines PowerShell reliability with comprehensive monitoring
2. Always use `-CleanStart` for fresh deployments
3. Monitor with `python monitor_system.py`

### For **Development**:
1. Use `launch_all.ps1` - simple and fast
2. Use VS Code terminals for debugging
3. Check individual service logs

### For **Orchestration**:
1. Use `ritual_shell.py` - poetic monitoring and process management
2. Best for long-running deployments
3. Includes graceful shutdown

## 🔄 API Flow

All client requests should go through the **Overseer Gateway**:

```javascript
// ✅ Correct - Through Overseer
fetch('http://localhost:8010/generate', { ... })

// ❌ Avoid - Direct service access
fetch('http://localhost:8000/generate', { ... })
```

## 🛠️ Troubleshooting

### Port Conflicts
```powershell
# Kill process on specific port
Get-NetTCPConnection -LocalPort 8000 -State Listen | Stop-Process -Id {$_.OwningProcess} -Force
```

### Service Won't Start
1. Check if port is free: `netstat -ano | findstr :8000`
2. Kill conflicting process
3. Check service logs
4. Verify dependencies are installed

### Memory Issues
- Go services use compiled executables (`.exe`)
- Python services need virtual environment
- Monitor memory usage with Task Manager

## 📋 Checklist

### Pre-Launch
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Compile Go services: `cd memory && go build -o *.exe *.go`
- [ ] Set environment variables
- [ ] Clear ports if needed

### Post-Launch
- [ ] Verify all services are running
- [ ] Test Overseer Gateway: `curl http://localhost:8010`
- [ ] Test individual services
- [ ] Start monitoring: `python monitor_system.py`

### Maintenance
- [ ] Monitor logs regularly
- [ ] Update dependencies periodically
- [ ] Backup configuration files
- [ ] Test failover scenarios

## 🎯 Key Benefits

1. **Centralized Routing**: Overseer Gateway provides single entry point
2. **Service Isolation**: Each service runs independently
3. **Easy Scaling**: Add/remove services without affecting others
4. **Monitoring**: Comprehensive health checking
5. **Graceful Shutdown**: Clean termination of all services
6. **Development Friendly**: Multiple launch options for different needs
