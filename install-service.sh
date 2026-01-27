#!/bin/bash

# WhisperX Gradio Service Installation Script
set -e

echo "🎙️ Installing WhisperX Gradio System Service..."

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run as root or with sudo"
    echo "💡 Run: sudo ./install-service.sh"
    exit 1
fi

# Get the actual username (not root when using sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
SERVICE_NAME="whisperx-gradio"

echo "📁 Current directory: $(pwd)"
echo "👤 Service will run as user: $ACTUAL_USER"

# Check if docker compose exists
if ! docker compose version &> /dev/null; then
    echo "❌ docker compose not found"
    echo "💡 Please install Docker Compose first"
    exit 1
fi

# Update the service file with correct paths and user
SERVICE_FILE="$SERVICE_NAME.service"
WORKING_DIR=$(pwd)

# Create the service file with correct paths
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=WhisperX Gradio Service
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$WORKING_DIR
ExecStartPre=/usr/bin/docker compose build
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
ExecReload=/usr/bin/docker compose restart
User=$ACTUAL_USER
Group=$ACTUAL_USER
Restart=on-failure
RestartSec=10
Environment="PATH=/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=multi-user.target
EOF

# Copy service file to systemd directory
echo "📋 Installing service file..."
cp "$SERVICE_FILE" /etc/systemd/system/

# Set proper permissions
chmod 644 /etc/systemd/system/$SERVICE_FILE

# Reload systemd daemon
echo "🔄 Reloading systemd daemon..."
systemctl daemon-reload

# Enable the service for startup
echo "✅ Enabling service for system startup..."
systemctl enable $SERVICE_NAME

echo ""
echo "✅ Service installation completed!"
echo ""
echo "📋 Available commands:"
echo "   Start service:    sudo systemctl start $SERVICE_NAME"
echo "   Stop service:     sudo systemctl stop $SERVICE_NAME"
echo "   Restart service:  sudo systemctl restart $SERVICE_NAME"
echo "   Service status:   sudo systemctl status $SERVICE_NAME"
echo "   View logs:        sudo journalctl -u $SERVICE_NAME -f"
echo "   Disable startup:  sudo systemctl disable $SERVICE_NAME"
echo ""
echo "🚀 The service is now configured to start automatically at boot!"
echo "🌐 After starting, access the application at: http://localhost:7861"
