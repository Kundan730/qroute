# Deploying qroute on any small VPS

Works on Hetzner, DigitalOcean, Contabo, Linode, or anything else that gives you
an Ubuntu box with root. Oracle's free tier has two quirks of its own and has a
separate guide at `deploy/oracle/DEPLOY.md`; everything here otherwise applies
to that too.

## What to buy

The service needs about 410 MB with a city loaded and roughly 3 GB of disk for
the image and data. **2 GB of memory is the sensible floor** - not because the
service uses it, but because the image build compiles wheels and a 1 GB box
will have the build killed partway through.

| Provider | Plan | Cost | Notes |
| --- | --- | --- | --- |
| Hetzner | CAX11, 2 vCPU Arm, 4 GB | about €4/month | Best value by a distance. Arm, which this project supports. |
| Hetzner | CX22, 2 vCPU x86, 4 GB | about €4/month | If you would rather stay on x86. |
| DigitalOcean | Basic, 1 vCPU, 2 GB | $12/month | More expensive, very well documented. |
| Contabo | VPS S | about €5/month | Cheap and generous, mixed reputation for support. |

Arm is fine: every dependency this project has, including Numba, llvmlite,
OR-Tools, SciPy and PyVRP, publishes an `aarch64` wheel. That was verified, not
assumed.

## 1. Create the server

Ubuntu 22.04 or 24.04, add your SSH public key during creation rather than
using a root password. Note the IP.

## 2. Connect and prepare

```bash
ssh root@YOUR_SERVER_IP
```

```bash
adduser --disabled-password --gecos "" qroute
usermod -aG sudo qroute
rsync --archive --chown=qroute:qroute ~/.ssh /home/qroute/
```

From here on, work as that user: `ssh qroute@YOUR_SERVER_IP`.

## 3. Install Docker

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker qroute && newgrp docker
```

## 4. Add swap before building

Two cores compiling SciPy and OR-Tools wheels can exhaust 2 GB and have the
build killed with no useful message. Swap costs nothing and prevents it:

```bash
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 5. Clone and build

```bash
git clone https://github.com/Kundan730/qroute.git
cd qroute
docker compose build
```

Fifteen to twenty-five minutes. The final step solves a benchmark instance so
the compiled kernels are baked in and your first visitor does not wait for them.

## 6. Publish it

The compose file binds `127.0.0.1:8000` on purpose, so a box with Docker on it
does not start serving the world by accident. Exposing it is a deliberate step.
Create `docker-compose.override.yml`:

```yaml
services:
  qroute:
    ports: !override
      - "80:8000"
    environment:
      QROUTE_API_PRELOAD: first
      QROUTE_MAX_ACTIVE_RUNS: "3"
```

`!override` needs Compose v2.24+ (`docker compose version`). Without it, drop
the tag and both mappings coexist, which works.

```bash
docker compose up -d
```

## 7. Firewall

Most VPS providers leave the host open and give you a cloud firewall in their
console. If you use `ufw` on the box:

```bash
sudo ufw allow OpenSSH && sudo ufw allow 80/tcp && sudo ufw --force enable
```

## 8. Road networks

```bash
docker compose exec qroute qroute osm fetch
```

About a minute per city. They land on a named volume and survive rebuilds.

## 9. Check

```bash
curl http://YOUR_SERVER_IP/api/health
```

Then open the address in a browser. The map should draw, the time-of-day slider
should recolour the roads, and **Generate instance** then **Optimise** should
return routes in about ten seconds.

## Optional: domain and HTTPS

Point an A record at the IP and put Caddy in front; it gets a certificate
automatically. Add to the override file:

```yaml
  caddy:
    image: caddy:2-alpine
    restart: unless-stopped
    ports: ["80:80", "443:443"]
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
volumes:
  caddy_data:
```

```
your-domain.example { reverse_proxy qroute:8000 }
```

Remove the `80:8000` mapping from the qroute service so Caddy can take the port.

## When it does not work

**Nothing loads.** Check from the box first: `curl localhost:80/api/health`. If
that works and the public IP does not, it is a firewall, not the application.

**The build is killed.** Memory. Step 4.

**No roads on the map.** Step 8 has not run. The Solver and Benchmark pages work
regardless, since the benchmark instances ship inside the image.

**First solve slow, later ones fast.** Numba keys its cache to the CPU it
compiled on. Expected once, never an error.
