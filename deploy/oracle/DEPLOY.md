# Deploying qroute on Oracle Cloud Always Free

The whole process, from an empty Oracle account to a public URL. Budget about
an hour, most of which is waiting for the image to build on the server.

**Cost: nothing, permanently.** Oracle asks for a card to verify identity and
does not charge for Always Free resources. Set the account to stay on the free
tier when you sign up and it cannot bill you.

**Why this rather than the alternatives.** The Always Free allowance is four Arm
cores and 24 GB of memory. The service needs roughly 410 MB with a city loaded,
so there is a great deal of room, and unlike a free tier that sleeps, the machine
stays up. Every dependency this project has, including the awkward ones - Numba,
llvmlite, OR-Tools, SciPy and PyVRP - publishes an `aarch64` wheel, so the Arm
architecture costs nothing. That was verified before writing this guide, not
assumed.

---

## 1. Create the account

<https://signup.oraclecloud.com>. Pick the region closest to you and **remember
which one** - resources live in a region and the console silently shows you only
the one you are in. For India, Mumbai or Hyderabad.

At the end, do **not** upgrade to Pay As You Go. An Always Free account cannot
charge you.

## 2. Create the instance

Console → **Compute → Instances → Create instance**.

| Field | Value |
| --- | --- |
| Name | `qroute` |
| Image | **Ubuntu 22.04** (Canonical) |
| Shape | **Ampere VM.Standard.A1.Flex** |
| OCPUs | 2 |
| Memory | 12 GB |
| Boot volume | 50 GB |
| SSH keys | Paste your public key, or let Oracle generate one and download it |

Two OCPUs and 12 GB is half the free allowance, which leaves room for a second
machine later. Taking all four cores is fine too.

### If it says "Out of host capacity"

This is the one genuinely annoying part of Oracle's free tier and it is not
your mistake. Ampere capacity is scarce in popular regions. In order of
effectiveness:

1. Try a different availability domain in the same region (the dropdown above
   the shape).
2. Try again at a different time of day. Capacity is released continuously.
3. Create the account in a less busy region.

An AMD `VM.Standard.E2.1.Micro` is also Always Free but has 1 GB of memory,
which is not enough for this service. Wait for Arm capacity rather than
settling for it.

## 3. Open the port, in both places

Oracle blocks traffic at two independent layers and forgetting the second is
the most common reason a deployment appears dead.

**Layer one, the cloud firewall.** Instance page → **Virtual cloud network** →
**Security lists** → the default list → **Add ingress rule**:

| Field | Value |
| --- | --- |
| Source CIDR | `0.0.0.0/0` |
| IP protocol | TCP |
| Destination port | `80` |

**Layer two, the host firewall.** Ubuntu images on Oracle ship with iptables
rules that drop everything but SSH. Over SSH:

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo netfilter-persistent save
```

## 4. Connect and install Docker

```bash
ssh -i /path/to/your/private/key ubuntu@YOUR_INSTANCE_IP
```

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker ubuntu && newgrp docker
```

Check it: `docker run --rm hello-world`.

## 5. Get the project and build

```bash
git clone https://github.com/Kundan730/qroute.git
cd qroute
docker compose build
```

The build takes fifteen to twenty-five minutes on two Arm cores. Most of it is
installing SciPy, OR-Tools and Numba; the last step solves a real benchmark
instance so the compiled kernels are baked into the image and your first
visitor does not wait for them.

## 6. Serve on port 80

The compose file deliberately publishes `127.0.0.1:8000:8000` - loopback only,
so that a machine which happens to have Docker installed does not find itself
serving the world by accident. Exposing it is a decision you make explicitly,
which is what this step is.

Create `docker-compose.override.yml` next to the compose file:

```yaml
services:
  qroute:
    ports: !override
      - "80:8000"
    environment:
      QROUTE_API_PRELOAD: first
      QROUTE_MAX_ACTIVE_RUNS: "3"
```

The `!override` tag replaces the base file's port list rather than adding to
it, so the loopback mapping goes away and only port 80 is published. It needs
Compose v2.24 or newer; check with `docker compose version`. On an older
version, drop the tag - both mappings then coexist, which works but leaves the
service reachable on 8000 as well, so close that port in the Oracle security
list if you go that way.

`preload: first` loads one city at startup rather than all three, so the service
is answering in a few seconds instead of half a minute. The others load when
someone selects them.

```bash
docker compose up -d
docker compose logs -f
```

## 7. Fetch the road networks

They are about 40 MB and are not in the image, by design - they are
regenerable, and shipping them would bloat every deploy.

```bash
docker compose exec qroute qroute osm fetch
```

About a minute per city, and it needs outbound internet, which the instance has.
They land on the named volume, so they survive a rebuild.

## 8. Check it

```bash
curl http://YOUR_INSTANCE_IP/api/health
```

A healthy reply says `"status": "ok"` with the instance and network counts. Open
`http://YOUR_INSTANCE_IP` in a browser: the map should draw, the time-of-day
slider should recolour the roads, and **Generate instance** then **Optimise**
should produce routes in about ten seconds.

## 9. Survive a reboot

```bash
sudo systemctl enable docker
```

The compose file already sets a restart policy, so the container comes back on
its own.

---

## Optional: a real domain and HTTPS

Judges will click an `http://` IP address quite happily, so this is optional.
If you want a name and a padlock, point a domain's A record at the instance and
put Caddy in front, which obtains a certificate automatically:

```yaml
# add to docker-compose.override.yml
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
# Caddyfile
your-domain.example { reverse_proxy qroute:8000 }
```

Remember to open 443 in the Oracle security list as well as 80, and remove the
`80:8000` mapping from the qroute service so Caddy can take the port.

---

## When it does not work

**The page never loads and there is no error.** Almost always the host
firewall in step 3, layer two. Confirm from the instance itself with
`curl localhost:80/api/health`: if that works and the public IP does not, it is
a firewall, not the app.

**The build is killed partway through.** Two Arm cores compiling wheels can
exhaust memory if you chose a small shape. `free -h` while it builds. Give the
instance more memory, or add swap:
`sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`.

**The map has no roads.** Step 7 has not been run, or it failed. Check with
`docker compose exec qroute ls data/osm`. The Solver and Benchmark pages work
regardless, since the benchmark instances ship in the image.

**The first solve is slow, later ones are fast.** Numba keys its compilation
cache to the CPU it compiled on. This is the expected one-off cost and never an
error.

**Everything is slow under several visitors.** Lower `QROUTE_MAX_ACTIVE_RUNS`,
or give the instance the other two free cores.
